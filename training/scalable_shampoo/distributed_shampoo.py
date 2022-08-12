# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# An implementation of distributed Shampoo optimizer from:
#
#  Scalable Second Order Optimization for Deep Learning
#  Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer
#  Preprint Paper: https://arxiv.org/abs/2002.09018
#
# This implementation moves computation of inverse pth root back to the
# accelerator (if higher precision is available).
#
# Authors: Rohan Anil (rohananil at google dot com)
#          Vineet Gupta (vineet at google dot com)
#          James Lottes (jlottes at google dot com)
#          Anudhyan Boral (anudhyan at google dot com)
#
"""Distributed Shampoo Implementation."""

import enum
import functools
import itertools
from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from flax import struct
from jax import lax
from jax.experimental import pjit
from jax.experimental.sparse import linalg

from .quantization_utils import QuantizedValue
from .symmetric_matrices import symmetric_matrices

# Dtype for inverse-pth root routine
# Switch to f64 if you have hardware that supports it. Enable the jax flag
# jax_enable_x64 for this to work, otherwise it will default to float32.
_MAT_INV_PTH_ROOT_DTYPE = jnp.float64


@struct.dataclass
class TrainingMetrics:
    inverse_pth_root_errors: chex.Array  # Error for inverse-pth roots.
    # TODO(rohananil): Add more important metrics to track during training.


# Per parameter optimizer state used in data-parallel training.
class ParameterStats(NamedTuple):
    """State associated to each parameter of the model being trained."""

    diagonal_statistics: QuantizedValue  # Accumulator for diagonal preconditioner
    statistics: List[Any]  # Statistics (QuantizedValue, chex.Array)
    preconditioners: List[Any]  # Preconditioners (QuantizedValue, chex.Array)
    diagonal_momentum: QuantizedValue  # Momentum for the diagonal preconditioner
    momentum: QuantizedValue  # Momentum for the shampoo preconditioner
    training_metrics: TrainingMetrics  # Metrics (optional for training).


# For training extremely large model; We keep a global state with a concatenated
# statistics and preconditioner states for all vars. This is so that we can
# annotate the leading axis to be sharded to save memory at the cost of
# communication.
@struct.dataclass
class GlobalShardedParameterStats:
    statistics: chex.Array  # Statistics
    preconditioners: chex.Array  # Preconditioners
    exponents: chex.Array  # exponents


# These are per-parameter local states; All statistics here mirror the parameter
# Thus the sharding is copied over from the param specification.
@struct.dataclass
class LocalShardedParameterStats:
    """State associated to each parameter of the model being trained."""

    diagonal_statistics: QuantizedValue  # Accumulator for diagonal preconditioner
    diagonal_momentum: QuantizedValue  # Momentum for the diagonal preconditioner
    momentum: QuantizedValue  # Momentum for the shampoo preconditioner
    training_metrics: TrainingMetrics  # Metrics (optional for training).
    index_start: np.int32 = struct.field(pytree_node=False)  # Index into global statistics array
    sizes: Any = struct.field(pytree_node=False)  # Sizes of the statistics.


def init_training_metrics(num_statistics):
    # Since the downstream apis expect a jnp.array - we create a dummy one if
    # num_statistics=0.
    if not num_statistics:
        return TrainingMetrics(jnp.array(0, jnp.float32))
    else:
        return TrainingMetrics(jnp.zeros([num_statistics], jnp.float32))


def init_training_metrics_shapes(num_statistics):
    # Since the downstream apis expect a jnp.array - we create a dummy one if
    # num_statistics=0.
    if not num_statistics:
        return TrainingMetrics([[], jnp.float32])
    else:
        return TrainingMetrics([[num_statistics], jnp.float32])


def init_training_metrics_pspec():
    return TrainingMetrics(pjit.PartitionSpec())


class ShardedShampooStats(NamedTuple):
    """Shampoo state in sharded mode."""

    global_stats: Any
    local_stats: Any


class ShampooState(NamedTuple):
    count: chex.Array
    stats: Any


class InitFnState(NamedTuple):
    init_fn: Any
    pspec_fn: Any
    shape_and_dtype_fn: Any


class GraftingType(enum.IntEnum):
    SGD = 1
    ADAGRAD = 2
    RMSPROP = 3
    RMSPROP_NORMALIZED = 4
    SQRT_N = 5
    ADAGRAD_NORMALIZED = 6


class PreconditionerType(enum.IntEnum):
    # Default, computes preconditioner for each dim
    ALL = 1
    # One sided Shampoo, in this cases only on input dim.
    # Assumes last dim is always the output dim and everything else input dim.
    INPUT = 2


def power_iteration(
    matrix,
    num_iters=100,
    error_tolerance=1e-6,
    precision=lax.Precision.HIGHEST,
):
    r"""Power iteration algorithm.

    The power iteration algorithm takes a symmetric PSD matrix `A`, and produces
    a scalar `\lambda` , which is the greatest (in absolute value) eigenvalue
    of `A`, and a vector v, which is the corresponding eigenvector of `A`.

    References:
      [Wikipedia, 2021](https://en.wikipedia.org/wiki/Power_iteration)

    Args:
      matrix: the symmetric PSD matrix.
      num_iters: Number of iterations.
      error_tolerance: Iterative exit condition.
      precision: precision XLA related flag, the available options are: a)
        lax.Precision.DEFAULT (better step time, but not precise) b)
        lax.Precision.HIGH (increased precision, slower) c) lax.Precision.HIGHEST
        (best possible precision, slowest)

    Returns:
      eigen vector, eigen value
    """
    matrix_size = matrix.shape[-1]

    def _iter_condition(state):
        i, unused_v, unused_s, unused_s_v, run_step = state
        return jnp.logical_and(i < num_iters, run_step)

    def _iter_body(state):
        """One step of power iteration."""
        i, new_v, s, s_v, unused_run_step = state
        new_v = new_v / jnp.linalg.norm(new_v)

        s_v = jnp.einsum("ij,j->i", matrix, new_v, precision=precision)
        s_new = jnp.einsum("i,i->", new_v, s_v, precision=precision)
        return (
            i + 1,
            s_v,
            s_new,
            s_v,
            jnp.greater(jnp.abs(s_new - s), error_tolerance),
        )

    # Figure out how to use step as seed for random.
    v_0 = np.random.RandomState(1729).uniform(-1.0, 1.0, matrix_size).astype(matrix.dtype)

    init_state = tuple([0, v_0, jnp.zeros([], dtype=matrix.dtype), v_0, True])
    _, v_out, s_out, _, _ = lax.while_loop(_iter_condition, _iter_body, init_state)
    v_out = v_out / jnp.linalg.norm(v_out)
    return v_out, s_out


def mat_power(
    mat_m,
    p,
    precision=lax.Precision.HIGHEST,
):
    """A simple matrix power method. M^p where p can be TracedValue."""
    power = jnp.eye(mat_m.shape[0], dtype=_MAT_INV_PTH_ROOT_DTYPE)

    def _iter_condition(state):
        i, _, _ = state
        return i > 0

    def _iter_body(state):
        i, power, mat = state

        power = jax.lax.cond(
            i % 2 == 1,
            lambda: jnp.matmul(mat, power, precision=precision),
            lambda: power,
        )
        i //= 2
        mat = jnp.matmul(mat, mat, precision=precision)
        return i, power, mat

    _, result, _ = lax.while_loop(_iter_condition, _iter_body, (p, power, mat_m))
    return result


def _pth_root_difference(w, alpha, beta, p):
    """Computes (w+alpha)^(-1/p)-(w+beta)^(-1/p)."""

    a = w + alpha
    b = w + beta
    a_minus_b = alpha - beta
    exp = -1 / p

    def _stable_subtract(b, a_minus_b):
        # Mathematically identical to the target expression, with (w+beta)^(-1/p)
        # term factored out and w cancellation in the subtraction.
        return (b**exp) * jnp.expm1(exp * jnp.log1p(a_minus_b / b))

    return jnp.where(
        # Choose the branch with the best log1p approximation.
        jnp.abs(a_minus_b / b) < jnp.abs(a_minus_b / a),
        -_stable_subtract(a, -a_minus_b),
        _stable_subtract(b, a_minus_b),
    )


def matrix_inverse_pth_root(
    matrix,
    p,
    num_iters=100,
    ridge_epsilon=1e-6,
    error_tolerance=1e-6,
    precision=lax.Precision.HIGHEST,
    relative_matrix_epsilon=True,
    lobpcg_topk_precondition=0,
    lobpcg_max_iter=0,
):
    """Computes `matrix^(-1/p)`, where `p` is a positive integer.

    This function uses the Coupled newton iterations algorithm for
    the computation of a matrix's inverse pth root.


    References:
      [Functions of Matrices, Theory and Computation,
       Nicholas J Higham, Pg 184, Eq 7.18](
       https://epubs.siam.org/doi/book/10.1137/1.9780898717778)

    Args:
      matrix: the symmetric PSD matrix whose power it to be computed
      p: exponent, for p a positive integer.
      num_iters: Maximum number of iterations.
      ridge_epsilon: Ridge epsilon added to make the matrix positive definite.
      error_tolerance: Error indicator, useful for early termination.
      precision: precision XLA related flag, the available options are: a)
        lax.Precision.DEFAULT (better step time, but not precise) b)
        lax.Precision.HIGH (increased precision, slower) c) lax.Precision.HIGHEST
        (best possible precision, slowest)
      relative_matrix_epsilon: Whether to use relative epsilon to the max eigen
        value when computing inverse-pth root.
      lobpcg_topk_precondition: If nonzero, specifies the number of top
        eigenvectors to subtract out before performing LOBPCG. Note this makes
        relative_matrix_epsilon essentially free.
      lobpcg_max_iter: Maximum iteration count for LOBPCG, defaults to
        `lobpcg_topk_precondition`.

    Returns:
      matrix^(-1/p) and the error
    """

    # If the input is not square, materialize it from the concatenated form.
    if matrix.shape[0] != matrix.shape[1]:
        matrix = symmetric_matrices.materialize_matrix_from_concat(matrix)

    assert matrix.shape[0] == matrix.shape[1]

    # We use _MAT_INV_PTH_ROOT_DTYPE for the matrix inverse pth root.
    # Switch to f64 if you have hardware that supports it. Enable the jax flag
    # jax_enable_x64 for this to work.
    matrix_size = matrix.shape[0]
    orig_dtype = matrix.dtype
    matrix = matrix.astype(_MAT_INV_PTH_ROOT_DTYPE)
    alpha = jnp.asarray(-1.0 / p, _MAT_INV_PTH_ROOT_DTYPE)
    identity = jnp.eye(matrix_size, dtype=_MAT_INV_PTH_ROOT_DTYPE)
    original_matrix = matrix

    if lobpcg_topk_precondition > 0:
        # TODO(vladf): reuse previous top-k as the initial search directions
        pad_shape = (matrix_size - lobpcg_topk_precondition, lobpcg_topk_precondition)
        search_dirs = jnp.concatenate((jnp.eye(lobpcg_topk_precondition), jnp.zeros(pad_shape)), axis=0)
        eigvals, eigvecs, actual_iters = linalg.lobpcg_standard(
            matrix,
            search_dirs,
            lobpcg_topk_precondition if lobpcg_max_iter == 0 else lobpcg_max_iter,
        )
        del actual_iters  # TODO(vladf): return diagnostics dictionary

        # The minimal eigenvalue among top-k becomes the maximal one in the whole
        # matrix after deflation.
        max_ev = jnp.min(eigvals)
        deflation = eigvals - max_ev
        scaled_vecs = eigvecs * jnp.sqrt(deflation)

        # Deflate out top eigenvectors to reduce matrix condition number.
        matrix -= scaled_vecs.dot(scaled_vecs.T, precision=jax.lax.Precision.HIGHEST)

    # Only use power iteration if lobpcg wasn't already used to derive the
    # top eigenvalue.
    elif relative_matrix_epsilon:
        _, max_ev = power_iteration(matrix=matrix, num_iters=100, error_tolerance=1e-6, precision=precision)
        eigvals, eigvecs = None, None  # Unused but required by pytype.

    # Use absolute matrix epsilon scaling otherwise.
    else:
        max_ev = 1.0
        eigvals, eigvecs = None, None  # Unused but required by pytype.

    ridge_epsilon = ridge_epsilon * jnp.maximum(max_ev, 1e-6)

    def _iter_condition(state):
        (i, unused_mat_m, unused_mat_h, unused_old_mat_h, error, run_step) = state
        error_above_threshold = jnp.logical_and(error > error_tolerance, run_step)
        return jnp.logical_and(i < num_iters, error_above_threshold)

    def _iter_body(state):
        (i, mat_m, mat_h, unused_old_mat_h, error, unused_run_step) = state
        mat_m_i = (1 - alpha) * identity + alpha * mat_m
        new_mat_m = jnp.matmul(mat_power(mat_m_i, p), mat_m, precision=precision)
        new_mat_h = jnp.matmul(mat_h, mat_m_i, precision=precision)
        new_error = jnp.max(jnp.abs(new_mat_m - identity))
        # sometimes error increases after an iteration before decreasing and
        # converging. 1.2 factor is used to bound the maximal allowed increase.
        return (i + 1, new_mat_m, new_mat_h, mat_h, new_error, new_error < error * 1.2)

    if matrix_size == 1:
        resultant_mat_h = (matrix + ridge_epsilon) ** alpha
        error = jnp.array(0, jnp.float32)
    else:
        damped_matrix = matrix + ridge_epsilon * identity

        z = (1 + p) / (2 * jnp.linalg.norm(damped_matrix))
        new_mat_m_0 = damped_matrix * z
        new_error = jnp.max(jnp.abs(new_mat_m_0 - identity))
        new_mat_h_0 = identity * jnp.power(z, 1.0 / p)
        init_state = tuple([0, new_mat_m_0, new_mat_h_0, new_mat_h_0, new_error, True])
        _, mat_m, mat_h, old_mat_h, error, convergence = lax.while_loop(_iter_condition, _iter_body, init_state)
        error = jnp.max(jnp.abs(mat_m - identity)).astype(jnp.float32)
        is_converged = jnp.asarray(convergence, old_mat_h.dtype)
        resultant_mat_h = is_converged * mat_h + (1 - is_converged) * old_mat_h
        resultant_mat_h = jnp.asarray(resultant_mat_h, orig_dtype)

    if lobpcg_topk_precondition > 0:
        # Since we deflated the top eigenvectors prior to p-th root inverse,
        # the resultant matrix has larger eigenvalues associated with those
        # same eigenvectors, which we need to now re-deflate.
        #
        # Note that _pth_root_difference returns positive values for this
        # particular argument ordering as min(eigvals) <= eigvals for the
        # jnp.sqrt below.
        pth_diff = _pth_root_difference(ridge_epsilon, jnp.min(eigvals), eigvals, p)
        scaled_vecs = eigvecs * jnp.sqrt(pth_diff)
        resultant_mat_h = (
            resultant_mat_h.astype(scaled_vecs.dtype)
            - scaled_vecs.dot(scaled_vecs.T, precision=jax.lax.Precision.HIGHEST)
        ).astype(orig_dtype)
        mat_m = jnp.matmul(
            mat_power(resultant_mat_h, p),
            original_matrix,
            precision=jax.lax.Precision.HIGHEST,
        )
        error = jnp.max(jnp.abs(mat_m - identity)).astype(jnp.float32)

    return resultant_mat_h, error


def merge_small_dims(shape_to_merge, max_dim):
    """Merge small dimensions.

    If there are some small dimensions, we collapse them:
    e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if max_dim = 1024
         [1, 2, 768, 1, 2048] --> [2, 768, 2048]

    Args:
      shape_to_merge: Shape to merge small dimensions.
      max_dim: Maximal dimension of output shape used in merging.

    Returns:
      Merged shape.
    """
    if shape_to_merge and np.all(np.array(shape_to_merge) == 1):
        return [1]

    resulting_shape = []
    product = 1
    for d in shape_to_merge:
        if product * d <= max_dim:
            product *= d
        else:
            if product > 1:
                resulting_shape.append(product)
            product = d
    if product > 1:
        resulting_shape.append(product)
    return resulting_shape


def pad_square_matrix(mat, max_size):
    """Pad a square matrix up to max_size.

    Args:
      mat: a matrix to pad.
      max_size: matrix size requested.

    Returns:
      Given M returns [[M, 0], [0, I]]
    """
    rows, cols = mat.shape
    if rows != cols:
        raise ValueError(f"Must have rows == cols, instead got rows={rows}, cols={cols}")
    if cols > max_size:
        raise ValueError(f"Must have cols <= max_size. Instead got cols={cols}, max_size={max_size}.")
    if rows == max_size:
        return mat
    pad_size = max_size - rows

    zs1 = jnp.zeros([rows, pad_size], dtype=mat.dtype)
    zs2 = jnp.zeros([pad_size, rows], dtype=mat.dtype)
    eye = jnp.eye(pad_size, dtype=mat.dtype)
    mat = jnp.concatenate([mat, zs1], 1)
    mat = jnp.concatenate([mat, jnp.concatenate([zs2, eye], 1)], 0)
    return mat


def make_sliced_padding(
    symmetric_block_size,
    num_blocks,
    starting_block,
    dtype,
):
    """Returns padding for symmetric block matrix.

    Specifically, the padding is given concatenated rectangular matrices
    representing the lower-triangular rows below the starting block. For example,
    if we want to pad the symmetric matrix

    M = [[A, B^T]
         [B, C]],

    the desired output (in terms of the full matrix) with num_blocks = 4 is

    M_padded = [[A, B^T, 0, 0]
                [B, C,   0, 0]
                [0, 0,   I, 0]
                 0, 0,   0, I].

    We would represent M as the block matrix mat = [A, B, C]. In this form, the
    additional padding to provide has form [0, 0, I, 0, 0, 0, I] (only the lower
    triangular parts in the third and fourth rows).

    Args:
      symmetric_block_size: The size of each block.
      num_blocks: The total number of blocks.
      starting_block: The block where to start the padding.
      dtype: The type to use for the blocks.
    """
    if starting_block == num_blocks:
        return jnp.zeros(shape=(symmetric_block_size, 0), dtype=dtype)

    blocks = []
    for i in range(starting_block, num_blocks):
        blocks.append(jnp.zeros(shape=(symmetric_block_size, symmetric_block_size * i), dtype=dtype))
        blocks.append(jnp.eye(symmetric_block_size, dtype=dtype))
    return jnp.concatenate(blocks, axis=-1)


def pad_block_symmetric_matrix(
    mat,
    symmetric_block_size,
    max_num_blocks,
):
    """Returns the padded blocked symmetric matrix.

    The size of the padded matrix will be:
      [symmetric_block_size, symmetric_block_size * max_num_blocks]

    The input matrix can either:
      - Be square with size less or equal to symmetric_block_size. In this case,
        mat will first be padded to a square matrix of size symmetric_block_size,
        and then be padded again up to the full size of the blocked matrix.
      - Be a rectangle with number of rows equal to block size.
        In this case, number of columns must be a multiple of number of rows, and
        the ratio must correspond to a block representation of a symmetric matrix.
        That is, the ratio must have form x * (x + 1) / 2. Here, x represents the
        number of block rows represented by the matrix.

    Args:
      mat: The input block matrix.
      symmetric_block_size: The size of blocks.
      max_num_blocks: The largest number of blocks to pad to.
    """
    rows, cols = mat.shape
    if rows > symmetric_block_size:
        raise ValueError(
            "Must have rows <= symmetric_block_size. Instead got "
            f"rows={rows}, symmetric_block_size={symmetric_block_size}."
        )
    if rows > cols:
        raise ValueError(f"Must have rows <= cols, instead got rows={rows}, cols={cols}.")
    if cols > symmetric_block_size * max_num_blocks:
        raise ValueError(
            "Must have cols <= symmetric_block_size * max_num_blocks "
            f"Instead got cols={cols}, "
            f"symmetric_block_size={symmetric_block_size}, "
            f"max_num_blocks={max_num_blocks}."
        )
    if rows < symmetric_block_size:
        mat = pad_square_matrix(mat, max_size=symmetric_block_size)
    # Update rows and cols after possibly padding in pad_square_matrix.
    rows, cols = mat.shape
    assert rows == symmetric_block_size
    assert cols % rows == 0
    filled_blocks = cols // rows
    padding_blocks = make_sliced_padding(
        symmetric_block_size=symmetric_block_size,
        num_blocks=symmetric_matrices.num_blocks_from_total_blocks(max_num_blocks),
        starting_block=symmetric_matrices.num_blocks_from_total_blocks(filled_blocks),
        dtype=mat.dtype,
    )
    return jnp.concatenate([mat, padding_blocks], axis=-1)


def pad_vector(vec, max_size):
    """Pad a vector to a max_size.

    Args:
      vec: a vector to pad.
      max_size: matrix size requested.

    Returns:
      Given V returns [V, 0]
    """
    size = vec.shape[0]
    assert size <= max_size
    if size == max_size:
        return vec
    pad_size = max_size - size
    zs1 = jnp.zeros([pad_size], dtype=vec.dtype)
    return jnp.concatenate([vec, zs1], 0)


def efficient_cond(predicate, compute_fn, init_state, *args, **kwargs):
    """Avoids wasteful buffer allocation with XLA."""

    def _iter_body(unused_state):
        results = compute_fn(*args, **kwargs)
        return tuple([False] + list(results))

    def _iter_condition(state):
        return state[0]

    results = jax.lax.while_loop(_iter_condition, _iter_body, tuple([predicate] + init_state))
    return tuple(results[1:])


class BlockPartitioner:
    """Partitions a tensor into smaller tensors."""

    def __init__(self, param, block_size):
        self._shape = param.shape
        self._splits = []
        split_sizes = []
        # We split params into smaller blocks. Here we store the metadata to make
        # that split.
        for i, d in enumerate(param.shape):
            if 0 < block_size < d:
                # d-1, otherwise split appends a 0-size array.
                nsplit = (d - 1) // block_size
                indices = (np.arange(nsplit, dtype=np.int32) + 1) * block_size
                sizes = np.ones(nsplit + 1, dtype=np.int32) * block_size
                sizes[-1] = d - indices[-1]
                self._splits.append((i, indices))
                split_sizes.append(sizes)
            else:
                split_sizes.append(np.array([d], dtype=np.int32))
        self._split_sizes = split_sizes

    def split_sizes(self):
        return self._split_sizes

    def partition(self, tensor):
        """Partition tensor into blocks."""

        assert tensor.shape == self._shape
        tensors = [tensor]
        for i, indices in self._splits:
            tensors_local = []
            for t in tensors:
                tensors_local.extend(jnp.split(t, indices_or_sections=indices, axis=i))
            tensors = tensors_local
        return tensors

    def merge_partitions(self, partitions):
        """Merge partitions back to original shape."""

        for i, indices in reversed(self._splits):
            n = len(indices) + 1
            partial_merged_tensors = []
            ind = 0
            while ind < len(partitions):
                partial_merged_tensors.append(jnp.concatenate(partitions[ind : ind + n], axis=i))
                ind += n
            partitions = partial_merged_tensors
        assert len(partitions) == 1
        return partitions[0]


def gram_weighted_update(old_stats, g, axis, w1, w2, precision=None):
    """Updated statistics via weighted average with new Gram matrix.

      Returns w₁ R + w₂ Gᵀ G where R is `old_stats` and G is the matrix whose
      columns are the flattened slices of the tensor `g` along the given `axis`.
      (So, `old_stats` and the returned matrix have dimensions n x n where
      n = `g.shape[axis]`).

    Args:
      old_stats:  Old statistics.
      g:  Gradient tensor.
      axis:  Axis along which to slice `g`.
      w1:  Scalar weight for old statistics.
      w2:  Scalar weight for new Gram matrix.
      precision: Optional precision XLA related flag, the available options are:
        a) lax.Precision.DEFAULT (better step time, but not precise)
        b) lax.Precision.HIGH (increased precision, slower)
        c) lax.Precision.HIGHEST (best possible precision, slowest)

    Returns:
      Weighted average of old and new statistics.
    """
    axes = [i for i in range(g.ndim) if i != axis]
    gram_matrix = jnp.tensordot(g, g, axes=(axes, axes), precision=precision)
    return w1 * old_stats + w2 * gram_matrix


class Preconditioner:
    """Compute statistics/shape from gradients for preconditioning."""

    def __init__(
        self,
        param,
        block_size,
        merge_small_dims_block_size,
        best_effort_shape_interpretation,
        preconditioner_type=PreconditionerType.ALL,
    ):
        """Initializes the preconditioner.

        Args:
          param: parameter to precondition.
          block_size: Block size used to split param.
          merge_small_dims_block_size: Block size for merging dims.
          best_effort_shape_interpretation: Whether to collapse/merge dims together.
          preconditioner_type: Type of preconditioner to use.
        """
        self._original_shape = param.shape
        self._transformed_shape = param.shape
        if best_effort_shape_interpretation:
            self._transformed_shape = merge_small_dims(self._original_shape, merge_small_dims_block_size)
        reshaped_param = jnp.reshape(param, self._transformed_shape)
        self._partitioner = BlockPartitioner(reshaped_param, block_size)
        self._preconditioner_type = preconditioner_type

    def updated_statistics_from_grad(
        self,
        stats,
        grad,
        w1,
        w2,
        to_float=None,
        from_float=None,
        precision=None,
    ):
        """Update statistics from gradients.

        Args:
          stats: Old statistics or its Cholesky factor if `cholesky` is True.
          grad: Gradient to compute statistics from.
          w1: Weight for old statistics.
          w2: Weight for new statistics.
          to_float: Optional function for converting stats to floating point.
          from_float: Optional function for converting from floating point.
          precision: Optional precision XLA related flag, the available options are:
            a) lax.Precision.DEFAULT (better step time, but not precise)
            b) lax.Precision.HIGH (increased precision, slower)
            c) lax.Precision.HIGHEST (best possible precision, slowest)

        Returns:
          A list of updated gradient statistics for each partition.
        """
        to_float = to_float if to_float is not None else (lambda x: x)
        from_float = from_float if from_float is not None else (lambda x: x)
        update = functools.partial(gram_weighted_update, precision=precision)
        reshaped_grad = jnp.reshape(grad, self._transformed_shape)
        partitioned_grads = self._partitioner.partition(reshaped_grad)
        new_stats = []
        index = 0
        for g in partitioned_grads:
            should_preconditioned_dims = self.should_precondition_dims()
            num_preconditioners = sum(should_preconditioned_dims)
            for axis in range(num_preconditioners):
                new_stat = update(to_float(stats[index]), g, axis, w1, w2)
                new_stats.append(from_float(new_stat))
                index += 1
        return new_stats

    def should_precondition_dims(self):
        """A vector containing indicator indicating if the dim is preconditioned."""
        split_sizes = self._partitioner.split_sizes()
        rank = len(split_sizes)
        if self._preconditioner_type == PreconditionerType.ALL or rank <= 1:
            return [True] * rank
        else:
            return [True] * (rank - 1) + [False]

    def shapes_for_preconditioners(self):
        """Returns shape from statistics."""
        split_sizes = self._partitioner.split_sizes()
        rank = len(split_sizes)
        # We ignore preconditioner types if rank == 1
        preconditioner_shapes = []
        for t in itertools.product(*split_sizes):
            if self._preconditioner_type == PreconditionerType.ALL or rank <= 1:
                preconditioner_shapes.extend([[d, d] for d in t])
            else:
                preconditioner_shapes.extend([[d, d] for d in t[:-1]])
        return preconditioner_shapes

    def exponent_for_preconditioner(self):
        """Returns exponent to use for inverse-pth root M^{-1/p}."""
        should_preconditioned_dims = self.should_precondition_dims()
        num_preconditioners = sum(should_preconditioned_dims)
        return 2 * num_preconditioners

    def preconditioned_grad(self, grad, preconditioners):
        """Precondition the gradient.

        Args:
          grad: A gradient tensor to precondition.
          preconditioners: A list of preconditioners to apply.

        Returns:
          A preconditioned gradient.
        """

        reshaped_grad = jnp.reshape(grad, self._transformed_shape)
        partitioned_grads = self._partitioner.partition(reshaped_grad)
        preconditioned_partitioned_grads = []
        for i, g in enumerate(partitioned_grads):
            should_preconditioned_dims = self.should_precondition_dims()
            num_preconditioners = sum(should_preconditioned_dims)
            preconditioners_for_grad = preconditioners[i * num_preconditioners : (i + 1) * num_preconditioners]
            precond_g = g
            rank = len(g.shape)
            for j, precondition in enumerate(should_preconditioned_dims):
                if precondition:
                    precond_g = jnp.tensordot(precond_g, preconditioners_for_grad[j], axes=[[0], [0]])
                else:
                    precond_g = jnp.transpose(precond_g, axes=(*range(1, rank), 0))
            preconditioned_partitioned_grads.append(precond_g)
        merged_grad = self._partitioner.merge_partitions(preconditioned_partitioned_grads)
        return jnp.reshape(merged_grad, self._original_shape)


def _convert_to_parameter_stats(global_stats, local_stat, convert_statistics=True):
    """Creates parameter stats from sharded stats."""
    index_start = int(local_stat.index_start)
    index_end = int(len(local_stat.sizes)) + index_start
    statistics = global_stats.statistics[index_start:index_end, :, :]
    preconditioners = global_stats.preconditioners[index_start:index_end, :, :]
    new_statistics = []
    new_preconditioners = []
    for i, size in enumerate(local_stat.sizes):
        new_statistics.append(statistics[i][:size, :size])
        new_preconditioners.append(preconditioners[i][:size, :size])
    if not convert_statistics:
        new_statistics = None
    return ParameterStats(
        local_stat.diagonal_statistics,
        new_statistics,
        new_preconditioners,
        local_stat.diagonal_momentum,
        local_stat.momentum,
        local_stat.training_metrics,
    )


def _convert_from_parameter_stats(parameter_stats, local_stats):
    """Creates sharded stats from paramter stats."""
    return LocalShardedParameterStats(
        parameter_stats.diagonal_statistics,
        parameter_stats.diagonal_momentum,
        parameter_stats.momentum,
        parameter_stats.training_metrics,
        local_stats.index_start,
        local_stats.sizes,
    )


def _add_error_into_local_stats(local_stats, errors, inverse_failure_threshold):
    """Adds errors back into local statistics."""
    new_local_stats = []
    for local_stat in local_stats:
        if local_stat.sizes:
            index_start = int(local_stat.index_start)
            index_end = int(len(local_stat.sizes)) + index_start
            per_stat_error = errors[index_start:index_end]
        else:
            per_stat_error = jnp.array(0, jnp.float32)
        if local_stat.sizes:
            per_stat_error = jnp.where(
                jnp.logical_and(per_stat_error > 0.0, per_stat_error != inverse_failure_threshold),
                per_stat_error,
                local_stat.training_metrics.inverse_pth_root_errors,
            )
        new_local_stats.append(
            LocalShardedParameterStats(
                local_stat.diagonal_statistics,
                local_stat.diagonal_momentum,
                local_stat.momentum,
                TrainingMetrics(per_stat_error),
                local_stat.index_start,
                local_stat.sizes,
            )
        )
    return new_local_stats


def batch(x, num_devices):
    """Batch `x` so that so that leading axis is num_devices."""
    n = len(x)
    b = int(n / num_devices)
    return jnp.stack([jnp.stack(x[idx : idx + b]) for idx in range(0, n, b)])


def unbatch(batched_values):
    """Unbatch values across leading axis and return a list of elements."""
    b1, b2 = batched_values.shape[0], batched_values.shape[1]
    results = []
    for v_array in jnp.split(batched_values, indices_or_sections=b1, axis=0):
        v_array = jnp.squeeze(v_array)
        # b2 = batches (number of preconditioner computation) per core.
        if b2 > 1:
            for v in jnp.split(v_array, indices_or_sections=b2, axis=0):
                results.append(jnp.squeeze(v))
        else:
            results.append(v_array)
    return results


def distributed_shampoo(
    learning_rate,
    block_size,
    beta1=0.9,
    beta2=0.999,
    diagonal_epsilon=1e-10,
    matrix_epsilon=1e-6,
    weight_decay=0.0,
    start_preconditioning_step=5,
    preconditioning_compute_steps=1,
    statistics_compute_steps=1,
    best_effort_shape_interpretation=True,
    graft_type=GraftingType.SGD,
    nesterov=True,
    exponent_override=0,
    # Pass pmap 'batch axis name' in pmap mode.
    batch_axis_name=None,
    ### Only set following 3 params in pjit/spmd mode.
    ### WARNING: Experimental
    statistics_partition_spec=None,
    preconditioner_partition_spec=None,
    num_devices_for_pjit=None,
    shard_optimizer_states=False,
    ###
    ### Experimental memory reduction mode
    best_effort_memory_usage_reduction=False,
    ###
    inverse_failure_threshold=0.1,
    moving_average_for_momentum=False,
    skip_preconditioning_dim_size_gt=4096,
    clip_by_scaled_gradient_norm=None,
    precision=lax.Precision.HIGHEST,
    tensordot_precision=None,
    relative_matrix_epsilon=True,
    merge_small_dims_block_size=4096,
    lobpcg_topk_precondition=0,
    lobpcg_max_iter=0,
    precondtioner_type=PreconditionerType.ALL,
    skip_preconditioning_rank_lt=1,
    decoupled_learning_rate=True,
    decoupled_weight_decay=False,
):
    """Distributed Shampoo optimizer.

    Distributed Shampoo is a second-order preconditioned method (concretely, a
    variant of full-matrix Adagrad), that provides significant convergence and
    wall-clock time improvements compared to conventional first-order methods,
    and that has been shown to scale to large state-of-the-art deep learning
    models.

    References:
      Scalable Second Order Optimization for Deep Learning,
      Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer

      Preprint: https://arxiv.org/abs/2002.09018

    Args:
      learning_rate: the step size used to update the parameters.
      block_size: Block size for large layers (if > 0). Preconditioning compute
        operation is cubic in the dimension of the tensor. Block size allows us to
        chunk the layers into sub-layers of maximal dimension dictated by this
        value. Use 128 as default (increase if you have compute budget).
      beta1: momentum parameter.
      beta2: second moment averaging parameter.
      diagonal_epsilon: epsilon for diagonal adagrad (only if layerwise grafting
        to AdaGrad is enabled).
      matrix_epsilon: epsilon to add to statistics before computing inverse pth
        root. If you are running in f32 precision for inverse pth root
        (recommended today) this can go upto 1e-6. If you have latest hardware
        with native f64 precision, set this upto 1e-12.
      weight_decay: Weight decay for regularization.
      start_preconditioning_step: When to start Shampoo update before which
        diagonal update is used. This is because we dont have enough information
        to do stable inverse.
      preconditioning_compute_steps: How often to compute preconditioner.
        Performance tuning params for controlling memory and compute requirements.
        Ideally set this and statistics_compute_steps params to 1.
      statistics_compute_steps: How often to compute statistics.
      best_effort_shape_interpretation: If there are some small dimensions,
        collapse them e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if
        block = 1024, [1, 2, 768, 1, 2048] --> [2, 768, 2048]
      graft_type: Grafting is a technique to fix the layerwise scale of Shampoo
        optimizer. This allows us to plugin the Shampoo optimizer into settings
        where SGD/AdaGrad is already well tuned.
      nesterov: Nesterov momentum.
      exponent_override: Override the exponent used in matrix inverse.
      batch_axis_name: labeled axis over pmap for data-parallel training the
        optimizer used for.
      statistics_partition_spec: PartitionSpec to be used in sharded mode.
      preconditioner_partition_spec: PartitionSpec to be used in sharded mode.
      num_devices_for_pjit: Number of devices to parallelize over when using pjit.
      shard_optimizer_states: Shard optimizer states to save memory in model
        parallel training.
      best_effort_memory_usage_reduction: Best effort memory usage reduction. -
        diagonal_statistics -> jnp.bfloat16 - momentum buffers (2x) -> jnp.int8 -
        statistics, preconditioners -> jnp.int16 + diagonals
      inverse_failure_threshold: numerics are hard and inverses fail sometimes; we
        determine that using this threshold.
      moving_average_for_momentum: Whether to use moving average for momentum
        instead of exponential moving average.
      skip_preconditioning_dim_size_gt: Skip if preconditioning dim size is
        greater than this value.
      clip_by_scaled_gradient_norm: Clip by scaled gradient norm (only useful when
        using RMSProp Grafting).
      precision: precision XLA related flag, the available options are: a)
        lax.Precision.DEFAULT (better step time, but not precise) b)
        lax.Precision.HIGH (increased precision, slower) c) lax.Precision.HIGHEST
        (best possible precision, slowest)
      tensordot_precision: Optional precision to use for the tensordot operation
        when computing statistics (e.g., G Gᵀ). Same options as `precision` above.
      relative_matrix_epsilon: Whether to use relative epsilon to the max eigen
        value when computing inverse-pth root.
      merge_small_dims_block_size: Used as the maximum block size
        to merge the shapes.
      lobpcg_topk_precondition: If nonzero, specifies the number of top
        eigenvectors to subtract out before performing LOBPCG. Note this makes
        relative_matrix_epsilon essentially free.
      lobpcg_max_iter: Number of LOBPCG iterations, if zero defaults to
        `lobpcg_topk_precondition`.
      precondtioner_type: Preconditioner type to select all, left only or right
        only preconditioners.
      skip_preconditioning_rank_lt: Skips preconditioning for parameters with
        rank less than this value.
      decoupled_learning_rate: If True, use decoupled learning rate, otherwise
        couple it with preconditioned gradient computation. (Default True)
      decoupled_weight_decay: If True, use decoupled weight decay, otherwise
        couple with weight decay. (Default False)
    Returns:
      a GradientTransformation.
    """

    def _graft_type_has_diagonal_statistics():
        """Returns True if using diagonal firt order method for grafting."""
        return graft_type != GraftingType.SGD and graft_type != GraftingType.SQRT_N

    def quantized_dtype_for_momentum_buffers(var):
        return jnp.int8 if best_effort_memory_usage_reduction and len(var.shape) > 1 else jnp.float32

    # Preconditioner and statistics are both stores as int16 in this mode.
    # We take out the diagonal to make quantization easier.
    def quantized_dtype_for_second_moment_statistics_buffers():
        return jnp.int16 if best_effort_memory_usage_reduction and batch_axis_name else jnp.float32

    # Preconditioner and statistics are both stores as int16 in this mode.
    # We take out the diagonal to make quantization easier.
    def quantized_dtype_for_second_moment_preconditioner_buffers():
        return jnp.int16 if best_effort_memory_usage_reduction and batch_axis_name else jnp.float32

    def _to_float(maybe_quantized):
        if isinstance(maybe_quantized, QuantizedValue):
            return maybe_quantized.to_float()
        else:
            return maybe_quantized

    def _maybe_quantize_statistics(statistics_list):
        return _maybe_quantize_matrices_with_dtype(
            statistics_list, quantized_dtype_for_second_moment_statistics_buffers()
        )

    def _maybe_quantize_preconditioners(statistics_list):
        return _maybe_quantize_matrices_with_dtype(
            statistics_list, quantized_dtype_for_second_moment_preconditioner_buffers()
        )

    def _maybe_quantize_matrices_with_dtype(statistics_list, quantized_dtype):
        if quantized_dtype != jnp.float32:
            return [
                QuantizedValue.from_float_value(s, quantized_dtype, extract_diagonal=True) for s in statistics_list
            ]
        else:
            return statistics_list

    def _maybe_dequantize_preconditioners(preconditioner_list):
        return _maybe_dequantize_matrices_with_dtype(
            preconditioner_list,
            quantized_dtype_for_second_moment_preconditioner_buffers(),
        )

    def _maybe_dequantize_matrices_with_dtype(statistics_list, quantized_dtype):
        if quantized_dtype != jnp.float32:
            return [s.to_float() for s in statistics_list]
        else:
            return statistics_list

    def _quantize_diagonal_statistics(diagonal_statistics):
        return QuantizedValue.from_float_value(diagonal_statistics, jnp.float32)

    def _quantize_momentum(momentum_statistics):
        return QuantizedValue.from_float_value(
            momentum_statistics,
            quantized_dtype_for_momentum_buffers(momentum_statistics),
        )

    def preconditioner_from_params(param):
        """Returns a Preconditioner object for given param."""
        return Preconditioner(
            param,
            block_size,
            merge_small_dims_block_size,
            best_effort_shape_interpretation,
            precondtioner_type,
        )

    def sharded_init_fn(params):
        """Returns optimizer state (for PJIT mode).

        Args:
          params: the parameters that should be updated.
        """
        params_flat, treedef = jax.tree_flatten(params)
        # Find max size to pad to.
        max_size = 0
        for param in params_flat:
            preconditioner = preconditioner_from_params(param)
            if not _skip_preconditioning(param):
                shapes = preconditioner.shapes_for_preconditioners()
                sizes = [s[0] for s in shapes]
                max_size = max(max(sizes), max_size)

        padded_statistics = []
        padded_preconditioners = []
        local_stats_flat = []
        exponents = []
        for param in params_flat:
            preconditioner = preconditioner_from_params(param)
            shapes = preconditioner.shapes_for_preconditioners()
            sizes = []

            statistics = []
            preconditioners = []
            index_start = len(padded_statistics)
            if not _skip_preconditioning(param):
                sizes = [s[0] for s in shapes]
                shapes = preconditioner.shapes_for_preconditioners()
                statistics = [matrix_epsilon * jnp.eye(max_size, dtype=jnp.float32) for s in shapes]
                preconditioners = [jnp.eye(max_size, dtype=jnp.float32) for s in shapes]
                padded_statistics.extend(statistics)
                padded_preconditioners.extend(preconditioners)
                exponent = (
                    preconditioner.exponent_for_preconditioner() if exponent_override == 0 else exponent_override
                )
                exponents.extend([exponent] * len(shapes))

            diagonal_statistics = _quantize_diagonal_statistics(jnp.zeros_like(param))
            diagonal_momentum = _quantize_momentum(jnp.zeros_like(param))
            momentum = _quantize_momentum(jnp.zeros_like(param))

            local_stats_flat.append(
                LocalShardedParameterStats(
                    diagonal_statistics,
                    diagonal_momentum,
                    momentum,
                    init_training_metrics(len(sizes)),
                    index_start,
                    sizes,
                )
            )

        local_stats = jax.tree_unflatten(treedef, local_stats_flat)
        to_pad = -len(padded_statistics) % num_devices_for_pjit
        if max_size == 0:
            to_pad = num_devices_for_pjit
            max_size = block_size
            stat_dtype = jnp.float32
        else:
            stat_dtype = padded_statistics[0].dtype
        # Pad the statistics and preconditioner matrices to be a multiple of
        # num devices.
        # TODO(rohananil): Relax to only the size of the mesh axis where the dim
        # is split on.
        padded_statistics.extend([jnp.eye(max_size, dtype=stat_dtype) for _ in range(to_pad)])
        padded_preconditioners.extend([jnp.eye(max_size, dtype=stat_dtype) for _ in range(to_pad)])
        exponents.extend([1 for _ in range(to_pad)])
        global_stats = GlobalShardedParameterStats(
            jnp.stack(padded_statistics),
            jnp.stack(padded_preconditioners),
            jnp.stack(exponents),
        )
        return ShampooState(
            count=jnp.zeros([], jnp.int32),
            stats=ShardedShampooStats(global_stats, local_stats),
        )

    def _max_statistics_size_from_params(params):
        max_size = 0
        for param in params:
            param_clone = jnp.zeros(param.shape, dtype=param.dtype)
            preconditioner = preconditioner_from_params(param_clone)
            if not _skip_preconditioning(param):
                shapes = preconditioner.shapes_for_preconditioners()
                sizes = [s[0] for s in shapes]
                max_size = max(max(sizes), max_size)
        return max_size

    def _remove_leading_sharding_annotation(pspec):
        """Mapping from N-d to (N-1)-d, used for quantization, factoring etc."""
        # None and PSpec(None) are valid PSpecs.
        if pspec and len(pspec) > 1:
            return pjit.PartitionSpec(*pspec[1:])
        else:
            return []

    def sharded_init_partition_spec_fn(params, params_partition_spec, partition_spec_for_statistics):
        """Returns a parallel state tree with PartitionSpec associated with state.


        Args:
          params: A pytree with params.
          params_partition_spec: A pytree with PartitionSpec for params.
          partition_spec_for_statistics: PartitionSpec for the statistics.
        """
        # Parallel lists of spec, and params.
        param_pspec_flat, _ = jax.tree_flatten(params_partition_spec, is_leaf=lambda x: x is None)
        params_flat, treedef = jax.tree_flatten(params)
        assert param_pspec_flat
        assert params_flat
        # Step is replicated across cores.
        # None means cores.
        local_stats_flat = []
        num_statistics = 0
        for param, param_pspec in zip(params_flat, param_pspec_flat):
            param_clone = jnp.zeros(param.shape, dtype=param.dtype)
            preconditioner = preconditioner_from_params(param_clone)
            shapes = preconditioner.shapes_for_preconditioners()
            sizes = []

            index_start = num_statistics
            if not _skip_preconditioning(param):
                sizes = [s[0] for s in shapes]
                shapes = preconditioner.shapes_for_preconditioners()
                num_statistics += len(shapes)

            qdtype = quantized_dtype_for_momentum_buffers(param)
            m1_pspec = param_pspec
            m2_pspec = param_pspec
            m1_scale_pspec = []
            m2_scale_pspec = []
            if qdtype != jnp.float32:
                m1_scale_pspec = _remove_leading_sharding_annotation(m1_pspec)
                m2_scale_pspec = _remove_leading_sharding_annotation(m2_pspec)

            local_stats_flat.append(
                LocalShardedParameterStats(
                    QuantizedValue(param_pspec, [], [], jnp.float32, False, list(param.shape)),
                    QuantizedValue(m1_pspec, [], m1_scale_pspec, qdtype, False, list(param.shape)),
                    QuantizedValue(m2_pspec, [], m2_scale_pspec, qdtype, False, list(param.shape)),
                    init_training_metrics_pspec(),
                    index_start,
                    sizes,
                )
            )

        local_stats = jax.tree_unflatten(treedef, local_stats_flat)
        global_stats = GlobalShardedParameterStats(
            partition_spec_for_statistics,
            partition_spec_for_statistics,
            pjit.PartitionSpec(),
        )
        count_pspec = pjit.PartitionSpec()
        return ShampooState(count=count_pspec, stats=ShardedShampooStats(global_stats, local_stats))

    def sharded_init_shape_and_dtype_fn(params):
        """Returns a parallel state tree with shape, dtype associated with state.


        Args:
          params: A pytree with params.
        """
        # Parallel lists of spec, and params.
        params_flat, treedef = jax.tree_flatten(params)
        assert params_flat
        # Step is replicated across cores.
        # None means cores.
        local_stats_flat = []
        num_statistics = 0
        for param in params_flat:
            param_clone = jnp.zeros(param.shape, dtype=param.dtype)
            preconditioner = preconditioner_from_params(param_clone)
            shapes = preconditioner.shapes_for_preconditioners()
            sizes = []

            index_start = num_statistics
            if not _skip_preconditioning(param):
                sizes = [s[0] for s in shapes]
                shapes = preconditioner.shapes_for_preconditioners()
                num_statistics += len(shapes)

            qdtype = quantized_dtype_for_momentum_buffers(param)
            m1_shape_and_dtype = [list(param.shape), param.dtype]
            m2_shape_and_dtype = [list(param.shape), param.dtype]
            m1_scale_shape_and_dtype = []
            m2_scale_shape_and_dtype = []
            if qdtype != jnp.float32:
                m1_scale_shape_and_dtype = [list(param.shape)[1:], qdtype]
                m2_scale_shape_and_dtype = [list(param.shape)[1:], qdtype]

            diagonal_statistics_shape_and_dtype = [list(param.shape), param.dtype]
            local_stats_flat.append(
                LocalShardedParameterStats(
                    QuantizedValue(
                        diagonal_statistics_shape_and_dtype,
                        [],
                        [],
                        jnp.float32,
                        False,
                        list(param.shape),
                    ),
                    QuantizedValue(
                        m1_shape_and_dtype,
                        [],
                        m1_scale_shape_and_dtype,
                        qdtype,
                        False,
                        list(param.shape),
                    ),
                    QuantizedValue(
                        m2_shape_and_dtype,
                        [],
                        m2_scale_shape_and_dtype,
                        qdtype,
                        False,
                        list(param.shape),
                    ),
                    init_training_metrics_shapes(len(sizes)),
                    index_start,
                    sizes,
                )
            )

        local_stats = jax.tree_unflatten(treedef, local_stats_flat)
        max_statistics_size = _max_statistics_size_from_params(params_flat)
        to_pad = -num_statistics % num_devices_for_pjit
        num_statistics += to_pad
        if num_statistics == 0:
            num_statistics = num_devices_for_pjit
            max_statistics_size = block_size
        statistics_shape = [num_statistics, max_statistics_size, max_statistics_size]
        global_stats = GlobalShardedParameterStats(
            [statistics_shape, jnp.float32],
            [statistics_shape, jnp.float32],
            [[num_statistics], jnp.int32],
        )
        return ShampooState(
            count=[[], jnp.float32],
            stats=ShardedShampooStats(global_stats, local_stats),
        )

    def sharded_update_fn(grads, state, params):
        """Transform the input gradient and update all statistics in sharded mode.

        Args:
          grads: the gradient tensors for the parameters.
          state: a named tuple containing the state of the optimizer
          params: the parameters that should be updated.

        Returns:
          A tuple containing the new parameters and the new optimizer state.
        """
        params_flat, treedef = jax.tree_flatten(params)
        grads_flat = treedef.flatten_up_to(grads)

        global_stats = state.stats.global_stats
        local_stats_flat = treedef.flatten_up_to(state.stats.local_stats)
        stats_flat = [_convert_to_parameter_stats(global_stats, local_stat) for local_stat in local_stats_flat]
        new_stats_flat = jax.tree_map(
            lambda g, s, p: _compute_stats(g, s, p, state.count),
            grads_flat,
            stats_flat,
            params_flat,
        )

        outputs = jax.tree_map(
            lambda g, s, p: _transform_grad(g, s, p, state.count),
            grads_flat,
            new_stats_flat,
            params_flat,
        )
        updates_flat, new_stats_flat = list(zip(*outputs)) if outputs else ((), ())

        updates = jax.tree_unflatten(treedef, updates_flat)
        # Create new local_stats
        new_local_stats_flat = [
            _convert_from_parameter_stats(new_stat, local_stat)
            for new_stat, local_stat in zip(new_stats_flat, local_stats_flat)
        ]

        max_size = global_stats.statistics.shape[1]
        new_padded_statistics = []
        for stat in new_stats_flat:
            new_padded_statistics.extend([pad_square_matrix(stat, max_size) for stat in stat.statistics])

        # Create global stats
        # TODO(rohananil): Preconditioner is not updated every step, so cost of
        # stack/pad can be obviated away.
        # Pad the statistics and preconditioner matrices to be a multiple of
        # num devices.
        # TODO(rohananil): Relax to only the size of the mesh axis where the dim
        # is split on.
        to_pad = -len(new_padded_statistics) % num_devices_for_pjit
        if not new_padded_statistics:
            to_pad = num_devices_for_pjit
            stat_dtype = jnp.float32
        else:
            stat_dtype = new_padded_statistics[0].dtype

        new_padded_statistics.extend([jnp.eye(max_size, dtype=stat_dtype) for _ in range(to_pad)])
        new_stacked_padded_statistics = jnp.stack(new_padded_statistics)
        new_stacked_padded_statistics = pjit.with_sharding_constraint(
            new_stacked_padded_statistics, statistics_partition_spec
        )

        def _internal_inverse_pth_root_all():
            preconditioners, errors = _matrix_inverse_pth_root_pjit(
                new_stacked_padded_statistics,
                global_stats.exponents,
                statistics_partition_spec,
            )
            return preconditioners, errors

        if preconditioning_compute_steps == 1:
            new_preconditioners, errors = _internal_inverse_pth_root_all()
        else:
            # Passing statistics instead of preconditioners as they are similarly
            # shaped tensors. Note statistics will be ignored as we are passing in
            # a large init value for error.
            preconditioners_init = new_stacked_padded_statistics
            n = new_stacked_padded_statistics.shape[0]
            errors_init = jnp.ones([n], jnp.float32) * inverse_failure_threshold
            init_state = [preconditioners_init, errors_init]
            perform_step = state.count % preconditioning_compute_steps == 0
            new_preconditioners, errors = efficient_cond(perform_step, _internal_inverse_pth_root_all, init_state)

        new_local_stats_flat = _add_error_into_local_stats(new_local_stats_flat, errors, inverse_failure_threshold)
        new_local_stats = jax.tree_unflatten(treedef, new_local_stats_flat)
        errors = errors.reshape((-1, 1, 1))
        predicate = jnp.logical_or(jnp.isnan(errors), errors >= inverse_failure_threshold).astype(
            new_preconditioners.dtype
        )
        # TODO(rohananil): Check for numerical instabilities.
        new_conditional_preconditioners = (
            predicate * global_stats.preconditioners + (1.0 - predicate) * new_preconditioners
        )
        new_global_stats = GlobalShardedParameterStats(
            new_stacked_padded_statistics,
            new_conditional_preconditioners,
            global_stats.exponents,
        )
        new_shampoo_state = ShampooState(
            count=state.count + 1,
            stats=ShardedShampooStats(new_global_stats, new_local_stats),
        )
        return updates, new_shampoo_state

    def init_fn(params):
        """Initialise the optimiser's state."""

        def _init(param):
            preconditioner = preconditioner_from_params(param)
            statistics = []
            preconditioners = []
            if not _skip_preconditioning(param):
                shapes = preconditioner.shapes_for_preconditioners()
                statistics = [matrix_epsilon * jnp.eye(s[0], dtype=jnp.float32) for s in shapes]
                preconditioners = [jnp.eye(s[0], dtype=jnp.float32) for s in shapes]

            diagonal_statistics = []
            if _graft_type_has_diagonal_statistics():
                diagonal_statistics = jnp.zeros_like(param)

            diagonal_momentum = _quantize_momentum(jnp.zeros_like(param))
            momentum = _quantize_momentum(jnp.zeros_like(param))

            return ParameterStats(
                _quantize_diagonal_statistics(diagonal_statistics),
                _maybe_quantize_statistics(statistics),
                _maybe_quantize_preconditioners(preconditioners),
                diagonal_momentum,
                momentum,
                init_training_metrics(len(statistics)),
            )

        return ShampooState(count=jnp.zeros([], jnp.int32), stats=jax.tree_map(_init, params))

    def _skip_preconditioning(param):
        return len(param.shape) < skip_preconditioning_rank_lt or any(
            [s > skip_preconditioning_dim_size_gt for s in param.shape]
        )

    def _compute_stats(grad, state, param, step):
        """Compute per-parameter statistics."""
        preconditioner = preconditioner_from_params(param)
        new_statistics = [[]] * len(state.statistics)
        w1 = beta2
        w2 = beta2 if beta2 == 1.0 else (1.0 - beta2)
        if not _skip_preconditioning(param):

            def compute_updated_statistics():
                return preconditioner.updated_statistics_from_grad(
                    state.statistics,
                    grad,
                    w1=w1,
                    w2=w2,
                    to_float=_to_float,
                    from_float=lambda x: _maybe_quantize_statistics([x])[0],
                    precision=tensordot_precision,
                )

            if statistics_compute_steps > 1:
                perform_step = step % statistics_compute_steps == 0
                init_state = state.statistics
                new_statistics = list(efficient_cond(perform_step, compute_updated_statistics, init_state))
            else:
                new_statistics = compute_updated_statistics()
        return ParameterStats(
            state.diagonal_statistics,
            new_statistics,
            state.preconditioners,
            state.diagonal_momentum,
            state.momentum,
            state.training_metrics,
        )

    mi_pth_root = functools.partial(
        matrix_inverse_pth_root,
        ridge_epsilon=matrix_epsilon,
        precision=precision,
        relative_matrix_epsilon=relative_matrix_epsilon,
        lobpcg_topk_precondition=lobpcg_topk_precondition,
        lobpcg_max_iter=lobpcg_max_iter,
    )

    def _matrix_inverse_pth_root_vmap(xs, ps):
        return jax.vmap(mi_pth_root)(xs, ps)

    def _quantized_matrix_inverse_pth_root_vmap(qxs, qds, qbs, ps):
        def _quantized_to_float(qx, qd, qb):
            qv = QuantizedValue(qx, qd, qb, qx.dtype, True, list(qx.shape))
            return qv.to_float()

        def matrix_inverse_pth_root_wrapper(qx, qd, qb, p):
            v = _quantized_to_float(qx, qd, qb)
            preconditioner, error = mi_pth_root(v, p)
            qp = QuantizedValue.from_float_value(preconditioner, qx.dtype, True)
            return qp.quantized, qp.diagonal, qp.bucket_size, error

        return jax.vmap(matrix_inverse_pth_root_wrapper)(qxs, qds, qbs, ps)

    def _matrix_inverse_pth_root_pjit(xs, ps, statistics_partition_spec=None):
        # Partition the concatenated statistics matrix across all cores.
        pspec_for_partition = preconditioner_partition_spec
        partitioned_xs = pjit.with_sharding_constraint(xs, pspec_for_partition)
        if preconditioner_partition_spec:
            partitioned_ps_spec = pjit.PartitionSpec(preconditioner_partition_spec[0])
        else:
            partitioned_ps_spec = None
        partitioned_ps = pjit.with_sharding_constraint(ps, partitioned_ps_spec)
        # Run matrix inverse pth root on each shard.
        partitioned_preconditioners, partitioned_errors = _matrix_inverse_pth_root_vmap(partitioned_xs, partitioned_ps)
        # Reshard output to have the same PSpec as input. This is required to avoid
        # vmap seeing the full set of statistics.
        partitioned_preconditioners = pjit.with_sharding_constraint(partitioned_preconditioners, pspec_for_partition)
        # Recombine the outputs at each core.
        preconditioners = pjit.with_sharding_constraint(partitioned_preconditioners, statistics_partition_spec)
        errors = pjit.with_sharding_constraint(partitioned_errors, pjit.PartitionSpec())
        return preconditioners, errors

    def _pmap_compute_preconditioners(
        states,
        step,
        statistics,
        num_statistics_per_state,
        original_shapes,
        exponents,
        max_size,
        prev_preconditioners,
    ):
        """Computes preconditioners for given statistics in states in PMAP mode.

        Args:
          states: A list of optimizer states.
          step: Current step number
          statistics: A list of statistics for all variables (for every dim)
          num_statistics_per_state: Number of statistis per state to reconstruct
            output states.
          original_shapes: A list of shapes of the statistics.
          exponents: Exponent power to use for inverse-pth roots.
          max_size: Maximum dim of the statistics to pad.
          prev_preconditioners: Previously available preconditioner.

        Returns:
          New optimizer states after computing the preconditioner.
        """
        if batch_axis_name:
            num_devices = lax.psum(1, batch_axis_name)
        else:
            num_devices = 1
        num_statistics = len(statistics)
        # Pad statistics and exponents to next multiple of num_devices.
        packed_statistics = [pad_square_matrix(stat, max_size) for stat in statistics]
        to_pad = -num_statistics % num_devices
        packed_statistics.extend([jnp.eye(max_size, dtype=packed_statistics[0].dtype) for _ in range(to_pad)])
        exponents.extend([1 for _ in range(to_pad)])

        if not packed_statistics:
            return states

        all_statistics = batch(packed_statistics, num_devices)
        all_exponents = batch(exponents, num_devices)

        def _internal_inverse_pth_root_all():
            if batch_axis_name:
                current_replica = lax.axis_index(batch_axis_name)
                preconditioners, errors = _matrix_inverse_pth_root_vmap(
                    all_statistics[current_replica], all_exponents[current_replica]
                )
                preconditioners = jax.lax.all_gather(preconditioners, batch_axis_name)
                errors = jax.lax.all_gather(errors, batch_axis_name)
                preconditioners_flat = unbatch(preconditioners)
                errors_flat = unbatch(errors)
            else:
                preconditioners, errors = _matrix_inverse_pth_root_vmap(all_statistics[0], all_exponents[0])
                preconditioners_flat = unbatch(jnp.stack([preconditioners]))
                errors_flat = unbatch(jnp.stack([errors]))

            return preconditioners_flat, errors_flat

        if preconditioning_compute_steps == 1:
            preconditioners_flat, errors_flat = _internal_inverse_pth_root_all()
        else:
            # Passing statistics instead of preconditioners as they are similarly
            # shaped tensors. Note statistics will be ignored as we are passing in
            # a large init value for error.
            preconditioners_init = packed_statistics
            errors_init = [inverse_failure_threshold] * len(packed_statistics)
            init_state = [preconditioners_init, errors_init]
            perform_step = step % preconditioning_compute_steps == 0
            preconditioners_flat, errors_flat = efficient_cond(
                perform_step, _internal_inverse_pth_root_all, init_state
            )

        def _skip(error):
            condition = jnp.logical_or(jnp.isnan(error), error >= inverse_failure_threshold)
            return condition.astype(error.dtype)

        def _select_preconditioner(error, new_p, old_p):
            return lax.cond(_skip(error), lambda _: old_p, lambda _: new_p, operand=None)

        new_preconditioners_flat = []
        new_errors_flat = []
        for p, shape, prev_p, error in zip(preconditioners_flat, original_shapes, prev_preconditioners, errors_flat):
            new_preconditioners_flat.append(_select_preconditioner(error, p[: shape[0], : shape[1]], prev_p))
            new_errors_flat.append(error)

        assert len(states) == len(num_statistics_per_state)
        assert len(new_preconditioners_flat) == num_statistics
        assert len(new_errors_flat) == num_statistics

        # Add back empty preconditioners so we that we can set the optimizer state.
        preconditioners_for_states = []
        idx = 0
        errors_for_states = []
        for num_statistics, state in zip(num_statistics_per_state, states):
            if num_statistics == 0:
                preconditioners_for_states.append([])
                errors_for_states.append(jnp.array(0, jnp.float32))
            else:
                preconditioners_for_state = new_preconditioners_flat[idx : idx + num_statistics]
                assert len(state.statistics) == len(preconditioners_for_state)
                preconditioners_for_states.append(preconditioners_for_state)

                errors_for_state = jnp.stack(new_errors_flat[idx : idx + num_statistics])
                assert len(state.statistics) == len(errors_for_state)
                errors_for_states.append(errors_for_state)

                idx += num_statistics
        new_states = []
        for state, new_preconditioners, new_errors in zip(states, preconditioners_for_states, errors_for_states):
            if state.statistics:
                new_errors = jnp.where(
                    jnp.logical_and(new_errors > 0.0, new_errors != inverse_failure_threshold),
                    new_errors,
                    state.training_metrics.inverse_pth_root_errors,
                )
            new_training_metrics = TrainingMetrics(new_errors)
            new_states.append(
                ParameterStats(
                    state.diagonal_statistics,
                    state.statistics,
                    new_preconditioners,
                    state.diagonal_momentum,
                    state.momentum,
                    new_training_metrics,
                )
            )

        return new_states

    def _pmap_quantized_compute_preconditioners(
        states,
        step,
        statistics,
        num_statistics_per_state,
        original_shapes,
        exponents,
        max_size,
        prev_preconditioners,
    ):
        """Computes preconditioners for given statistics in states in PMAP mode.

        For quantization, each statistic is represented by three values:
          quantized matrix, diagonal, and bucket sizes, we run inverse pth-roots
          without ever recreating the original matrix in f32.

        Args:
          states: A list of optimizer states.
          step: Current step number
          statistics: A list of statistics for all variables (for every dim)
          num_statistics_per_state: Number of statistis per state to reconstruct
            output states.
          original_shapes: A list of shapes of the statistics.
          exponents: Exponent power to use for inverse-pth roots.
          max_size: Maximum dim of the statistics to pad.
          prev_preconditioners: Previously available preconditioner.

        Returns:
          New optimizer states after computing the preconditioner.
        """
        num_devices = lax.psum(1, batch_axis_name)
        num_statistics = len(statistics)
        quantized_dtype = quantized_dtype_for_second_moment_statistics_buffers()
        # Complexity here is around: shapes needing be statically shaped,
        # our custom quantization type requires a different type of packing.

        # Parallel tensors:
        # quantized [dxd]
        # diagonals [d] f32
        # bucket_sizes [d] f32
        packed_quantized_statistics = [pad_square_matrix(stat.quantized, max_size) for stat in statistics]
        packed_quantized_diagonals = [pad_vector(stat.diagonal, max_size) for stat in statistics]
        packed_quantized_bucket_sizes = [pad_vector(stat.bucket_size, max_size) for stat in statistics]

        to_pad = -num_statistics % num_devices
        padded_eye = jnp.eye(max_size, dtype=jnp.float32)
        quantized_eye = QuantizedValue.from_float_value(padded_eye, quantized_dtype, True)
        packed_quantized_statistics.extend([quantized_eye.quantized for _ in range(to_pad)])
        packed_quantized_diagonals.extend([quantized_eye.diagonal for _ in range(to_pad)])
        packed_quantized_bucket_sizes.extend([quantized_eye.bucket_size for _ in range(to_pad)])
        exponents.extend([1 for _ in range(to_pad)])

        if not packed_quantized_statistics:
            return states

        all_quantized_statistics = batch(packed_quantized_statistics, num_devices)
        all_quantized_diagonals = batch(packed_quantized_diagonals, num_devices)
        all_quantized_bucket_sizes = batch(packed_quantized_bucket_sizes, num_devices)
        all_exponents = batch(exponents, num_devices)

        def _internal_inverse_pth_root_all():
            current_replica = lax.axis_index(batch_axis_name)
            (
                quantized_preconditioners,
                quantized_diagonals,
                quantized_bucket_sizes,
                errors,
            ) = _quantized_matrix_inverse_pth_root_vmap(
                all_quantized_statistics[current_replica],
                all_quantized_diagonals[current_replica],
                all_quantized_bucket_sizes[current_replica],
                all_exponents[current_replica],
            )
            quantized_preconditioners = jax.lax.all_gather(quantized_preconditioners, batch_axis_name)
            quantized_diagonals = jax.lax.all_gather(quantized_diagonals, batch_axis_name)
            quantized_bucket_sizes = jax.lax.all_gather(quantized_bucket_sizes, batch_axis_name)
            errors = jax.lax.all_gather(errors, batch_axis_name)
            quantized_preconditioners_flat = unbatch(quantized_preconditioners)
            quantized_diagonals_flat = unbatch(quantized_diagonals)
            quantized_bucket_sizes_flat = unbatch(quantized_bucket_sizes)
            errors_flat = unbatch(errors)
            return (
                quantized_preconditioners_flat,
                quantized_diagonals_flat,
                quantized_bucket_sizes_flat,
                errors_flat,
            )

        if preconditioning_compute_steps == 1:
            (
                quantized_preconditioners_flat,
                quantized_diagonals_flat,
                quantized_bucket_sizes_flat,
                errors_flat,
            ) = _internal_inverse_pth_root_all()
        else:
            # Passing statistics instead of preconditioners as they are similarly
            # shaped tensors. Note statistics will be ignored as we are passing in
            # a large init value for error.
            quantized_preconditioners_init = packed_quantized_statistics
            quantized_diagonals_init = packed_quantized_diagonals
            quantized_bucket_sizes_init = packed_quantized_bucket_sizes
            errors_init = [inverse_failure_threshold] * len(quantized_preconditioners_init)
            init_state = [
                quantized_preconditioners_init,
                quantized_diagonals_init,
                quantized_bucket_sizes_init,
                errors_init,
            ]
            perform_step = step % preconditioning_compute_steps == 0
            (
                quantized_preconditioners_flat,
                quantized_diagonals_flat,
                quantized_bucket_sizes_flat,
                errors_flat,
            ) = efficient_cond(perform_step, _internal_inverse_pth_root_all, init_state)

        def _skip(error):
            condition = jnp.logical_or(jnp.isnan(error), error >= inverse_failure_threshold)
            return condition.astype(error.dtype)

        def _select_preconditioner(error, new_p, old_p):
            return lax.cond(_skip(error), lambda _: old_p, lambda _: new_p, operand=None)

        new_quantized_preconditioners_flat = []
        new_quantized_diagonals_flat = []
        new_quantized_bucket_sizes_flat = []
        new_errors_flat = []
        for p, d, b, shape, prev_p, error in zip(
            quantized_preconditioners_flat,
            quantized_diagonals_flat,
            quantized_bucket_sizes_flat,
            original_shapes,
            prev_preconditioners,
            errors_flat,
        ):
            new_quantized_preconditioners_flat.append(
                _select_preconditioner(error, p[: shape[0], : shape[1]], prev_p.quantized)
            )
            new_quantized_diagonals_flat.append(_select_preconditioner(error, d[: shape[0]], prev_p.diagonal))
            new_quantized_bucket_sizes_flat.append(_select_preconditioner(error, b[: shape[0]], prev_p.bucket_size))
            new_errors_flat.append(error)

        assert len(states) == len(num_statistics_per_state)
        assert len(new_quantized_preconditioners_flat) == num_statistics
        assert len(new_quantized_diagonals_flat) == num_statistics
        assert len(new_quantized_bucket_sizes_flat) == num_statistics

        # Add back empty preconditioners so we that we can set the optimizer state.
        preconditioners_for_states = []
        errors_for_states = []
        idx = 0
        for num_statistics, state in zip(num_statistics_per_state, states):
            if num_statistics == 0:
                preconditioners_for_states.append([])
                errors_for_states.append(jnp.array(0, jnp.float32))
            else:
                quantized_preconditioners_for_state = new_quantized_preconditioners_flat[idx : idx + num_statistics]
                quantized_diagonals_for_state = new_quantized_diagonals_flat[idx : idx + num_statistics]
                quantized_bucket_sizes_for_state = new_quantized_bucket_sizes_flat[idx : idx + num_statistics]
                errors_for_state = jnp.stack(new_errors_flat[idx : idx + num_statistics])

                assert len(state.statistics) == len(quantized_preconditioners_for_state)
                assert len(state.statistics) == len(quantized_diagonals_for_state)
                assert len(state.statistics) == len(quantized_bucket_sizes_for_state)
                assert len(state.statistics) == len(errors_for_state)

                quantized_preconditioners = []
                for qv, qd, qb in zip(
                    quantized_preconditioners_for_state,
                    quantized_diagonals_for_state,
                    quantized_bucket_sizes_for_state,
                ):
                    quantized_preconditioners.append(QuantizedValue(qv, qd, qb, qv.dtype, True, list(qv.shape)))
                preconditioners_for_states.append(quantized_preconditioners)
                errors_for_states.append(errors_for_state)
                idx += num_statistics
        new_states = []
        for state, new_preconditioners, new_errors in zip(states, preconditioners_for_states, errors_for_states):
            if state.statistics:
                new_errors = jnp.where(
                    jnp.logical_and(new_errors > 0.0, new_errors != inverse_failure_threshold),
                    new_errors,
                    state.training_metrics.inverse_pth_root_errors,
                )
            new_training_metrics = TrainingMetrics(new_errors)
            new_states.append(
                ParameterStats(
                    state.diagonal_statistics,
                    state.statistics,
                    new_preconditioners,
                    state.diagonal_momentum,
                    state.momentum,
                    new_training_metrics,
                )
            )

        return new_states

    def _pjit_compute_preconditioners(
        states,
        step,
        statistics,
        num_statistics_per_state,
        original_shapes,
        exponents,
        max_size,
        prev_preconditioners,
    ):
        """Computes preconditioners for given statistics in states in PJIT mode.

        Args:
          states: A list of optimizer states.
          step: Current step number
          statistics: A list of statistics for all variables (for every dim)
          num_statistics_per_state: Number of statistis per state to reconstruct
            output states.
          original_shapes: A list of shapes of the statistics.
          exponents: Exponent power to use for inverse-pth roots.
          max_size: Maximum dim of the statistics to pad.
          prev_preconditioners: Previously available preconditioner.

        Returns:
          New optimizer states after computing the preconditioner.
        """
        num_statistics = len(statistics)
        to_pad = -num_statistics % num_devices_for_pjit
        padded_statistics = [pad_square_matrix(stat, max_size) for stat in statistics]
        padded_statistics.extend([jnp.eye(max_size, dtype=padded_statistics[0].dtype) for _ in range(to_pad)])
        exponents.extend([1 for _ in range(to_pad)])
        all_statistics = jnp.stack(padded_statistics)
        all_exponents = jnp.stack(exponents)

        def _internal_inverse_pth_root_all():
            preconditioners, errors = _matrix_inverse_pth_root_pjit(all_statistics, all_exponents)
            b1 = preconditioners.shape[0]

            def split(batched_values):
                return [jnp.squeeze(v) for v in jnp.split(batched_values, indices_or_sections=b1, axis=0)]

            return split(preconditioners), split(errors)

        if preconditioning_compute_steps == 1:
            preconditioners_flat, errors_flat = _internal_inverse_pth_root_all()
        else:
            # Passing statistics instead of preconditioners as they are similarly
            # shaped tensors. Note statistics will be ignored as we are passing in
            # a large init value for error.
            preconditioners_init = padded_statistics
            errors_init = [inverse_failure_threshold] * len(padded_statistics)
            init_state = [preconditioners_init, errors_init]
            perform_step = step % preconditioning_compute_steps == 0
            preconditioners_flat, errors_flat = efficient_cond(
                perform_step, _internal_inverse_pth_root_all, init_state
            )

        def _skip(error):
            condition = jnp.logical_or(jnp.isnan(error), error >= inverse_failure_threshold)
            return condition.astype(error.dtype)

        def _select_preconditioner(error, new_p, old_p):
            return lax.cond(_skip(error), lambda _: old_p, lambda _: new_p, operand=None)

        new_preconditioners_flat = []
        new_errors_flat = []
        for p, shape, prev_p, error in zip(preconditioners_flat, original_shapes, prev_preconditioners, errors_flat):
            new_preconditioners_flat.append(_select_preconditioner(error, p[: shape[0], : shape[1]], prev_p))
            new_errors_flat.append(error)

        assert len(states) == len(num_statistics_per_state)
        assert len(new_preconditioners_flat) == num_statistics

        # Add back empty preconditioners so we that we can set the optimizer state.
        preconditioners_for_states = []
        errors_for_states = []
        idx = 0
        for num_statistics, state in zip(num_statistics_per_state, states):
            if num_statistics == 0:
                preconditioners_for_states.append([])
                errors_for_states.append(jnp.array(0, jnp.float32))
            else:
                preconditioners_for_state = new_preconditioners_flat[idx : idx + num_statistics]
                assert len(state.statistics) == len(preconditioners_for_state)
                preconditioners_for_states.append(preconditioners_for_state)

                errors_for_state = jnp.stack(new_errors_flat[idx : idx + num_statistics])
                assert len(state.statistics) == len(errors_for_state)
                errors_for_states.append(errors_for_state)
                idx += num_statistics

        new_states = []
        for state, new_preconditioners, new_errors in zip(states, preconditioners_for_states, errors_for_states):
            if state.statistics:
                new_errors = jnp.where(
                    jnp.logical_and(new_errors > 0.0, new_errors != inverse_failure_threshold),
                    new_errors,
                    state.training_metrics.inverse_pth_root_errors,
                )
            new_training_metrics = TrainingMetrics(new_errors)
            new_states.append(
                ParameterStats(
                    state.diagonal_statistics,
                    state.statistics,
                    new_preconditioners,
                    state.diagonal_momentum,
                    state.momentum,
                    new_training_metrics,
                )
            )

        return new_states

    def _compute_preconditioners(states, params, step):
        """Computes preconditioners for given statistics in states.

        Args:
          states: A list of optimizer states.
          params: A list of params.
          step: Current step number

        Returns:
          New optimizer states after computing the preconditioner.
        """
        statistics = []
        num_statistics_per_state = []
        original_shapes = []
        exponents = []
        max_size = 0
        prev_preconditioners = []

        for state, param in zip(states, params):
            num_statistics = len(state.statistics)
            num_statistics_per_state.append(num_statistics)
            original_shapes_for_state = []
            if num_statistics > 0:
                preconditioner = preconditioner_from_params(param)
                for statistic in state.statistics:
                    exponents.append(
                        preconditioner.exponent_for_preconditioner() if exponent_override == 0 else exponent_override
                    )
                    original_shapes_for_state.append(statistic.shape)
                    max_size = max(max_size, statistic.shape[0])

                statistics.extend(state.statistics)
                prev_preconditioners.extend(state.preconditioners)
                original_shapes.extend(original_shapes_for_state)

        if not shard_optimizer_states:
            # Quantization is only enabled if batch_axis_name is not set.
            quantized_dtype = quantized_dtype_for_second_moment_statistics_buffers()

            if quantized_dtype == jnp.float32:
                return _pmap_compute_preconditioners(
                    states,
                    step,
                    statistics,
                    num_statistics_per_state,
                    original_shapes,
                    exponents,
                    max_size,
                    prev_preconditioners,
                )
            else:
                return _pmap_quantized_compute_preconditioners(
                    states,
                    step,
                    statistics,
                    num_statistics_per_state,
                    original_shapes,
                    exponents,
                    max_size,
                    prev_preconditioners,
                )

        else:
            return _pjit_compute_preconditioners(
                states,
                step,
                statistics,
                num_statistics_per_state,
                original_shapes,
                exponents,
                max_size,
                prev_preconditioners,
            )

    def _transform_grad(grad, state, param, step):
        """Transform per-parameter gradients."""
        preconditioner = preconditioner_from_params(param)
        sgd_update = grad
        new_diagonal_statistics = state.diagonal_statistics.to_float()

        if graft_type == GraftingType.ADAGRAD or graft_type == GraftingType.ADAGRAD_NORMALIZED:
            scaled_grad = grad
            if graft_type == GraftingType.ADAGRAD_NORMALIZED:
                scaled_grad = grad / (jnp.linalg.norm(grad) + 1e-16)

            new_diagonal_statistics = state.diagonal_statistics.to_float() + jnp.square(scaled_grad)
            adagrad_update = scaled_grad / (jnp.sqrt(new_diagonal_statistics) + diagonal_epsilon)
            grafting_update = adagrad_update
        elif graft_type == GraftingType.RMSPROP or graft_type == GraftingType.RMSPROP_NORMALIZED:
            scaled_grad = grad
            if graft_type == GraftingType.RMSPROP_NORMALIZED:
                scaled_grad = grad / (jnp.linalg.norm(grad) + 1e-16)

            w1 = beta2
            w2 = beta2 if beta2 == 1.0 else (1.0 - beta2)

            new_diagonal_statistics = w1 * state.diagonal_statistics.to_float() + w2 * jnp.square(scaled_grad)
            rmsprop_update = scaled_grad / (jnp.sqrt(new_diagonal_statistics) + diagonal_epsilon)

            if clip_by_scaled_gradient_norm:
                scaled_grad_norm = jnp.linalg.norm(rmsprop_update) / (jnp.sqrt(float(rmsprop_update.size)))
                clipping_denom = jnp.maximum(1.0, scaled_grad_norm / clip_by_scaled_gradient_norm)
                rmsprop_update /= clipping_denom

            grafting_update = rmsprop_update
        elif graft_type == GraftingType.SGD:
            grafting_update = sgd_update
        else:
            grafting_update = jnp.ones_like(sgd_update) * jnp.sign(sgd_update)

        lr = learning_rate
        if callable(learning_rate):
            lr = learning_rate(step)

        preconditioner_multiplier = lr if not decoupled_learning_rate else 1.0
        grafting_update = grafting_update * preconditioner_multiplier

        precond_grad = grad
        if not _skip_preconditioning(param):
            precond_grad = preconditioner.preconditioned_grad(
                precond_grad, _maybe_dequantize_preconditioners(state.preconditioners)
            )
        else:
            precond_grad = grafting_update

        grafting_update_norm = jnp.linalg.norm(grafting_update)
        precond_grad_norm = jnp.linalg.norm(precond_grad)

        multiplier = grafting_update_norm / (precond_grad_norm + 1e-16)
        shampoo_update = precond_grad * multiplier

        shampoo_update_with_wd = shampoo_update
        grafting_update_with_wd = grafting_update

        if weight_decay != 0 and not decoupled_weight_decay:
            shampoo_update_with_wd = shampoo_update + weight_decay * param
            grafting_update_with_wd = grafting_update + weight_decay * param

        w = (1.0 - beta1) if moving_average_for_momentum else 1.0

        shampoo_update_with_wd_momentum = state.momentum.to_float() * beta1 + w * shampoo_update_with_wd

        grafting_update_with_wd_momentum = state.diagonal_momentum.to_float() * beta1 + w * grafting_update_with_wd

        run_shampoo = (step >= start_preconditioning_step).astype(grafting_update_with_wd_momentum.dtype)

        momentum_update = (
            run_shampoo * shampoo_update_with_wd_momentum + (1.0 - run_shampoo) * grafting_update_with_wd_momentum
        )

        wd_update = run_shampoo * shampoo_update_with_wd + (1.0 - run_shampoo) * grafting_update_with_wd

        nesterov_momentum_update = momentum_update

        if nesterov:
            nesterov_momentum_update = w * wd_update + beta1 * momentum_update

        if weight_decay != 0 and decoupled_weight_decay:
            nesterov_momentum_update = nesterov_momentum_update + lr * weight_decay * param

        momentum_multiplier = lr if decoupled_learning_rate else 1.0
        transformed_update = -1.0 * momentum_multiplier * nesterov_momentum_update

        new_diagonal_momentum = grafting_update_with_wd_momentum
        new_momentum = shampoo_update_with_wd_momentum

        param_stats = ParameterStats(
            _quantize_diagonal_statistics(new_diagonal_statistics),
            state.statistics,
            state.preconditioners,
            _quantize_momentum(new_diagonal_momentum),
            _quantize_momentum(new_momentum),
            state.training_metrics,
        )

        return transformed_update, param_stats

    def update_fn(grads, state, params):
        """Transform the input gradient and update all statistics.

        Args:
          grads: the gradient tensors for the parameters
            and any custom gradients for preconditioners.
          state: a named tuple containing the state of the optimizer
          params: the parameters that should be updated.

        Returns:
          A tuple containing the new parameters and the new optimizer state.
        """
        params_flat, treedef = jax.tree_flatten(params)
        stats_flat = treedef.flatten_up_to(state.stats)
        grads_flat = treedef.flatten_up_to(grads)
        stats_grads = grads_flat

        new_stats_flat = jax.tree_map(
            lambda g, s, p: _compute_stats(g, s, p, state.count),
            stats_grads,
            stats_flat,
            params_flat,
        )

        new_stats_flat = _compute_preconditioners(new_stats_flat, params_flat, state.count)
        outputs = jax.tree_map(
            lambda g, s, p: _transform_grad(g, s, p, state.count),
            grads_flat,
            new_stats_flat,
            params_flat,
        )
        updates_flat, new_stats_flat = list(zip(*outputs)) if outputs else ((), ())

        updates = jax.tree_unflatten(treedef, updates_flat)
        new_stats = jax.tree_unflatten(treedef, new_stats_flat)

        new_state = ShampooState(count=state.count + 1, stats=new_stats)
        return updates, new_state

    if shard_optimizer_states:
        # Hijacks the init_fn signature so we can return an OptState with
        # appropriate init_fns.
        opt_init_fn = sharded_init_fn

        def _init_fns(unused_params):
            return InitFnState(
                init_fn=opt_init_fn,
                pspec_fn=sharded_init_partition_spec_fn,
                shape_and_dtype_fn=sharded_init_shape_and_dtype_fn,
            )

        opt_update_fn = sharded_update_fn
        return optax.GradientTransformation(_init_fns, opt_update_fn)
    else:
        return optax.GradientTransformation(init_fn, update_fn)

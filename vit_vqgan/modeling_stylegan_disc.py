# ------------------------------------------------------------------------------------
# Modified from stylegan2.jax (https://github.com/dsuess/stylegan2.jax)
# ------------------------------------------------------------------------------------

"""
[1] T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila,
    “Analyzing and Improving the Image Quality of StyleGAN,” arXiv:1912.04958
    [cs, eess, stat], Mar. 2020
"""


import functools as ft
from math import log2, sqrt
from typing import Callable, List, Tuple, Union

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from numpy import block
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel

from .configuration_stylegan_disc import StyleGANDiscriminatorConfig
from .utils import PretrainedFromWandbMixin

ActivationFunction = Callable[[jnp.ndarray], jnp.ndarray]


def _apply_filter_2d(
    x: jnp.ndarray,
    filter_kernel: jnp.ndarray,
    padding: Tuple[int, int] = (0, 0),
) -> jnp.ndarray:
    """
    >>> x = jnp.zeros((1, 64, 64, 2))
    >>> kernel = jnp.zeros((4, 4))
    >>> _apply_filter_2d(x, kernel, padding=(2, 2)).shape
    (1, 65, 65, 2)
    """
    dimension_numbers = ("NHWC", "HWOI", "NHWC")
    filter_kernel = filter_kernel[:, :, None, None]
    x = x[..., None]
    vmap_over_axis = 3

    conv_func = ft.partial(
        jax.lax.conv_general_dilated,
        rhs=filter_kernel,
        window_strides=(1, 1),
        padding=[padding, padding],
        dimension_numbers=dimension_numbers,
    )
    y = jax.vmap(conv_func, in_axes=vmap_over_axis, out_axes=vmap_over_axis)(x)
    return jnp.squeeze(y, axis=vmap_over_axis + 1)


class ConvDownsample2D(nn.Module):
    """This is the `_simple_upfirdn_2d` part of
    https://github.com/NVlabs/stylegan2-ada/blob/main/dnnlib/tflib/ops/upfirdn_2d.py#L313

    >>> module = _init(
    ...     ConvDownsample2D,
    ...     output_channels=8,
    ...     kernel_shape=3,
    ...     resample_kernel=jnp.array([1, 3, 3, 1]),
    ...     downsample_factor=2)
    >>> x = jax.numpy.zeros((1, 64, 64, 4))
    >>> params = module.init(jax.random.PRNGKey(0), x)
    >>> y = module.apply(params, None, x)
    >>> tuple(y.shape)
    (1, 32, 32, 8)
    """

    output_channels: int
    kernel_shape: Union[int, Tuple[int, int]]
    resample_kernel: jnp.array
    downsample_factor: int = 1
    gain: float = 1.0
    dtype: jnp.dtype = jnp.float32

    def setup(
        self,
    ):
        if self.resample_kernel.ndim == 1:
            resample_kernel = self.resample_kernel[:, None] * self.resample_kernel[None, :]
        elif 0 <= self.resample_kernel.ndim > 2:
            raise ValueError(f"Resample kernel has invalid shape {self.resample_kernel.shape}")

        self.conv = nn.Conv(
            self.output_channels,
            kernel_size=(self.kernel_shape, self.kernel_shape),
            strides=(self.downsample_factor, self.downsample_factor),
            padding="VALID",
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(),
        )
        self.resample_kernel_ = jnp.array(resample_kernel) * self.gain / resample_kernel.sum()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # pylint: disable=invalid-name
        kh, kw = self.resample_kernel_.shape
        ch, cw = self.kernel_shape, self.kernel_shape
        assert kh == kw
        assert ch == cw

        # See https://github.com/NVlabs/stylegan2-ada/blob/main/dnnlib/tflib/ops/upfirdn_2d.py#L362
        pad_0 = (kw - self.downsample_factor + cw) // 2
        pad_1 = (kw - self.downsample_factor + cw - 1) // 2
        y = _apply_filter_2d(
            x,
            self.resample_kernel_,
            padding=(pad_0, pad_1),
        )
        return self.conv(y)


def minibatch_stddev_layer(
    x: jnp.ndarray,
    group_size: int = None,
    num_new_features: int = 1,
) -> jnp.ndarray:
    """Minibatch standard deviation layer. Adds the standard deviation of
    subsets of size `group_size` taken over the batch dimension as features
    to x.

    Args:
        x ([type]): [description]
        group_size (int, optional): [description]. Defaults to None.
        num_new_features (int, optional): [description]. Defaults to 1.
        data_format (str, optional): [description]. Defaults to "channels_last".

    Returns:
        [type]: [description]

    >>> x = jnp.zeros((4, 23, 26, 3))
    >>> y = minibatch_stddev_layer(x, group_size=2, data_format=ChannelOrder.channels_last)
    >>> y.shape
    (4, 23, 26, 4)
    >>> x = jnp.zeros((4, 8, 23, 26))
    >>> y = minibatch_stddev_layer(x, num_new_features=4, data_format=ChannelOrder.channels_first)
    >>> y.shape
    (4, 12, 23, 26)

    FIXME Rewrite using allreduce ops like psum to allow non-batched definition
          of networks
    """
    # pylint: disable=invalid-name
    N, H, W, C = x.shape

    group_size = min(group_size, N) if group_size is not None else N
    C_ = C // num_new_features

    y = jnp.reshape(x, (group_size, -1, H, W, num_new_features, C_))

    y_centered = y - jnp.mean(y, axis=0, keepdims=True)
    y_std = jnp.sqrt(jnp.mean(y_centered * y_centered, axis=0) + 1e-8)

    y_std = jnp.mean(y_std, axis=(1, 2, 4))
    y_std = y_std.reshape((-1, 1, 1, num_new_features))
    y_std = jnp.tile(y_std, (group_size, H, W, 1))

    return jnp.concatenate((x, y_std), axis=3)


class DiscriminatorBlock(nn.Module):
    in_features: int
    out_features: int
    activation_function: ActivationFunction = jnn.leaky_relu
    resample_kernel: jnp.ndarray = jnp.array([1, 3, 3, 1])
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv_in = nn.Conv(
            self.in_features,
            kernel_size=(3, 3),
            padding="SAME",
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(),
        )
        self.downsampl1 = ConvDownsample2D(
            self.out_features,
            kernel_shape=3,
            resample_kernel=self.resample_kernel,
            downsample_factor=2,
            dtype=self.dtype,
        )
        self.downsample2 = ConvDownsample2D(
            self.out_features,
            kernel_shape=1,
            resample_kernel=self.resample_kernel,
            downsample_factor=2,
            dtype=self.dtype,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = self.conv_in(x)
        y = self.activation_function(y)
        y = self.downsampl1(y)
        y = self.activation_function(y)

        residual = self.downsample2(x)
        return (y + residual) / sqrt(2)


def _get_num_features(base_features: int, image_size: Tuple[int, int], max_hidden_feature_size: int) -> List[int]:
    """
    Gets number of features for the blocks. Each block includes a downsampling
    step by a factor of two and at the end, we want the resolution to be
    down to 4x4 (for square images)

    >>> features = _get_num_features(64, (512, 512), 1024)
    >>> 512 // 2**(len(features) - 1)
    4
    >>> features[3]
    512
    """
    for size in image_size:
        assert 2 ** int(log2(size)) == size, f"Image size must be a power of 2, got {image_size}"
    # determine the number of layers based on smaller side length
    shortest_side = min(*image_size)
    num_blocks = int(log2(shortest_side)) - 1
    num_features = (base_features * (2**i) for i in range(num_blocks))
    # we want to bring it down to 4x4 at the end of the last block
    return [min(n, max_hidden_feature_size) for n in num_features]


class StyleGANDiscriminatorModule(nn.Module):
    # pylint: disable=line-too-long
    """Residular discriminator architecture, see [1], Fig. 7c for details.

    Args:
        image_size (Union[int, Tuple[int, int]]): Size of the image. If
            only a single integer is passed, we assume square images.
            Each value must be a power of two.
        base_features(int): Number of features for the first convolutional layer.
            The i-th layer of the network is of size `base_features * 2**i`
        max_hidden_feature_size (int, optional): Maximum number of channels
            for intermediate convolutions. Defaults to 512.
        name (str, optional): Name of Haiku module. Defaults to None.

    >>> module = _init(ResidualDiscriminator, image_size=64, max_hidden_feature_size=16)
    >>> x = jnp.zeros((2, 64, 64, 3))
    >>> params = module.init(jax.random.PRNGKey(0), x)
    >>> y = module.apply(params, None, x)
    >>> y.shape
    (2, 1)
    >>> grad = _module_grad(module, params, None, x)
    >>> set(grad) == set(params)
    True

    TODO:
        - add Attention similar to https://github.com/lucidrains/stylegan2-pytorch/blob/54c79f430d0da3b02f570c6e1ef74d09190cd311/stylegan2_pytorch/stylegan2_pytorch.py#L557
        - add adaptive dropout: https://github.com/NVlabs/stylegan2-ada/blob/main/training/networks.py#L526
    """

    config: StyleGANDiscriminatorConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        image_size = self.config.image_size
        base_features = self.config.base_features
        max_hidden_feature_size = self.config.max_hidden_feature_size
        activation_function = jnn.leaky_relu
        mbstd_group_size = self.config.mbstd_group_size
        mbstd_num_features = self.config.mbstd_num_features

        size_t: Tuple[int, int] = (
            image_size
            if isinstance(image_size, (tuple, list))
            else (
                image_size,
                image_size,
            )
        )
        self.num_features = _get_num_features(2 * base_features, size_t, max_hidden_feature_size)
        self.activation_function = activation_function
        self.stddev_layer = ft.partial(
            minibatch_stddev_layer,
            group_size=mbstd_group_size,
            num_new_features=mbstd_num_features,
        )

        self.conv_in = nn.Conv(base_features, kernel_size=(1, 1), padding="SAME", dtype=self.dtype)

        blocks = []
        for _, (n_in, n_out) in enumerate(zip(self.num_features[1:], self.num_features)):
            blocks.append(
                DiscriminatorBlock(
                    n_in,
                    n_out,
                    activation_function=self.activation_function,
                    # dtype=self.dtype,
                )
            )

        self.blocks = blocks

        self.conv_out = nn.Conv(self.num_features[-2], kernel_size=(3, 3), padding="VALID", dtype=self.dtype)

        self.dense = nn.Dense(self.num_features[-1], dtype=self.dtype)
        self.classifier = nn.Dense(1, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = self.conv_in(x)
        y = self.activation_function(y)

        for block in self.blocks:
            y = block(y)

        # final block running on 4x4 feature maps
        assert min(y.shape[1:3]) == 4

        y = self.stddev_layer(y)
        y = self.conv_out(y)
        y = self.activation_function(y)
        y = jnp.reshape(y, (y.shape[0], -1))
        y = self.dense(y)
        y = self.activation_function(y)

        # Prediction head
        y = self.classifier(y)
        return y


class StyleGANDiscriminatorPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.
    """

    config_class = StyleGANDiscriminatorConfig
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: StyleGANDiscriminatorConfig,
        input_shape=(1, 256, 256, 3),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, **kwargs)
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        pixel_values = jnp.zeros(input_shape, dtype=jnp.float32)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(rngs, pixel_values)["params"]

    def __call__(
        self,
        pixel_values,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
    ):
        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values),
            # not train,
            rngs=rngs,
        )


class StyleGANDiscriminator(PretrainedFromWandbMixin, StyleGANDiscriminatorPreTrainedModel):
    module_class = StyleGANDiscriminatorModule

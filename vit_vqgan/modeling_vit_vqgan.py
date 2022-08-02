from functools import partial
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.linen.initializers import variance_scaling
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel

from .configuration_vit_vqgan import ViTVQConfig
from .utils import PretrainedFromWandbMixin

remat = nn_partitioning.remat


ACT2FN["tanh"] = nn.tanh

# copied from https://github.com/deepmind/dm-haiku/blob/3f31e279d4ce613ae3e47b97031f8b2d732071b7/haiku/_src/spectral_norm.py#L46
def l2_normalize(x, axis=None, eps=1e-12):
    """Normalizes along dimension `axis` using an L2 norm.
    This specialized function exists for numerical stability reasons.
    Args:
      x: An input ndarray.
      axis: Dimension along which to normalize, e.g. `1` to separately normalize
        vectors in a batch. Passing `None` views `t` as a flattened vector when
        calculating the norm (equivalent to Frobenius norm).
      eps: Epsilon to avoid dividing by zero.
    Returns:
      An array of the same shape as 'x' L2-normalized along 'axis'.
    """
    return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


# We could simply use einops for this, but I'm a little crazy.
def to_patches(image, patch_size):
    batch, height, _, channels = image.shape
    num_patches = (height // patch_size) ** 2
    patch_height = patch_width = height // patch_size

    patches = image.reshape(batch, patch_height, patch_size, patch_width, patch_size, channels)
    # (batch, patch_height, patch_width, patch_size, patch_size, channels)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    # (batch, patch_height*patch_width, patch_size * patch_size * channels)
    patches = patches.reshape(batch, num_patches, -1)
    return patches


def to_image(patches, patch_size):
    batch, num_patches, channels = patches.shape
    patch_height = patch_width = int(num_patches**0.5)
    image_size = patch_height * patch_size
    patches = patches.reshape(batch, patch_height, patch_width, channels)
    patches = patches.reshape(batch, patch_height, patch_width, patch_size, patch_size, -1)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(batch, image_size, image_size, -1)
    return patches


def patch_1D_to_2D(patch):
    _, n_patches, dim = patch.shape
    w_h = int(n_patches**0.5)
    assert w_h * w_h == n_patches
    return patch.reshape((-1, w_h, w_h, dim))


def patch_2D_to_1D(patch):
    _, w_h, _, dim = patch.shape
    return patch.reshape((-1, w_h * w_h, dim))


class ConvPatches(nn.Module):
    config: ViTVQConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, pixel_values):
        embed_dim = self.config.hidden_size
        patch_size = self.config.patch_size

        patch_embeds = nn.Conv(
            embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=self.config.use_bias,
            dtype=self.dtype,
            name="patch_embeds",
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )(pixel_values)
        batch_size, height, width, channels = patch_embeds.shape
        patch_embeds = jnp.reshape(patch_embeds, (batch_size, height * width, channels))
        return patch_embeds


class FeedForwardLayer(nn.Module):
    dim1: int
    dim2: int
    activation: str
    config: ViTVQConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, deterministic: bool = True):
        Dense = partial(
            nn.Dense,
            use_bias=self.config.use_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        hidden_states = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype, name="layernorm_0")(
            hidden_states
        )
        hidden_states = Dense(features=self.dim1, name="fc1")(hidden_states)
        hidden_states = ACT2FN[self.activation](hidden_states)

        if self.config.mid_ffn_conv:
            # suggestion from Katherine Crowson
            hidden_states = patch_1D_to_2D(hidden_states)
            hidden_states = nn.Conv(
                self.dim1,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                feature_group_count=self.dim1,
                use_bias=self.config.use_bias,
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                name="mid_ffn_conv",
            )(hidden_states)
            hidden_states = patch_2D_to_1D(hidden_states)

        if self.config.ln_positions == "normformer":
            hidden_states = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype, name="layernorm_1")(
                hidden_states
            )

        hidden_states = nn.Dropout(rate=self.config.dropout)(hidden_states, deterministic=deterministic)
        hidden_states = Dense(features=self.dim2, name="fc2")(hidden_states)
        hidden_states = nn.Dropout(rate=self.config.dropout)(hidden_states, deterministic=deterministic)
        return hidden_states


class GLU(nn.Module):
    dim1: int
    dim2: int
    activation: str
    config: ViTVQConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, deterministic: bool = True):
        Dense = partial(
            nn.Dense,
            use_bias=self.config.use_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        hidden_states = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype, name="layernorm_0")(
            hidden_states
        )

        hidden_gelu = Dense(features=self.dim1, name="fc1")(hidden_states)
        hidden_gelu = ACT2FN[self.activation](hidden_gelu)

        hidden_linear = Dense(features=self.dim1, name="fc2")(hidden_states)

        hidden_states = hidden_gelu * hidden_linear

        if self.config.mid_ffn_conv:
            # suggestion from Katherine Crowson
            hidden_states = patch_1D_to_2D(hidden_states)
            hidden_states = nn.Conv(
                self.dim1,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                feature_group_count=self.dim1,
                use_bias=self.config.use_bias,
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                name="mid_ffn_conv",
            )(hidden_states)
            hidden_states = patch_2D_to_1D(hidden_states)

        if self.config.ln_positions == "normformer":
            hidden_states = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype, name="layernorm_1")(
                hidden_states
            )

        hidden_states = nn.Dropout(rate=self.config.dropout)(hidden_states, deterministic=deterministic)
        hidden_states = Dense(features=self.dim2, name="fc_out")(hidden_states)
        hidden_states = nn.Dropout(rate=self.config.dropout)(hidden_states, deterministic=deterministic)
        return hidden_states


class Attention(nn.Module):
    config: ViTVQConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_dim = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = self.config.attention_dropout

        Dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.config.use_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        self.q_proj, self.k_proj, self.v_proj = Dense(), Dense(), Dense()
        self.out_proj = Dense()

        if self.config.post_attention_conv:
            # suggestion from Phil Wang
            self.post_attn_conv = nn.Conv(
                self.embed_dim,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                feature_group_count=self.embed_dim,
                use_bias=self.config.use_bias,
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(self, hidden_states, deterministic: bool = True):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=None,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        if self.config.post_attention_conv:
            attn_output = patch_1D_to_2D(attn_output)
            attn_output = self.post_attn_conv(attn_output)
            attn_output = patch_2D_to_1D(attn_output)

        return attn_output


class TransformerBlock(nn.Module):
    config: ViTVQConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, deterministic: bool = True):
        residual = hidden_states

        hidden_states = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)(hidden_states)
        hidden_states = Attention(self.config, dtype=self.dtype)(
            hidden_states=hidden_states, deterministic=deterministic
        )

        if self.config.ln_positions == "normformer":
            hidden_states = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)(hidden_states)
        hidden_states = residual + hidden_states

        FFN = GLU if self.config.use_glu else FeedForwardLayer
        residual = hidden_states
        hidden_states = FFN(
            dim1=self.config.intermediate_size,
            dim2=self.config.hidden_size,
            activation=self.config.hidden_act,
            config=self.config,
            dtype=self.dtype,
        )(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states

        if self.config.use_scan:
            return hidden_states, ()

        return hidden_states


class Transformer(nn.Module):
    num_layers: int
    config: ViTVQConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, deterministic: bool = True):
        layer = (
            remat(TransformerBlock, static_argnums=(1,), prevent_cse=not self.config.use_scan)
            if self.config.gradient_checkpointing
            else TransformerBlock
        )
        if self.config.use_scan:
            hidden_states, _ = nn.scan(
                layer,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=(nn.broadcast,),
                length=self.num_layers,
            )(self.config, dtype=self.dtype, name="scanned")(hidden_states, deterministic)
        else:
            for i in range(self.num_layers):
                hidden_states = layer(self.config, name=str(i), dtype=self.dtype)(hidden_states, deterministic)
        return hidden_states


class VitEncoder(nn.Module):
    config: ViTVQConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, pixel_values, deterministic: bool = True):
        if self.config.use_conv_patches:
            hidden_states = ConvPatches(self.config, dtype=self.dtype)(pixel_values)
        else:
            hidden_states = partial(to_patches, patch_size=self.config.patch_size)(pixel_values)
            hidden_states = nn.Dense(
                self.config.hidden_size,
                use_bias=self.config.use_bias,
                dtype=self.dtype,
                name="to_hidden",
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            )(hidden_states)
        assert hidden_states.ndim == 3

        num_patches = (self.config.image_size // self.config.patch_size) ** 2
        position_embeddings = self.param(
            "pos_embedding",
            jax.nn.initializers.normal(self.config.initializer_range),
            (1, num_patches, self.config.hidden_size),
        )
        hidden_states += position_embeddings

        hidden_states = nn.Dropout(rate=self.config.dropout)(hidden_states, deterministic=deterministic)
        hidden_states = Transformer(self.config.num_encoder_layers, self.config, dtype=self.dtype)(
            hidden_states, deterministic=deterministic
        )
        return hidden_states


class VitDecoder(nn.Module):
    config: ViTVQConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, deterministic: bool = True):
        assert hidden_states.ndim == 3
        num_patches = (self.config.image_size // self.config.patch_size) ** 2
        position_embeddings = self.param(
            "pos_embedding",
            jax.nn.initializers.normal(self.config.initializer_range),
            (1, num_patches, self.config.hidden_size),
        )
        hidden_states += position_embeddings
        hidden_states = Transformer(self.config.num_decoder_layers, self.config, dtype=self.dtype)(
            hidden_states, deterministic=deterministic
        )
        return hidden_states


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    config: ViTVQConfig

    def setup(self):
        embed_init = variance_scaling(1.0, "fan_in", "normal", out_axis=0)
        self.embedding = self.param(
            "codebook_embedding",
            embed_init,
            (self.config.n_embed, self.config.codebook_embed_dim),
            jnp.float32,
        )

    def __call__(self, hidden_states):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        #  flatten
        hidden_states_flattened = hidden_states.reshape((-1, self.config.codebook_embed_dim))

        # normalize `z` here `hidden_states` and codebook latent variable `e` (here `embedding`)
        hidden_states_flattened = l2_normalize(hidden_states_flattened, axis=1)
        embedding = l2_normalize(self.embedding, axis=1)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        distance = (
            jnp.sum(hidden_states_flattened**2, axis=1, keepdims=True)
            + jnp.sum(embedding**2, axis=1)
            - 2 * jnp.dot(hidden_states_flattened, embedding.T)
        )

        # get quantized latent vectors
        min_encoding_indices = jnp.argmin(distance, axis=1)
        # reshape to (batch, num_tokens)
        min_encoding_indices = min_encoding_indices.reshape(hidden_states.shape[0], -1)
        z_q = self.get_codebook_entry(min_encoding_indices)

        hidden_states_normed = l2_normalize(hidden_states, axis=-1)
        e_latent_loss = jnp.mean(jnp.square(jax.lax.stop_gradient(z_q) - hidden_states_normed))
        q_latent_loss = jnp.mean(jnp.square(z_q - jax.lax.stop_gradient(hidden_states_normed)))

        # Straight Through Estimator
        z_q = hidden_states + jax.lax.stop_gradient(z_q - hidden_states)

        return z_q, min_encoding_indices, (q_latent_loss, e_latent_loss)

    def get_codebook_entry(self, indices):
        # indices are expected to be of shape (batch, num_tokens)
        # get quantized latent vectors
        z_q = jnp.take(self.embedding, indices, axis=0)
        # normalize latent variable (Ze(x) in the paper)
        z_q = l2_normalize(z_q, axis=-1)
        return z_q


class VitVQModule(nn.Module):
    config: ViTVQConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        input_dim = self.config.num_channels * (self.config.patch_size**2)
        self.latent_size = self.config.image_size // self.config.patch_size

        self.encoder = VitEncoder(self.config, dtype=self.dtype)
        self.decoder = VitDecoder(self.config, dtype=self.dtype)

        self.quantizer = VectorQuantizer(self.config)

        self.factor_in = nn.Dense(
            self.config.codebook_embed_dim,
            use_bias=self.config.use_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.factor_out = nn.Dense(
            self.config.hidden_size,
            use_bias=self.config.use_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        self.encoder_ln = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype, use_scale=False)

        if self.config.use_conv_patches:
            self.to_image = nn.ConvTranspose(
                self.config.num_channels,
                kernel_size=(self.config.patch_size, self.config.patch_size),
                strides=(self.config.patch_size, self.config.patch_size),
                padding="VALID",
                use_bias=self.config.use_bias,
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            )
        else:
            self.to_patches = FeedForwardLayer(
                dim1=self.config.intermediate_size,
                dim2=input_dim,
                activation=self.config.hidden_act,
                config=self.config,
                dtype=self.dtype,
            )

    def encode(self, pixel_values, deterministic: bool = True):
        # 1. create patches and encode
        hidden_states = self.encoder(pixel_values, deterministic=deterministic)
        hidden_states = ACT2FN[self.config.hidden_act](hidden_states)

        # 2. Project the embeddings to the codebook embedding space and quantize
        # this corresponds to the factorized codebook from the paper https://arxiv.org/abs/2110.04627 section 3.2
        hidden_states = self.factor_in(hidden_states)
        hidden_states = self.encoder_ln(hidden_states)
        quant_states, indices, losses = self.quantizer(hidden_states)

        return quant_states, indices, losses

    def decode(self, hidden_states, deterministic: bool = True):
        # 3. Project the quantized embeddings back to the original space
        hidden_states = self.factor_out(hidden_states)

        # 4. Decode the quantized embeddings back into the pixel space
        hidden_states = self.decoder(hidden_states, deterministic=deterministic)

        # 5. Reconstruct the image from the patches
        if self.config.use_conv_patches:
            batch, _, channels = hidden_states.shape
            hidden_states = hidden_states.reshape(batch, self.latent_size, self.latent_size, channels)
            pixel_values = self.to_image(hidden_states)
        else:
            patches = self.to_patches(hidden_states)
            pixel_values = to_image(patches, self.config.patch_size)

        return pixel_values

    def decode_code(self, code_b):
        hidden_states = self.quantizer.get_codebook_entry(code_b)
        pixel_values = self.decode(hidden_states)
        return pixel_values

    def __call__(self, pixel_values, deterministic: bool = True):
        quant_states, _, losses = self.encode(pixel_values, deterministic=deterministic)
        pixel_values = self.decode(quant_states, deterministic=deterministic)
        return pixel_values, losses


class ViTVQGANPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.
    """

    config_class = ViTVQConfig
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: ViTVQConfig,
        input_shape=(1, 256, 256, 3),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
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
        input_shape = (
            1,
            self.config.image_size,
            self.config.image_size,
            self.config.num_channels,
        )
        pixel_values = jnp.zeros(input_shape, dtype=jnp.float32)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(rngs, pixel_values)["params"]

    def num_params(self, params=None):
        if params is None:
            params = self.params
        num_params = jax.tree_util.tree_map(lambda param: param.size, flatten_dict(unfreeze(params))).values()
        return sum(list(num_params))

    def unscan(self, params):
        if self.config.use_scan:
            self.config.use_scan = False
            params = flatten_dict(params)
            scanned_keys = [k for k in params.keys() if "scanned" in k]
            for k in scanned_keys:
                v = params[k]
                name_idx = k.index("scanned")
                for i in range(len(v)):
                    new_k = (
                        *k[:name_idx],
                        f"{i}",
                        *k[name_idx + 1 :],
                    )
                    params[new_k] = v[i]
                del params[k]
            params = unflatten_dict(params)
        return params

    def encode(
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
            not train,
            rngs=rngs,
            method=self.module.encode,
        )

    def decode(
        self,
        hidden_states,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
    ):
        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(hidden_states),
            not train,
            rngs=rngs,
            method=self.module.decode,
        )

    def decode_code(self, indices, params: dict = None):
        return self.module.apply(
            {"params": params or self.params},
            jnp.array(indices, dtype="i4"),
            method=self.module.decode_code,
        )

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
            not train,
            rngs=rngs,
        )


class ViTVQModel(PretrainedFromWandbMixin, ViTVQGANPreTrainedModel):
    module_class = VitVQModule

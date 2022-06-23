import math
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.linen.attention import dot_product_attention_weights
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel

from .configuration_vit_vqgan import ViTVQGANConfig

ACT2FN["tanh"] = nn.tanh


class VisionEmbeddings(nn.Module):
    config: ViTVQGANConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        embed_dim = self.config.hidden_size
        image_size = self.config.image_size
        patch_size = self.config.patch_size

        self.patch_embedding = nn.Conv(
            embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(),
        )

        self.num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Embed(self.num_patches, embed_dim, embedding_init=jax.nn.initializers.normal())
        self.position_ids = jnp.expand_dims(jnp.arange(0, self.num_patches, dtype="i4"), axis=0)

    def __call__(self, pixel_values):
        patch_embeds = self.patch_embedding(pixel_values)
        batch_size, height, width, channels = patch_embeds.shape
        patch_embeds = jnp.reshape(patch_embeds, (batch_size, height * width, channels))

        embeddings = patch_embeds + self.position_embedding(self.position_ids)
        return embeddings


class FeedForwardLayer(nn.Module):
    config: ViTVQGANConfig
    activation: str = "relu"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.activation_fn = ACT2FN[self.activation]
        self.fc1 = nn.Dense(
            self.config.intermediate_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.01),
        )
        self.fc2 = nn.Dense(self.config.hidden_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))

    def __call__(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Attention(nn.Module):
    config: ViTVQGANConfig
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

        self.k_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.v_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.q_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.out_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))

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

        return attn_output


class TransformerBlock(nn.Module):
    config: ViTVQGANConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self_attn = Attention(self.config, dtype=self.dtype)
        self.layer_norm1 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.feed_forward = FeedForwardLayer(self.config, activation=self.config.hidden_act, dtype=self.dtype)
        self.layer_norm2 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic: bool = True):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Transformer(nn.Module):
    config: ViTVQGANConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [
            TransformerBlock(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
        ]
        self.feed_forward = FeedForwardLayer(
            self.config, activation=self.config.extra_feed_forward_act, dtype=self.dtype
        )

    def __call__(self, hidden_states, deterministic: bool = True):
        for layer in self.layers:
            hidden_states = layer(hidden_states, deterministic=deterministic)

        hidden_states = self.feed_forward(hidden_states)

        return hidden_states


class VitEncoder(nn.Module):
    config: ViTVQGANConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embeddings = VisionEmbeddings(self.config, dtype=self.dtype)
        self.transformer = Transformer(self.config, dtype=self.dtype)

    def __call__(self, pixel_values, deterministic: bool = True):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.transformer(hidden_states, deterministic=deterministic)
        return hidden_states


class VitDecoder(nn.Module):
    config: ViTVQGANConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.num_patches = (self.config.image_size // self.config.patch_size) ** 2
        self.latent_size = self.config.image_size // self.config.patch_size
        self.position_ids = jnp.expand_dims(jnp.arange(0, self.num_patches, dtype="i4"), axis=0)

        self.position_embeddings = nn.Embed(
            self.num_patches, self.config.hidden_size, embedding_init=jax.nn.initializers.normal()
        )
        self.transformer = Transformer(self.config, dtype=self.dtype)

        self.to_image = nn.ConvTranspose(
            self.config.num_channels,
            kernel_size=(self.config.patch_size, self.config.patch_size),
            strides=(self.config.patch_size, self.config.patch_size),
            padding="VALID",
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(),
        )

    def __call__(self, hidden_states, deterministic: bool = True):
        hidden_states = hidden_states + self.position_embeddings(self.position_ids)
        hidden_states = self.transformer(hidden_states, deterministic=deterministic)

        batch, _, channels = hidden_states.shape
        hidden_states = hidden_states.reshape(batch, self.latent_size, self.latent_size, channels)
        pixel_values = self.to_image(hidden_states)

        return pixel_values


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

    config: ViTVQGANConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embedding = nn.Embed(self.config.n_embed, self.config.codebook_embed_dim, dtype=self.dtype)  # TODO: init

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
        hidden_states_flattended = hidden_states.reshape((-1, self.config.codebook_embed_dim))

        # dummy op to init the weights, so we can access them below
        self.embedding(jnp.ones((1, 1), dtype="i4"))

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        emb_weights = self.variables["params"]["embedding"]["embedding"]
        distance = (
            jnp.sum(hidden_states_flattended**2, axis=1, keepdims=True)
            + jnp.sum(emb_weights**2, axis=1)
            - 2 * jnp.dot(hidden_states_flattended, emb_weights.T)
        )

        # get quantized latent vectors
        min_encoding_indices = jnp.argmin(distance, axis=1)
        z_q = self.embedding(min_encoding_indices).reshape(hidden_states.shape)

        # reshape to (batch, num_tokens)
        min_encoding_indices = min_encoding_indices.reshape(hidden_states.shape[0], -1)

        # compute the codebook_loss (q_loss) outside the model
        # here we return the embeddings and indices
        return z_q, min_encoding_indices

    def get_codebook_entry(self, indices, shape=None):
        # indices are expected to be of shape (batch, num_tokens)
        # get quantized latent vectors
        batch, num_tokens = indices.shape
        z_q = self.embedding(indices)
        z_q = z_q.reshape(batch, int(math.sqrt(num_tokens)), int(math.sqrt(num_tokens)), -1)
        return z_q


class VitVQModule(nn.Module):
    config: ViTVQGANConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.encoder = VitEncoder(self.config, dtype=self.dtype)
        self.factor_in = nn.Dense(
            self.config.codebook_embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01)
        )
        self.quantizer = VectorQuantizer(self.config, dtype=self.dtype)
        self.factor_out = nn.Dense(
            self.config.hidden_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01)
        )
        self.decoder = VitDecoder(self.config, dtype=self.dtype)

    def encode(self, pixel_values, deterministic: bool = True):
        hidden_states = self.encoder(pixel_values, deterministic=deterministic)
        hidden_states = self.factor_in(hidden_states)
        quant_states, indices = self.quantize(hidden_states)
        return quant_states, indices

    def decode(self, hidden_states, deterministic: bool = True):
        hidden_states = self.factor_out(hidden_states)
        hidden_states = self.decoder(hidden_states, deterministic=deterministic)
        return hidden_states

    def decode_code(self, code_b):
        hidden_states = self.quantize.get_codebook_entry(code_b)
        hidden_states = self.decode(hidden_states)
        return hidden_states

    def __call__(self, pixel_values, deterministic: bool = True):
        hidden_states = self.encoder(pixel_values, deterministic=deterministic)
        hidden_states = self.factor_in(hidden_states)
        hidden_states, _ = self.quantizer(hidden_states)
        hidden_states = self.factor_out(hidden_states)
        hidden_states = self.decoder(hidden_states, deterministic=deterministic)
        return hidden_states


class ViTVQGANPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.
    """

    config_class = ViTVQGANConfig
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: ViTVQGANConfig,
        input_shape=(1, 256, 256, 3),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        pixel_values = jnp.zeros(input_shape, dtype=jnp.float32)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(rngs, pixel_values)["params"]

    def encode(self, pixel_values, params: dict = None, dropout_rng: jax.random.PRNGKey = None, train: bool = False):
        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params}, jnp.array(pixel_values), not train, rngs=rngs, method=self.module.encode
        )

    def decode(self, hidden_states, params: dict = None, dropout_rng: jax.random.PRNGKey = None, train: bool = False):
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
            {"params": params or self.params}, jnp.array(indices, dtype="i4"), method=self.module.decode_code
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


class ViTVQModel(ViTVQGANPreTrainedModel):
    module_class = VitVQModule

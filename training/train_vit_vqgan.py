import logging
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional

import flax
import jax
import jax.nn as nn
import jax.numpy as jnp
import jaxlib
import numpy as np
import optax
import transformers
import wandb
from flax import core, struct
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.serialization import from_bytes, to_bytes
from huggingface_hub import Repository
from jax.experimental import PartitionSpec, maps
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.experimental.pjit import pjit, with_sharding_constraint
from lpips_j.lpips import LPIPS
from tqdm import tqdm
from transformers import HfArgumentParser, set_seed
from transformers.utils import get_full_repo_name

from vit_vqgan import (StyleGANDiscriminator, StyleGANDiscriminatorConfig,
                       ViTVQConfig, ViTVQModel)
from vit_vqgan.data import Dataset

logger = logging.getLogger(__name__)

cc.initialize_cache("jax_cache")


class TrainState(struct.PyTreeNode):
    step: int
    generator_params: core.FrozenDict[str, Any]
    discriminator_params: core.FrozenDict[str, Any]
    lpips_params: core.FrozenDict[str, Any]
    optimizer: Any
    generator_opt_state: optax.OptState
    discriminator_opt_state: optax.OptState
    dropout_rng: jnp.ndarray = None
    epoch: int = 0

    @classmethod
    def create(cls, generator_params, discriminator_params, lpips_params, optimizer, **kwargs):
        generator_opt_state = optimizer.init(generator_params)
        discriminator_opt_state = optimizer.init(discriminator_params)
        return cls(
            step=0,
            generator_params=generator_params,
            discriminator_params=discriminator_params,
            lpips_params=lpips_params,
            optimizer=optimizer,
            generator_opt_state=generator_opt_state,
            discriminator_opt_state=discriminator_opt_state,
            **kwargs
        )

    def apply_gradients_generator(self, grads, **kwargs):
        updates, new_opt_state = self.optimizer.update(grads, self.generator_opt_state, self.generator_params)
        new_params = optax.apply_updates(self.generator_params, updates)
        return self.replace(
            step=self.step + 1, generator_params=new_params, generator_opt_state=new_opt_state, **kwargs
        )


class TrainStateInitializer:
    def __init__(self, generator, discriminator, lpips, optimizer):
        def initialize_train_state(rng: jax.random.PRNGKey):
            generator_params = generator.init_weights(generator.key, generator.input_shape)
            discriminator_params = discriminator.init_weights(discriminator.key, discriminator.input_shape)
            lpips_params = init_lpips(rng, generator.image_size)
            return TrainState.create(
                generator_params,
                discriminator_params,
                lpips_params,
                optimizer,
            )

        self._initialize_train_state = initialize_train_state

    def from_scratch(self, init_rng: jax.random.PRNGKey):
        return self._initialize_train_state(init_rng)


# define lpips
lpips_fn = LPIPS()


def init_lpips(rng, image_size):
    x = jax.random.normal(rng, shape=(1, image_size, image_size, 3))
    return lpips_fn.init(rng, x, x)


generator_config = ViTVQConfig(num_hidden_layers=2)
generator = ViTVQModel(generator_config, _do_init=False)

discriminator_config = StyleGANDiscriminatorConfig()
discriminator = StyleGANDiscriminator(discriminator_config, _do_init=False)


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (jnp.mean(nn.softplus(-logits_real)) + jnp.mean(nn.softplus(logits_fake)))
    return d_loss


def generator_train_step(train_state: TrainState, batch):
    def loss_fn(generator_params):
        predicted_images, (q_latent_loss, e_latent_loss) = generator(
            batch, params=generator_params, dropout_rng=train_state.dropout_rng, train=True
        )

        # TODO: replace l1 with logit laplace
        loss_l1 = jnp.mean(jnp.abs(predicted_images - batch))
        loss_l2 = jnp.mean((predicted_images - batch) ** 2)
        loss_lpips = jnp.mean(lpips_fn.apply(train_state.lpips_params, batch, predicted_images))

        codebook_loss = (generator.config.cost_q_latent * q_latent_loss) + (
            generator.config.cost_e_latent * e_latent_loss
        )
        reconstruction_loss = (generator.config.cost_l1 * loss_l1) + (generator.config.cost_l2 * loss_l2)

        # add perceptual loss
        perceptual_weight = 1.0  # TODO: make this an argument
        reconstruction_loss = reconstruction_loss + (perceptual_weight * loss_lpips)

        # run discriminator
        logits_fake = discriminator(predicted_images, params=train_state.discriminator_params, train=True)
        d_weight = 1.0  # ????
        disc_factor = 1.0  # TODO: make this an argument
        codebook_weight = 1.0  # TODO: make this an argument
        g_loss = -jnp.mean(logits_fake)

        loss = reconstruction_loss + d_weight * disc_factor * g_loss + codebook_weight * codebook_loss
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(train_state.generator_params)
    return loss, train_state.apply_gradients_generator(grads)


def discriminator_train_step(train_state: TrainState, batch):
    def loss_fn(discriminator_params):
        predicted_images, _ = generator(
            batch, params=train_state.generator_params, dropout_rng=train_state.dropout_rng, train=True
        )

        # run discriminator
        logits_real = discriminator(batch, params=discriminator_params, train=True)
        logits_fake = discriminator(predicted_images, params=discriminator_params, train=True)

        disc_factor = 1.0  # TODO: make this an argument
        loss = disc_factor * vanilla_d_loss(logits_real, logits_fake)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(train_state.discriminator_params)
    return loss, train_state.apply_gradients_discriminator(grads)


def train_step(train_state: TrainState, batch):
    generator_loss, train_state = generator_train_step(train_state, batch)
    discriminator_loss, train_state = discriminator_train_step(train_state, batch)
    return generator_loss, discriminator_loss, train_state

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
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import optax
import transformers
import wandb
from flax import core, struct
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.serialization import from_bytes, to_bytes
from flax.traverse_util import flatten_dict
from huggingface_hub import Repository
from jax.experimental import PartitionSpec, maps
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.experimental.pjit import pjit, with_sharding_constraint
from lpips_j.lpips import LPIPS
from scalable_shampoo.distributed_shampoo import (GraftingType,
                                                  distributed_shampoo)
from tqdm import tqdm
from transformers import HfArgumentParser, set_seed
from transformers.utils import get_full_repo_name

from vit_vqgan import (StyleGANDiscriminator, StyleGANDiscriminatorConfig,
                       ViTVQConfig, ViTVQModel)
from vit_vqgan.data import Dataset, logits_to_image

logger = logging.getLogger(__name__)

cc.initialize_cache("jax_cache")


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    batch_size_per_node: Optional[int] = field(default=64, metadata={"help": "Batch size for training."})

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing an update pass."},
    )
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Use gradient checkpointing."})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate."})
    disc_learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate."})
    optim: str = field(
        default="distributed_shampoo",
        metadata={"help": 'The optimizer to use. Can be "distributed_shampoo" (default) or "adam"'},
    )
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay applied to parameters."})
    beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for Adam & Distributed Shampoo."},
    )
    beta2: float = field(
        default=0.999,
        metadata={"help": "Beta2 for for Adam & Distributed Shampoo."},
    )
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    block_size: int = field(
        default=1024,
        metadata={"help": "Chunked size for large layers with Distributed Shampoo."},
    )
    preconditioning_compute_steps: int = field(
        default=10, metadata={"help": "Number of steps to update preconditioner."}
    )
    skip_preconditioning_dim_size_gt: int = field(
        default=4096,
        metadata={"help": "Max size for preconditioning with Distributed Shampoo."},
    )
    graft_type: str = field(
        default="rmsprop_normalized",
        metadata={
            "help": (
                "The type of grafting to use. Can be 'rmsprop_normalized' (default), 'rmsprop', 'adagrad',"
                " 'adagrad_normalized', 'sgd' or 'sqrt_n'"
            )
        },
    )
    nesterov: bool = field(
        default=False,
        metadata={"help": "Use Nesterov momentum for Distributed Shampoo."},
    )
    optim_quantized: bool = field(
        default=False,
        metadata={"help": "Whether to quantize optimizer (only supported with Distributed Shampoo)."},
    )
    shard_shampoo_across: str = field(
        default="dp",
        metadata={"help": "Whether to shard the optimizer across data devices (dp), model devices (mp) or both (2d)."},
    )
    num_train_epochs: int = field(default=3, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=500, metadata={"help": "Linear warmup over warmup_steps."})
    lr_decay: str = field(
        default=None,
        metadata={
            "help": "Decay to be used in the learning rate scheduler. Can be None (default), linear or exponential."
        },
    )
    num_train_steps: int = field(
        default=None,
        metadata={
            "help": (
                "Total number of training steps to perform. Required only when defining using linear learning rate"
                " decay."
            )
        },
    )
    lr_transition_steps: int = field(
        default=None,
        metadata={
            "help": "Number of transition steps associated with learning rate decay when using exponential decay."
        },
    )
    lr_decay_rate: float = field(
        default=None,
        metadata={"help": "Decay rate associated with learning rate when using exponential decay."},
    )
    lr_staircase: bool = field(
        default=False,
        metadata={"help": "Whether to use staircase or continuous learning rate when using exponential decay."},
    )
    lr_offset: int = field(
        default=0,
        metadata={"help": "Number of steps to offset learning rate and keep it at 0."},
    )
    logging_steps: int = field(default=40, metadata={"help": "Log every X updates steps."})
    eval_steps: int = field(default=400, metadata={"help": "Run an evaluation every X steps."})
    save_steps: int = field(default=4000, metadata={"help": "Save checkpoint every X updates steps."})
    log_model: bool = field(
        default=False,
        metadata={"help": "Log model to wandb at `save_steps` frequency."},
    )
    log_n_samples: int = field(
        default=64,
        metadata={"help": "Number of sample predictions to log during evaluation."},
    )
    log_norm: bool = field(
        default=True,
        metadata={"help": "Log parameters and gradients norm at this frequency."},
    )
    log_histogram_steps: int = field(
        default=False,
        metadata={"help": "Log parameters and gradients histograms at this frequency. Slows down training."},
    )

    seed_model: int = field(
        default=42,
        metadata={"help": "Random seed for the model that will be set at the beginning of training."},
    )

    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "The wandb entity to use (for teams)."},
    )
    wandb_project: str = field(
        default="vit-vqgan",
        metadata={"help": "The name of the wandb project."},
    )
    wandb_job_type: str = field(
        default="train",
        metadata={"help": "The name of the wandb job type."},
    )

    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether or not to upload the trained model to the model hub after training."},
    )
    hub_model_id: str = field(
        default=None,
        metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."},
    )
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})

    assert_TPU_available: bool = field(
        default=False,
        metadata={"help": "Verify that TPU is not in use."},
    )

    use_vmap_trick: bool = field(
        default=True,
        metadata={"help": "Optimization trick that should lead to faster training."},
    )

    mp_devices: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of devices required for model parallelism. The other dimension of available devices is used"
                " for data parallelism."
            )
        },
    )

    dp_devices: int = field(init=False)
    local_dp_devices: int = field(init=False)
    node_groups: int = field(init=False)
    batch_size_per_step: int = field(init=False)
    train_batch_size: int = field(init=False)
    valid_batch_size: int = field(init=False)
    valid_batch_size_per_node: int = field(init=False)
    log_norm_steps: int = field(init=False)

    def __post_init__(self):
        if self.assert_TPU_available:
            assert jax.local_device_count() == 8, "TPUs in use, please check running processes"
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if (
            os.path.exists(self.output_dir)
            and os.listdir(self.output_dir)
            and self.do_train
            and not self.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({self.output_dir}) already exists and is not empty."
                "Use --overwrite_output_dir to overcome."
            )
        assert self.lr_decay in [
            None,
            "linear",
            "exponential",
        ], f"Selected learning rate decay not supported: {self.lr_decay}"
        if self.log_norm is True:
            self.log_norm_steps = self.logging_steps
        elif self.log_norm:
            self.log_norm_steps = self.log_norm
        else:
            self.log_norm_steps = False
        if not self.do_train:
            # eval only
            self.num_train_epochs = 1
        assert self.optim in [
            "distributed_shampoo",
            "adam",
        ], f"Unknown optimizer {self.optim}"
        assert self.graft_type in [
            "rmsprop_normalized",
            "rmsprop",
            "adagrad",
            "adagrad_normalized",
            "sgd",
            "sqrt_n",
        ], f"Selected graft type not supported: {self.graft_type}"
        assert self.shard_shampoo_across in [
            "dp",
            "mp",
            "2d",
        ], f"Shard shampoo across {self.shard_shampoo_across} not supported."
        assert self.mp_devices > 0, f"Number of devices for model parallelism must be > 0"
        assert jax.device_count() % self.mp_devices == 0, (
            f"Number of available devices ({jax.device_count()} must be divisible by number of devices used for model"
            f" parallelism ({self.mp_devices})."
        )
        self.dp_devices = jax.device_count() // self.mp_devices
        # consider batch distributed across nodes (mp > local devices)
        self.node_groups = max(1, self.mp_devices // jax.local_device_count())
        # local dp devices (1 when mp > local devices)
        self.local_dp_devices = jax.local_device_count() * self.node_groups // self.mp_devices
        # batch sizes
        assert self.batch_size_per_node % self.local_dp_devices == 0, (
            f"Batch size per node ({self.batch_size_per_node}) must be divisible by number of local devices"
            f" ({jax.local_device_count()})."
        )
        batch_size_per_node_per_step = self.batch_size_per_node * self.gradient_accumulation_steps
        self.batch_size_per_step = batch_size_per_node_per_step * jax.process_count()
        self.batch_size_per_local_dp_device = self.batch_size_per_node // self.local_dp_devices
        # define batch size for data loader
        self.train_batch_size = batch_size_per_node_per_step * self.node_groups
        self.valid_batch_size = self.batch_size_per_node * jax.process_count()
        self.valid_batch_size_per_node = self.batch_size_per_node * self.node_groups

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    disc_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The discriminator model checkpoint for weights initialization. Don't set if you want to train a model"
                " from scratch."
            )
        },
    )
    disc_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained discriminator config name or path if starting from scratch"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized and trained. Choose one of"
                " `[float32, float16, bfloat16]`."
            )
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    restore_state: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Restore optimizer and training state. Can be True (will retrieve associated wandb artifact) or a"
                " local directory."
            )
        },
    )

    def __post_init__(self):
        if self.restore_state is True:
            assert (
                self.model_name_or_path is not None
            ), "If you want to restore state, you must provide a model name or path."
            assert (
                self.disc_model_name_or_path is not None
            ), "If you want to restore state, you must provide a discriminator model name or path."

    def get_metadata(self):
        if self.model_name_or_path is not None and ":" in self.model_name_or_path:
            if jax.process_index() == 0:
                artifact = wandb.run.use_artifact(self.model_name_or_path)
            else:
                artifact = wandb.Api().artifact(self.model_name_or_path)
            return artifact.metadata
        else:
            return dict()

    def get_opt_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:  # avoid multiple artifact copies
            if self.restore_state is True:
                # wandb artifact
                state_artifact = self.model_name_or_path.replace("/model-", "/state-", 1)
                if jax.process_index() == 0:
                    artifact = wandb.run.use_artifact(state_artifact)
                else:
                    artifact = wandb.Api().artifact(state_artifact)
                artifact_dir = artifact.download(tmp_dir)
                self.restore_state = Path(artifact_dir)

            with (Path(self.restore_state) / "opt_state.msgpack").open("rb") as f:
                opt_state = f.read()
            with (Path(self.restore_state) / "disc_opt_state.msgpack").open("rb") as f:
                disc_opt_state = f.read()
            return opt_state, disc_opt_state


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_folder: str = field(metadata={"help": "Path to the root training directory which contains tfrecords."})
    valid_folder: str = field(metadata={"help": "Path to the root validation directory which contains tfrecords."})
    image_size: Optional[int] = field(default=256, metadata={"help": " The size (resolution) of each image."})
    min_original_image_size: Optional[int] = field(
        default=None,
        metadata={"help": " The minimum size (resolution) of each original image from training set."},
    )
    max_original_aspect_ratio: Optional[float] = field(
        default=None,
        metadata={"help": " The maximum aspect ratio of each original image from training set."},
    )
    seed_dataset: Optional[int] = field(
        default=None,
        metadata={"help": "The seed used to augment the dataset."},
    )
    format: Optional[str] = field(default="rgb", metadata={"help": "The format of the images (rgb or lab)."})


def flat_args(model_args, data_args, training_args):
    args = asdict(model_args)
    args.update(asdict(data_args))
    args.update(asdict(training_args))
    return args


assert jax.local_device_count() == 8


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # check arguments
    if training_args.mp_devices > jax.local_device_count():
        assert (
            data_args.seed_dataset is not None
        ), "Seed dataset must be provided when model is split over multiple hosts"

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # set seed for random transforms and torch dataloaders
    set_seed(training_args.seed_model)

    # Handle the repository creation
    if training_args.push_to_hub:
        if training_args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(training_args.output_dir).absolute().name,
                token=training_args.hub_token,
            )
        else:
            repo_name = training_args.hub_model_id
        repo = Repository(training_args.output_dir, clone_from=repo_name)

    # Initialize datasets and pre-processing transforms
    dataset = Dataset(
        train_batch_size=training_args.train_batch_size,
        valid_batch_size=training_args.valid_batch_size,
        **asdict(data_args),
    )

    # Info on local devices
    logger.info(f"Local TPUs/GPUs: {jax.local_device_count()}")
    logger.info(f"Global TPUs/GPUs: {jax.device_count()}")

    # Set up wandb run
    if jax.process_index() == 0:
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            job_type=training_args.wandb_job_type,
            config=flat_args(model_args, data_args, training_args),
        )

    # Set up model configs
    if model_args.config_name:
        config = ViTVQConfig.from_pretrained(model_args.config_name)
    else:
        config = None

    if model_args.disc_config_name:
        disc_config = StyleGANDiscriminatorConfig.from_pretrained(model_args.disc_config_name)
    else:
        disc_config = None

    # Load or create new models
    if model_args.model_name_or_path:
        model, params = ViTVQModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            seed=training_args.seed_model,
            dtype=getattr(jnp, model_args.dtype),
            _do_init=False,  # we overwrite them with loaded checkpoint
        )
    else:
        model = ViTVQModel(
            config,
            seed=training_args.seed_model,
            dtype=getattr(jnp, model_args.dtype),
            _do_init=False,
        )
        params = None

    if model_args.disc_model_name_or_path:
        disc_model, disc_params = StyleGANDiscriminator.from_pretrained(
            model_args.disc_model_name_or_path,
            config=disc_config,
            seed=training_args.seed_model,
            dtype=getattr(jnp, model_args.dtype),
            _do_init=False,  # we overwrite them with loaded checkpoint
        )
    else:
        disc_model = StyleGANDiscriminator(
            disc_config,
            seed=training_args.seed_model,
            dtype=getattr(jnp, model_args.dtype),
            _do_init=False,
        )
        disc_params = None

    # overwrite certain config parameters
    model.config.gradient_checkpointing = training_args.gradient_checkpointing
    disc_model.config.gradient_checkpointing = training_args.gradient_checkpointing

    # get model metadata
    model_metadata = model_args.get_metadata()

    # get PartitionSpec and shape for model params
    params_spec = None
    disc_params_spec = None
    lpips_spec = None
    if training_args.mp_devices > 1:
        raise NotImplementedError("Model Parallelism not implemented yet")
    params_shape = freeze(model.params_shape_tree)
    disc_params_shape = freeze(disc_model.params_shape_tree)

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed_model)
    rng, dropout_rng = jax.random.split(rng)

    # Store some constant
    num_epochs = training_args.num_train_epochs
    num_params = model.num_params(params_shape)

    # log some info
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Batch size per node = {training_args.batch_size_per_node}")
    logger.info(f"  Number of devices = {jax.device_count()}")
    logger.info(f"  Gradient accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Batch size per update = {training_args.batch_size_per_step}")
    logger.info(f"  Model parameters = {num_params:,}")

    if jax.process_index() == 0:
        # set default x-axis as 'train/step'
        wandb.define_metric("*", step_metric="train/step")

        # add interesting config parameters
        wandb.config.update(
            {
                "batch_size_per_step": training_args.batch_size_per_step,
                "num_params": num_params,
                "model_config": model.config.to_dict(),
                "num_devices": jax.device_count(),
                "versions": {
                    "jax": jax.__version__,
                    "jaxlib": jaxlib.__version__,
                    "flax": flax.__version__,
                    "optax": optax.__version__,
                    "transformers": transformers.__version__,
                    "wandb": wandb.__version__,
                },
            }
        )

    # Create learning rate schedule
    def create_learning_rate_fn(learning_rate) -> Callable[[int], jnp.array]:
        """Create the learning rate function."""
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=learning_rate,
            transition_steps=training_args.warmup_steps + 1,  # ensure not 0
        )
        last_boundary = training_args.warmup_steps
        # offset step when resuming
        if training_args.lr_offset:
            warmup_fn = optax.join_schedules(
                schedules=[optax.constant_schedule(0.0), warmup_fn],
                boundaries=[training_args.lr_offset],
            )
            last_boundary += training_args.lr_offset
        if training_args.lr_decay is None:
            return warmup_fn
        elif training_args.lr_decay == "linear":
            assert (
                training_args.num_train_steps is not None
            ), "linear decay requires specifying explicitly num_train_steps"
            assert training_args.num_train_steps > training_args.warmup_steps, (
                f"linear decay requires number of training steps > warmup steps, got {training_args.num_train_steps} <"
                f" {training_args.warmup_steps}"
            )
            decay_fn = optax.linear_schedule(
                init_value=learning_rate,
                end_value=0,
                transition_steps=training_args.num_train_steps - training_args.warmup_steps,
            )
        elif training_args.lr_decay == "exponential":
            decay_fn = optax.exponential_decay(
                init_value=learning_rate,
                transition_steps=training_args.lr_transition_steps,
                decay_rate=training_args.lr_decay_rate,
                staircase=training_args.lr_staircase,
            )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, decay_fn],
            boundaries=[last_boundary],
        )
        return schedule_fn

    learning_rate_fn = create_learning_rate_fn(training_args.learning_rate)
    disc_learning_rate_fn = create_learning_rate_fn(training_args.disc_learning_rate)

    # create optimizer
    if training_args.optim == "distributed_shampoo":
        # parameters from https://github.com/tensorflow/lingvo/blob/03ee9d7cd50764b0424c7c863733c91fc0b053ec/lingvo/jax/optimizers.py#L729
        graft_type = {
            "sgd": GraftingType.SGD,
            "adagrad": GraftingType.ADAGRAD,
            "rmsprop": GraftingType.RMSPROP,
            "rmsprop_normalized": GraftingType.RMSPROP_NORMALIZED,
            "sqrt_n": GraftingType.SQRT_N,
            "adagrad_normalized": GraftingType.ADAGRAD_NORMALIZED,
        }[training_args.graft_type]
        statistics_partition_spec = (
            PartitionSpec(None, training_args.shard_shampoo_across, None)
            if training_args.shard_shampoo_across != "2d"
            else PartitionSpec(None, "dp", "mp")
        )
        _opt = partial(
            distributed_shampoo,
            block_size=training_args.block_size,
            beta1=training_args.beta1,
            beta2=training_args.beta2,
            diagonal_epsilon=1e-10,
            matrix_epsilon=1e-6,
            weight_decay=training_args.weight_decay,
            start_preconditioning_step=max(training_args.preconditioning_compute_steps + 1, 101),
            preconditioning_compute_steps=training_args.preconditioning_compute_steps,
            statistics_compute_steps=1,
            best_effort_shape_interpretation=True,
            graft_type=graft_type,
            nesterov=training_args.nesterov,
            exponent_override=0,
            statistics_partition_spec=statistics_partition_spec,
            preconditioner_partition_spec=PartitionSpec(training_args.shard_shampoo_across, None, None)
            if training_args.shard_shampoo_across != "2d"
            else PartitionSpec(
                "mp" if training_args.mp_devices > training_args.dp_devices else "dp",
                None,
                None,
            ),
            num_devices_for_pjit=training_args.dp_devices,
            shard_optimizer_states=True,
            inverse_failure_threshold=0.1,
            moving_average_for_momentum=True,
            skip_preconditioning_dim_size_gt=training_args.skip_preconditioning_dim_size_gt,
            clip_by_scaled_gradient_norm=None,
            precision=jax.lax.Precision.HIGHEST,
            best_effort_memory_usage_reduction=training_args.optim_quantized,
        )
        # get the real optimizer and helper functions
        opt = _opt(learning_rate_fn)
        disc_opt = _opt(disc_learning_rate_fn)
        update_fn = opt.update
        disc_update_fn = disc_opt.update
        optimizer = opt.init(params_shape)
        disc_optimizer = disc_opt.init(disc_params_shape)
        opt_fn = NamedTuple("opt_fn", pspec_fn=Any, shape_and_dtype_fn=Any)(
            optimizer.pspec_fn, optimizer.shape_and_dtype_fn
        )
        disc_opt_fn = NamedTuple("opt_fn", pspec_fn=Any, shape_and_dtype_fn=Any)(
            disc_optimizer.pspec_fn, disc_optimizer.shape_and_dtype_fn
        )
        optimizer = optax.GradientTransformation(optimizer.init_fn, update_fn)
        disc_optimizer = optax.GradientTransformation(disc_optimizer.init_fn, disc_update_fn)

    elif training_args.optim == "adam":
        _opt = partial(
            optax.adamw,
            b1=training_args.beta1,
            b2=training_args.beta2,
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
        )
        optimizer = _opt(learning_rate=learning_rate_fn)
        disc_optimizer = _opt(learning_rate=disc_learning_rate_fn)

    # get PartitionSpec and shape of optimizer state
    def get_opt_state_spec_and_shape():
        # get opt_state shape without actual init
        opt_state_shape = jax.eval_shape(optimizer.init, params_shape)
        disc_opt_state_shape = jax.eval_shape(disc_optimizer.init, disc_params_shape)
        # get PartitionSpec
        if training_args.optim == "adam":

            def _opt_state_spec_per_leaf(x, spec):
                if isinstance(x, FrozenDict):
                    # variables with same structure as params
                    return spec
                else:
                    # other variables such as count
                    return None

            def pspec_fn(spec, shape):
                return (
                    None
                    if spec is None
                    else jax.tree_util.tree_map(
                        partial(_opt_state_spec_per_leaf, spec=spec),
                        shape,
                        # return None spec for empty elements
                        is_leaf=lambda x: isinstance(x, (FrozenDict, optax.EmptyState)),
                    )
                )

            opt_state_spec = pspec_fn(params_spec, opt_state_shape)
            disc_opt_state_spec = pspec_fn(disc_params_spec, disc_opt_state_shape)

        elif training_args.optim == "distributed_shampoo":
            params_spec = jax.tree_util.tree_map(lambda x: PartitionSpec(None), params_shape)
            opt_state_spec = opt_fn.pspec_fn(
                params_shape,
                params_spec,
                statistics_partition_spec,
            )
            disc_params_spec = jax.tree_util.tree_map(lambda x: PartitionSpec(None), disc_params_shape)
            disc_opt_state_spec = disc_opt_fn.pspec_fn(
                disc_params_shape,
                disc_params_spec,
                statistics_partition_spec,
            )
        else:
            raise NotImplementedError
        return opt_state_spec, opt_state_shape, disc_opt_state_spec, disc_opt_state_shape

    opt_state_spec, opt_state_shape, disc_opt_state_spec, disc_opt_state_shape = get_opt_state_spec_and_shape()

    # create a mesh
    mesh_shape = (training_args.dp_devices, training_args.mp_devices)
    devices = np.asarray(jax.devices()).reshape(*mesh_shape)
    mesh = maps.Mesh(devices, ("dp", "mp"))
    logger.info(f"  Mesh shape: {mesh_shape}")

    class TrainState(struct.PyTreeNode):
        step: int
        params: core.FrozenDict[str, Any]
        disc_params: core.FrozenDict[str, Any]
        lpips_params: core.FrozenDict[str, Any]
        opt_state: optax.OptState
        disc_opt_state: optax.OptState
        dropout_rng: jnp.ndarray = None
        epoch: int = 0
        train_time: float = 0.0  # total time the model trained
        train_samples: int = 0  # number of samples seen

        def apply_gradients(self, *, grads, disc_grads, **kwargs):
            updates, new_opt_state = optimizer.update(grads, self.opt_state, self.params)
            new_params = optax.apply_updates(self.params, updates)
            disc_updates, new_disc_opt_state = disc_optimizer.update(disc_grads, self.disc_opt_state, self.disc_params)
            new_disc_params = optax.apply_updates(self.disc_params, disc_updates)
            return self.replace(
                step=self.step + 1,
                params=new_params,
                opt_state=new_opt_state,
                disc_params=new_disc_params,
                disc_opt_state=new_disc_opt_state,
                **kwargs,
            )

        @classmethod
        def create(cls, *, params, disc_params, lpips_params, **kwargs):
            opt_state = optimizer.init(params)
            disc_opt_state = disc_optimizer.init(disc_params)
            return cls(
                step=0,
                params=params,
                disc_params=disc_params,
                lpips_params=lpips_params,
                opt_state=opt_state,
                disc_opt_state=disc_opt_state,
                **kwargs,
            )

    # define state spec
    state_spec = TrainState(
        params=params_spec,
        disc_params=disc_params_spec,
        opt_state=opt_state_spec,
        disc_opt_state=disc_opt_state_spec,
        dropout_rng=None,
        step=None,
        epoch=None,
        train_time=None,
        train_samples=None,
        lpips_params=lpips_spec,
    )

    # define lpips
    lpips_fn = LPIPS()

    def init_lpips(rng, image_size):
        x = jax.random.normal(rng, shape=(1, image_size, image_size, 3))
        return lpips_fn.init(rng, x, x)

    # init params if not available yet
    def maybe_init_params(params, m):
        if params is not None:
            # model params are correctly loaded
            return params
        else:
            # params have not been initialized yet
            return m.init_weights(m.key, m.input_shape)

    with mesh:
        logger.info("  Creating state")

        # restore metadata
        attr_state = {}
        keys = ["train_time", "train_samples"]
        if model_args.restore_state:
            keys += ["step", "epoch"]
        attr_state = {k: v for k, v in model_metadata.items() if k in keys}

        if not model_args.restore_state:

            def init_state(params, disc_params):
                lpips_params = init_lpips(rng, data_args.image_size)
                return TrainState.create(
                    params=maybe_init_params(params, model),
                    disc_params=maybe_init_params(disc_params, disc_model),
                    lpips_params=lpips_params,
                    dropout_rng=dropout_rng,
                    **attr_state,
                )

            state = pjit(
                init_state,
                in_axis_resources=(
                    params_spec if model_args.model_name_or_path else None,
                    disc_params_spec if model_args.disc_model_name_or_path else None,
                ),
                out_axis_resources=state_spec,
                donate_argnums=(0, 1),
            )(params, disc_params)

        else:
            # load opt_state
            opt_state, disc_opt_state = model_args.get_opt_state()
            opt_state = from_bytes(opt_state_shape, opt_state)
            disc_opt_state = from_bytes(disc_opt_state_shape, disc_opt_state)

            def restore_state(params, disc_params, opt_state, disc_opt_state):
                lpips_params = init_lpips(rng, data_args.image_size)
                return TrainState(
                    params=params,
                    disc_params=disc_params,
                    opt_state=opt_state,
                    disc_opt_state=disc_opt_state,
                    lpips_params=lpips_params,
                    dropout_rng=dropout_rng,
                    **attr_state,
                )

            state = pjit(
                restore_state,
                in_axis_resources=(
                    params_spec,
                    disc_params_spec,
                    opt_state_spec,
                    disc_opt_state_spec,
                ),
                out_axis_resources=state_spec,
                donate_argnums=(0, 1, 2, 3),
            )(params, disc_params, opt_state, disc_opt_state)

            # remove opt_state from CPU
            del opt_state, disc_opt_state, disc_params

    # free CPU memory
    del params, opt_state_spec, opt_state_shape, disc_opt_state_spec

    # define batch spec
    batch_spec = PartitionSpec("dp")
    grad_batch_spec = PartitionSpec(None, "dp")

    # "vmap trick" avoids a crash when mp_devices > 1 (not sure why it happens)
    # lead to better perf: see https://wandb.ai/dalle-mini/dalle-mini/reports/JAX-pmap-vs-pjit--VmlldzoxNDg1ODA2
    if training_args.use_vmap_trick:
        grad_params_spec = jax.tree_util.tree_map(
            lambda x: PartitionSpec(*("dp",) + (x if x is not None else (None,))),
            params_spec,
        )
        grad_disc_params_spec = jax.tree_util.tree_map(
            lambda x: PartitionSpec(*("dp",) + (x if x is not None else (None,))),
            disc_params_spec,
        )

    # Define loss
    def compute_loss(params, disc_params, minibatch, dropout_rng, model_fn, disc_model_fn, train):
        predicted_images, (q_latent_loss, e_latent_loss) = model_fn(
            minibatch, params=params, dropout_rng=dropout_rng, train=train
        )
        disc_fake_scores = disc_model_fn(predicted_images, params=disc_params, dropout_rng=dropout_rng, train=train)
        loss_disc = jnp.mean(nn.softplus(-disc_fake_scores))
        # TODO: replace l1 with logit laplace (only if necessary)
        loss_l1 = jnp.mean(jnp.abs(predicted_images - minibatch))
        loss_l2 = jnp.mean((predicted_images - minibatch) ** 2)
        loss_lpips = jnp.mean(lpips_fn.apply(state.lpips_params, minibatch, predicted_images))

        loss = (
            model.config.cost_l1 * loss_l1
            + model.config.cost_l2 * loss_l2
            + model.config.cost_q_latent * q_latent_loss
            + model.config.cost_e_latent * e_latent_loss
            + model.config.cost_lpips * loss_lpips
            + model.config.cost_stylegan * loss_disc
        )
        loss_details = {
            "loss_l1": model.config.cost_l1 * loss_l1,
            "loss_l2": model.config.cost_l2 * loss_l2,
            "loss_q_latent": model.config.cost_q_latent * q_latent_loss,
            "loss_e_latent": model.config.cost_e_latent * e_latent_loss,
            "loss_lpips": model.config.cost_lpips * loss_lpips,
            "loss_stylegan": model.config.cost_stylegan * loss_disc,
        }
        return loss, (loss_details, predicted_images)

    def compute_stylegan_loss(disc_params, minibatch, predicted_images, dropout_rng, disc_model_fn, train):
        disc_fake_scores = disc_model_fn(predicted_images, params=disc_params, dropout_rng=dropout_rng, train=train)
        disc_real_scores = disc_model_fn(minibatch, params=disc_params, dropout_rng=dropout_rng, train=train)
        disc_loss_stylegan = jnp.mean(nn.softplus(disc_fake_scores) + nn.softplus(-disc_real_scores))

        # gradient penalty r1: https://github.com/NVlabs/stylegan2/blob/bf0fe0baba9fc7039eae0cac575c1778be1ce3e3/training/loss.py#L63-L67
        r1_grads = jax.grad(
            lambda p: jnp.mean(disc_model_fn(minibatch, params=p, dropout_rng=dropout_rng, train=train))
        )(disc_params)
        # get the squares of gradients
        r1_grads = flatten_dict(unfreeze(r1_grads))
        r1_grads = jax.tree_util.tree_map(lambda x: jnp.sum(x**2), r1_grads)
        r1_grads = sum(list(r1_grads.values()))

        disc_loss = disc_loss_stylegan + model.config.cost_gradient_penalty * r1_grads
        disc_loss_details = {
            "disc_loss": disc_loss,
            "disc_loss_stylegan": disc_loss_stylegan,
            "disc_loss_gradient_penalty": model.config.cost_gradient_penalty * r1_grads,
        }
        return disc_loss, disc_loss_details

    # Define gradient update step fn
    def train_step(state, batch, train_time):
        # get a minibatch (one gradient accumulation slice)
        def get_minibatch(batch, grad_idx):
            return jax.tree_util.tree_map(
                lambda x: jax.lax.dynamic_index_in_dim(x, grad_idx, keepdims=False),
                batch,
            )

        train_compute_loss = partial(compute_loss, train=True)
        train_compute_stylegan_loss = partial(compute_stylegan_loss, train=True)
        grad_fn = jax.value_and_grad(train_compute_loss, has_aux=True)
        grad_stylegan_fn = jax.value_and_grad(train_compute_stylegan_loss, has_aux=True)

        def loss_and_grad(grad_idx, dropout_rng):
            # minibatch at grad_idx for gradient accumulation (None otherwise)
            minibatch = get_minibatch(batch, grad_idx) if grad_idx is not None else batch
            # ensure it is sharded properly
            minibatch = with_sharding_constraint(minibatch, batch_spec)
            # only 1 single rng per grad step, let us handle larger batch size (not sure why)
            dropout_rng, _ = jax.random.split(dropout_rng)

            if training_args.use_vmap_trick:
                # "vmap trick", calculate loss and grads independently per dp_device
                (loss, (loss_details, predicted_images)), grads = jax.vmap(
                    grad_fn, in_axes=(None, None, 0, None, None, None), out_axes=(0, 0)
                )(state.params, state.disc_params, minibatch, dropout_rng, model, disc_model)
                # ensure they are sharded correctly
                loss = with_sharding_constraint(loss, batch_spec)
                loss_details = with_sharding_constraint(loss_details, batch_spec)
                predicted_images = with_sharding_constraint(predicted_images, batch_spec)
                grads = with_sharding_constraint(grads, grad_params_spec)

                # discriminator grads
                (disc_loss, disc_loss_details), disc_grads = jax.vmap(
                    grad_stylegan_fn, in_axes=(None, 0, 0, None, None), out_axes=(0, 0)
                )(state.disc_params, minibatch, predicted_images, dropout_rng, disc_model)
                # ensure they are sharded correctly
                disc_loss = with_sharding_constraint(disc_loss, batch_spec)
                disc_loss_details = with_sharding_constraint(disc_loss_details, batch_spec)
                disc_grads = with_sharding_constraint(disc_grads, grad_disc_params_spec)

                # average across all devices
                # Note: we could average per device only after gradient accumulation, right before params update
                loss, grads, loss_details, disc_loss, disc_grads, disc_loss_details = jax.tree_util.tree_map(
                    lambda x: jnp.mean(x, axis=0),
                    (loss, grads, loss_details, disc_loss, disc_grads, disc_loss_details),
                )
            else:
                # "vmap trick" may not work in multi-hosts or require too much hbm
                (loss, (loss_details, predicted_images)), grads = grad_fn(
                    state.params, state.disc_params, minibatch, dropout_rng, model, disc_model
                )
                (disc_loss, disc_loss_details), disc_grads = grad_stylegan_fn(
                    state.disc_params, minibatch, predicted_images, dropout_rng, disc_model
                )
            # ensure grads are sharded
            grads = with_sharding_constraint(grads, params_spec)
            disc_grads = with_sharding_constraint(disc_grads, disc_params_spec)
            # return loss, grads and metrics
            loss_details = {**loss_details, **disc_loss_details}
            return loss, grads, disc_grads, dropout_rng, loss_details

        if training_args.gradient_accumulation_steps == 1:
            loss, grads, disc_grads, dropout_rng, loss_details = loss_and_grad(None, state.dropout_rng)
        else:
            # create initial state for cumul_minibatch_step loop
            init_minibatch_step = (
                0.0,
                with_sharding_constraint(jax.tree_util.tree_map(jnp.zeros_like, state.params), params_spec),
                with_sharding_constraint(jax.tree_util.tree_map(jnp.zeros_like, state.disc_params), disc_params_spec),
                state.dropout_rng,
                {
                    "loss_l1": 0.0,
                    "loss_l2": 0.0,
                    "loss_q_latent": 0.0,
                    "loss_e_latent": 0.0,
                    "loss_lpips": 0.0,
                    "disc_loss": 0.0,
                    "disc_loss_stylegan": 0.0,
                    "disc_loss_gradient_penalty": 0.0,
                },
            )

            # accumulate gradients
            def cumul_minibatch_step(grad_idx, cumul_loss_grad_dropout):
                (
                    cumul_loss,
                    cumul_grads,
                    cumul_disc_grads,
                    dropout_rng,
                    cumul_loss_details,
                ) = cumul_loss_grad_dropout
                loss, grads, disc_grads, dropout_rng, loss_details = loss_and_grad(grad_idx, dropout_rng)
                cumul_loss, cumul_grads, cumul_disc_grads, cumul_loss_details = jax.tree_util.tree_map(
                    jnp.add,
                    (cumul_loss, cumul_grads, cumul_disc_grads, cumul_loss_details),
                    (loss, grads, disc_grads, loss_details),
                )
                cumul_grads = with_sharding_constraint(cumul_grads, params_spec)
                cumul_disc_grads = with_sharding_constraint(cumul_disc_grads, disc_params_spec)
                return cumul_loss, cumul_grads, cumul_disc_grads, dropout_rng, cumul_loss_details

            # loop over gradients
            loss, grads, disc_grads, dropout_rng, loss_details = jax.lax.fori_loop(
                0,
                training_args.gradient_accumulation_steps,
                cumul_minibatch_step,
                init_minibatch_step,
            )
            grads = with_sharding_constraint(grads, params_spec)
            disc_grads = with_sharding_constraint(disc_grads, disc_params_spec)
            # sum -> mean
            loss, grads, disc_grads, loss_details = jax.tree_util.tree_map(
                lambda x: x / training_args.gradient_accumulation_steps,
                (loss, grads, disc_grads, loss_details),
            )

        grads = with_sharding_constraint(grads, params_spec)
        disc_grads = with_sharding_constraint(disc_grads, disc_params_spec)

        # update state
        state = state.apply_gradients(
            grads=grads,
            disc_grads=disc_grads,
            dropout_rng=dropout_rng,
            train_time=train_time,
            train_samples=state.train_samples + training_args.batch_size_per_step,
        )

        metrics = {
            "loss": loss,
            "learning_rate": learning_rate_fn(state.step),
            "disc_learning_rate": disc_learning_rate_fn(state.step),
            **loss_details,
        }

        # extract norms and histograms

        def maybe_fn(fn, val, zeros, freq):
            """Call fn only if it is a logging step"""
            return jax.lax.cond(
                state.step % freq == 0,
                fn,
                lambda _: zeros,
                val,
            )

        if training_args.log_norm_steps:
            zeros_norm = jax.tree_util.tree_map(lambda _: jnp.float32(0), state.params)

            def norm(val):
                return jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), val)

            gradients_norm = maybe_fn(norm, grads, zeros_norm, training_args.log_norm_steps)
            params_norm = maybe_fn(norm, state.params, zeros_norm, training_args.log_norm_steps)

            metrics.update(
                {
                    "gradients_norm": gradients_norm,
                    "params_norm": params_norm,
                }
            )

        if training_args.log_histogram_steps:
            zeros_hist = jax.tree_util.tree_map(lambda _: jnp.histogram(jnp.zeros(1), density=True), state.params)

            def histogram(val):
                return jax.tree_util.tree_map(lambda x: jnp.histogram(x, density=True), val)

            gradients_hist = maybe_fn(histogram, grads, zeros_hist, training_args.log_histogram_steps)
            params_hist = maybe_fn(histogram, state.params, zeros_hist, training_args.log_histogram_steps)

            metrics.update(
                {
                    "params_hist": params_hist,
                    "gradients_hist": gradients_hist,
                }
            )

        return state, metrics

    # Ensure eval_fn is in float32 to avoid numerical issues
    eval_model = (
        model
        if model_args.dtype == "float32"
        else ViTVQModel(
            model.config,
            seed=training_args.seed_model,
            dtype=jnp.float32,
            _do_init=False,
        )
    )
    eval_disc_model = (
        disc_model
        if model_args.dtype == "float32"
        else StyleGANDiscriminator(
            disc_model.config,
            seed=training_args.seed_model,
            dtype=jnp.float32,
            _do_init=False,
        )
    )

    # Evaluation step
    def eval_step(state, batch):
        def compute_eval_loss(batch):
            loss, (loss_details, predicted_images) = compute_loss(
                state.params,
                state.disc_params,
                batch,
                dropout_rng=None,
                model_fn=eval_model,
                disc_model_fn=eval_disc_model,
                train=False,
            )
            _, disc_loss_details = compute_stylegan_loss(
                state.disc_params, batch, predicted_images, None, eval_disc_model, train=False
            )
            loss_details = {**loss_details, **disc_loss_details}
            return {
                "loss": loss,
                **loss_details,
            }

        if training_args.use_vmap_trick:
            metrics = jax.vmap(compute_eval_loss)(batch)
            # ensure they are sharded correctly
            metrics = with_sharding_constraint(metrics, batch_spec)
            # average across all devices
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        else:
            metrics = compute_eval_loss(batch)

        return metrics

    # Inference step (to log samples)
    def inference_step(state, batch):
        def get_predictions(minibatch):
            predicted_images, _ = eval_model(minibatch, params=state.params, dropout_rng=None, train=False)
            return predicted_images

        if training_args.use_vmap_trick:
            preds = jax.vmap(get_predictions)(batch)
        else:
            preds = get_predictions(batch)

        return preds

    # Create parallel version of the train and eval step
    p_train_step = pjit(
        train_step,
        in_axis_resources=(
            state_spec,
            grad_batch_spec if training_args.gradient_accumulation_steps > 1 else batch_spec,
            None,
        ),
        out_axis_resources=(state_spec, None),
        donate_argnums=(0,),
    )
    p_eval_step = pjit(
        eval_step,
        in_axis_resources=(state_spec, batch_spec),
        out_axis_resources=None,
    )
    p_inference_step = pjit(inference_step, in_axis_resources=(state_spec, batch_spec), out_axis_resources=batch_spec)

    # define metrics logger
    class MetricsLogger:
        def __init__(self, step):
            # keep state to use any key as a custom x-axis
            self.state_dict = {}
            # estimate speed
            self.step = step
            self.time = time.perf_counter()
            self.offset_time = 0.0

        def update_state_metrics(self, state):
            """Update internal state metrics (logged at each call to be used as x-axis)"""
            self.state_dict = {
                f'train/{k.split("_")[-1]}': state[k] for k in ["step", "epoch", "train_time", "train_samples"]
            }
            # timing metrics
            new_step = int(state["step"])
            new_time = time.perf_counter()
            if new_step > self.step:
                # remove time for eval & save
                delta_time = new_time - self.time - self.offset_time
                self.offset_time = 0
                time_per_step = delta_time / (new_step - self.step)
                self.step = new_step
                self.time = new_time
                self.log_time("train_per_step", time_per_step, offset=False)
                self.log_time("train_per_log", delta_time, offset=False)

        def log_time(self, key, duration, offset=True):
            if jax.process_index() == 0:
                wandb.log({f"time/{key}": duration, **self.state_dict})
            if offset:
                self.offset_time += duration

        def log(self, metrics, prefix=None):
            if jax.process_index() == 0:
                log_metrics = {}
                for k, v in metrics.items():
                    if "_norm" in k:
                        if self.step % training_args.log_norm_steps == 0:
                            log_metrics[f"{k}/"] = unfreeze(v)
                    elif "_hist" in k:
                        if self.step % training_args.log_histogram_steps == 0:
                            v = jax.tree_util.tree_map(lambda x: jax.device_get(x), unfreeze(v))
                            v = jax.tree_util.tree_map(
                                lambda x: wandb.Histogram(np_histogram=x),
                                v,
                                is_leaf=lambda x: isinstance(x, tuple),
                            )
                            log_metrics[f"{k}/"] = v
                    else:
                        if prefix is not None:
                            k = f"{prefix}/{k}"
                        log_metrics[k] = v
                wandb.log({**log_metrics, **self.state_dict})

    # keep local copy of state to avoid communication
    local_state = {
        k: jax.device_get(getattr(state, k)).item() for k in ["step", "epoch", "train_time", "train_samples"]
    }

    # init variables
    start_time = time.perf_counter() - local_state["train_time"]
    train_metrics = None
    evaluation_ran = False
    save_model_ran = False
    metrics_logger = MetricsLogger(local_state["step"])
    epochs = tqdm(
        range(local_state["epoch"], num_epochs),
        desc=f"Epoch ... (1/{num_epochs})",
        position=0,
        disable=jax.process_index() > 0,
    )

    def run_evaluation():
        # ======================== Evaluating ==============================
        if training_args.do_eval:
            start_eval_time = time.perf_counter()
            metrics = []
            # number of steps required to have enough samples to log
            n_samples_step = training_args.log_n_samples / training_args.valid_batch_size_per_node
            images, predictions = [], []
            for i, batch in tqdm(
                enumerate(dataset.valid.as_numpy_iterator()),
                desc="Evaluating...",
                position=2,
                leave=False,
                disable=jax.process_index() > 0,
            ):
                # need to keep only items relevant to the node
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape(
                        (
                            jax.process_count() // training_args.node_groups,
                            training_args.valid_batch_size_per_node,
                        )
                        + x.shape[1:]
                    ),
                    batch,
                )
                batch = jax.tree_util.tree_map(lambda x: x[jax.process_index() // training_args.node_groups], batch)

                # add dp dimension when using "vmap trick"
                if training_args.use_vmap_trick:
                    bs_shape = (
                        training_args.local_dp_devices,
                        training_args.batch_size_per_local_dp_device,
                    )
                    batch = jax.tree_util.tree_map(lambda x: x.reshape(bs_shape + x.shape[1:]), batch)

                # accumulate losses async
                metrics.append(p_eval_step(state, batch))

                # add sample predictions
                if i < n_samples_step:
                    predictions.append(p_inference_step(state, batch))
                    images.append(batch)

            # get the mean of the metrics
            metrics = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *metrics)
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)

            # log metrics
            metrics_logger.log(metrics, prefix="valid")

            # log images
            images = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *images)
            predictions = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *predictions)
            # flatten images and preds
            images, predictions = jax.tree_util.tree_map(
                lambda x: x.reshape((-1,) + x.shape[-3:]), (images, predictions)
            )
            images = images[: training_args.log_n_samples]
            predictions = predictions[: training_args.log_n_samples]
            # convert to images
            images = logits_to_image(images, dataset.format)
            predictions = logits_to_image(predictions, dataset.format)
            # log in wandb
            if jax.process_index() == 0:
                images = [wandb.Image(img) for img in images]
                predictions = [wandb.Image(img) for img in predictions]
                wandb.log({"samples/ground_truth": images, "samples/predictions": predictions})

            # Print metrics and update progress bar
            desc = f"Epoch... ({epoch + 1}/{num_epochs} | Valid Loss: {metrics['loss']})"
            epochs.write(desc)
            epochs.desc = desc

            # log time
            metrics_logger.log_time("valid", time.perf_counter() - start_eval_time)

            return metrics

    def run_save_model(state, eval_metrics=None):
        if jax.process_index() == 0:
            start_save_time = time.perf_counter()
            output_dir = training_args.output_dir
            disc_output_dir = Path(output_dir) / "disc"

            # save model
            params = jax.device_get(state.params)
            model.save_pretrained(output_dir, params=params)
            disc_params = jax.device_get(state.disc_params)
            disc_model.save_pretrained(disc_output_dir, params=disc_params)

            # save state
            opt_state = jax.device_get(state.opt_state)
            with (Path(output_dir) / "opt_state.msgpack").open("wb") as f:
                f.write(to_bytes(opt_state))
            disc_opt_state = jax.device_get(state.disc_opt_state)
            with (Path(output_dir) / "disc_opt_state.msgpack").open("wb") as f:
                f.write(to_bytes(disc_opt_state))

            # save to HF hub
            if training_args.push_to_hub:
                repo.push_to_hub(
                    commit_message=f"Saving weights and logs of epoch {epoch}",
                    blocking=False,
                )

            # save to W&B
            if training_args.log_model:
                # save some space
                c = wandb.wandb_sdk.wandb_artifacts.get_artifacts_cache()
                c.cleanup(wandb.util.from_human_size("10GB"))

                metadata = {
                    k: jax.device_get(getattr(state, k)).item()
                    for k in ["step", "epoch", "train_time", "train_samples"]
                }
                metadata["num_params"] = num_params
                if eval_metrics is not None:
                    metadata["eval"] = eval_metrics

                # create model artifact
                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}",
                    type="ViT-VQGAN",
                    metadata=metadata,
                )
                for filename in [
                    "config.json",
                    "flax_model.msgpack",
                ]:
                    artifact.add_file(f"{Path(training_args.output_dir) / filename}")
                wandb.run.log_artifact(artifact)

                # create state artifact
                artifact_state = wandb.Artifact(
                    name=f"state-{wandb.run.id}",
                    type="state",
                    metadata=metadata,
                )
                artifact_state.add_file(f"{Path(training_args.output_dir) / 'opt_state.msgpack'}")
                artifact_state.add_file(f"{Path(training_args.output_dir) / 'disc_opt_state.msgpack'}")
                wandb.run.log_artifact(artifact_state)
            metrics_logger.log_time("save_model", time.perf_counter() - start_save_time)

    logger.info("  Ready to start training")
    with mesh:
        for epoch in epochs:
            state = state.replace(epoch=epoch)
            local_state["epoch"] = epoch
            # ======================== Training ================================
            metrics_logger.update_state_metrics(local_state)
            metrics_logger.log({})

            # train
            if training_args.do_train:
                for batch in tqdm(
                    dataset.train.as_numpy_iterator(),
                    desc="Training...",
                    position=1,
                    leave=False,
                    disable=jax.process_index() > 0,
                ):
                    # calculate delta time (we have a lag of one step but it's ok)
                    train_time = time.perf_counter() - start_time

                    # reset control variables
                    evaluation_ran = False
                    save_model_ran = False

                    # set correct shape to batch
                    bs_shape = (
                        (training_args.batch_size_per_node * training_args.node_groups,)
                        if not training_args.use_vmap_trick
                        else (
                            training_args.local_dp_devices,
                            training_args.batch_size_per_local_dp_device,
                        )
                    )
                    if training_args.gradient_accumulation_steps > 1:
                        # reshape data into (gradient_accumulation_steps, batch_per_node, ...)
                        # to avoid any data redistribution when sharding
                        bs_shape = (training_args.gradient_accumulation_steps,) + bs_shape

                    # reshape batch
                    batch = jax.tree_util.tree_map(
                        lambda x: x.reshape(bs_shape + x.shape[1:]),
                        batch,
                    )

                    # train step
                    state, train_metrics = p_train_step(state, batch, train_time)
                    local_state["step"] += 1
                    local_state["train_time"] = train_time
                    local_state["train_samples"] += training_args.batch_size_per_step

                    if local_state["step"] % training_args.logging_steps == 0 and jax.process_index() == 0:
                        metrics_logger.update_state_metrics(local_state)
                        metrics_logger.log(train_metrics, prefix="train")

                    eval_metrics = None
                    if local_state["step"] % training_args.eval_steps == 0:
                        eval_metrics = run_evaluation()
                        evaluation_ran = True

                    if local_state["step"] % training_args.save_steps == 0:
                        run_save_model(state, eval_metrics)
                        save_model_ran = True

                # log final train metrics
                if train_metrics is not None:
                    metrics_logger.update_state_metrics(local_state)
                    metrics_logger.log(train_metrics, prefix="train")

                    epochs.write(
                        f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {train_metrics['loss']}, Learning Rate:"
                        f" {train_metrics['learning_rate']})"
                    )

            # Final evaluation at the end of each epoch
            if not evaluation_ran:
                eval_metrics = run_evaluation()

            # save checkpoint after each epoch
            if not save_model_ran:
                run_save_model(state, eval_metrics)


if __name__ == "__main__":
    main()

import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional
import wandb
import jax
import jax.numpy as jnp
from jax.experimental.compilation_cache import compilation_cache as cc
import optax
import transformers
from flax import jax_utils
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.training.common_utils import shard, shard_prng_key
from huggingface_hub import Repository
from lpips_j.lpips import LPIPS
from tqdm import tqdm
from transformers import HfArgumentParser, set_seed
from transformers.utils import get_full_repo_name
from vit_vqgan import ViTVQConfig, ViTVQModel
from vit_vqgan.data import Dataset

cc.initialize_cache("./jax_cache", max_cache_size_bytes=10 * 2**30)

logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
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
    do_eval: bool = field(
        default=False, metadata={"help": "Whether to run eval on the dev set."}
    )
    commitment_cost: float = field(default=0.25, metadata={"help": "Commitment cost"})
    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate for AdamW."}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    adam_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for AdamW optimizer"}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"help": "Beta2 for AdamW optimizer"}
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."}
    )
    adafactor: bool = field(
        default=False,
        metadata={"help": "Whether or not to replace AdamW by Adafactor."},
    )
    num_train_epochs: float = field(
        default=3.0, metadata={"help": "Total number of training epochs to perform."}
    )
    warmup_steps: int = field(
        default=500, metadata={"help": "Linear warmup over warmup_steps."}
    )
    logging_steps: int = field(
        default=20, metadata={"help": "Log every X updates steps."}
    )
    save_steps: int = field(
        default=500, metadata={"help": "Save checkpoint every X updates steps."}
    )
    eval_steps: int = field(
        default=None, metadata={"help": "Run an evaluation every X steps."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )
    push_to_hub: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to upload the trained model to the model hub after training."
        },
    )
    hub_model_id: str = field(
        default=None,
        metadata={
            "help": "The name of the repository to keep in sync with the local `output_dir`."
        },
    )
    hub_token: str = field(
        default=None, metadata={"help": "The token to use to push to the Model Hub."}
    )

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

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
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_folder: str = field(
        metadata={
            "help": "Path to the root training directory which contains tfrecords."
        }
    )
    valid_folder: str = field(
        metadata={
            "help": "Path to the root validation directory which contains tfrecords."
        }
    )
    batch_size: Optional[int] = field(
        default=64, metadata={"help": "Batch size for training."}
    )
    image_size: Optional[int] = field(
        default=256, metadata={"help": " The size (resolution) of each image."}
    )
    seed_dataset: Optional[int] = field(
        default=None,
        metadata={"help": "The seed used to augment the dataset."},
    )
    format: Optional[str] = field(
        default="rgb", metadata={"help": "The format of the images (rgb or lab)."}
    )


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray = None

    def replicate(self):
        return jax_utils.replicate(self).replace(
            dropout_rng=shard_prng_key(self.dropout_rng)
        )


def create_learning_rate_fn(
    num_warmup_steps: int,
    learning_rate: float,
) -> Callable[[int], jnp.array]:
    """Returns the learning rate function."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps + 1
    )
    return warmup_fn


def get_config(model_args, data_args, training_args):
    if model_args.config_name is None:
        return ViTVQConfig(
            image_size=data_args.image_size,
            commitment_cost=training_args.commitment_cost,
            cache_dir=model_args.cache_dir,
        )
    return ViTVQConfig.from_pretrained(
        model_args.config_name,
        image_size=data_args.image_size,
        commitment_cost=training_args.commitment_cost,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )


def flat_args(model_args, data_args, training_args):
    args = asdict(model_args)
    args.update(asdict(data_args))
    args.update(asdict(training_args))
    return args


def init_lpips(rng, data_args):
    x = jax.random.normal(rng, shape=(1, data_args.image_size, data_args.image_size, 3))
    lpips = LPIPS()
    lpips_params = lpips.init(rng, x, x)
    return lpips, lpips_params


assert jax.local_device_count() == 8


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

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
    set_seed(training_args.seed)

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
    dataset = Dataset(**asdict(data_args))

    # Load or create model
    config = get_config(model_args, data_args, training_args)
    model = ViTVQModel(
        config,
        seed=training_args.seed,
        dtype=getattr(jnp, model_args.dtype),
    )

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(data_args.batch_size) * jax.device_count()
    eval_batch_size = train_batch_size

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Create learning rate schedule
    lr_schedule_fn = create_learning_rate_fn(
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    # create adam optimizer
    adamw = optax.adamw(
        learning_rate=lr_schedule_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )

    # Setup train state
    state = TrainState.create(
        apply_fn=model.__call__, params=model.params, tx=adamw, dropout_rng=dropout_rng
    )

    # Should we replicate these params?
    lpips, lpips_params = init_lpips(rng, data_args)

    # Define gradient update step fn
    def train_step(state, batch):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def compute_loss(params):
            predicted_images, (q_latent_loss, e_latent_loss) = state.apply_fn(
                batch, params=params, dropout_rng=dropout_rng, train=True
            )
            loss_l1 = jnp.mean(jnp.abs(predicted_images - batch))
            loss_l2 = jnp.mean((predicted_images - batch) ** 2)
            loss_lpips = jnp.mean(lpips.apply(lpips_params, batch, predicted_images))
            loss = (
                model.config.cost_l1 * loss_l1
                + model.config.cost_l2 * loss_l2
                + model.config.cost_q_latent * q_latent_loss
                + model.config.cost_e_latent * e_latent_loss
                + model.config.cost_lpips * loss_lpips
            )
            return loss, {
                "loss_l1": model.config.cost_l1 * loss_l1,
                "loss_l2": model.config.cost_l2 * loss_l2,
                "loss_q_latent": model.config.cost_q_latent * q_latent_loss,
                "loss_e_latent": model.config.cost_e_latent * e_latent_loss,
                "loss_lpips": model.config.cost_lpips * loss_lpips,
            }

        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
        (loss, loss_details), grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

        metrics = {
            "loss": loss,
            "learning_rate": lr_schedule_fn(state.step),
            **loss_details,
        }
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics

    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    # Replicate the train state on each device
    state = state.replicate()

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Total batch size = {data_args.batch_size}")

    train_time = 0
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)

    wandb.init(config=flat_args(model_args, data_args, training_args))
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()

        train_step_progress_bar = tqdm(desc="Training...", position=1, leave=False)

        # train
        for i, batch in enumerate(dataset.train.as_numpy_iterator()):
            batch = shard(batch)
            state, train_metric = p_train_step(state, batch)
            train_step_progress_bar.update(1)

            if jax.process_index() == 0:
                if i % training_args.logging_steps == 0:
                    metrics = unreplicate(train_metric)
                    wandb.log(metrics)

                if i % training_args.save_steps == 0:
                    params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
                    model.save_pretrained(training_args.output_dir, params=params)

        train_time += time.time() - train_start

        train_metric = unreplicate(train_metric)

        train_step_progress_bar.close()
        epochs.write(
            f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {train_metric['loss']}, Learning Rate:"
            f" {train_metric['learning_rate']})"
        )

        # save checkpoint after each epoch and push checkpoint to the hub
        if jax.process_index() == 0 and ((epoch + 1) % 10 == 0):
            params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
            model.save_pretrained(training_args.output_dir, params=params)
            if training_args.push_to_hub:
                repo.push_to_hub(
                    commit_message=f"Saving weights and logs of epoch {epoch}",
                    blocking=False,
                )


if __name__ == "__main__":
    main()

import random
from dataclasses import dataclass, field
from pathlib import Path

import jax
import tensorflow as tf
import tensorflow_io as tfio


@dataclass
class Dataset:
    train_folder: str = None
    valid_folder: str = None
    batch_size: int = 64
    image_size: int = 256
    min_image_size = 512
    max_aspect_ratio = 2.0
    seed_dataset: int = None
    format: str = "rgb"  # rgb or lab
    train: tf.data.Dataset = field(init=False)
    valid: tf.data.Dataset = field(init=False)
    rng: tf.random.Generator = field(init=False)
    multi_hosts: bool = field(init=False)

    def __post_init__(self):
        # verify valid args
        assert self.format in ["rgb", "lab"], f"Invalid format: {self.format}"

        # define rng
        if self.seed_dataset is None:
            self.seed_dataset = random.randint(0, 2**32 - 1)
        self.rng = tf.random.Generator.from_seed(self.seed_dataset, alg="philox")

        # check if we are on multi-hosts
        self.multi_hosts = jax.process_count() > 1

        # define parsing function
        features = {
            "webp": tf.io.FixedLenFeature([], tf.string),
            "original_width": tf.io.FixedLenFeature([], tf.int64),
            "original_height": tf.io.FixedLenFeature([], tf.int64),
        }

        def _parse_function(example_proto):
            parsed_features = tf.io.parse_single_example(example_proto, features)
            return (
                parsed_features["webp"],
                parsed_features["original_width"],
                parsed_features["original_height"],
            )

        def _filter_function(image, width, height):
            # filter out images that are too small
            if self.min_image_size is not None and (
                width < self.min_image_size or height < self.min_image_size
            ):
                return False
            # filter out images that have wrong aspect ratio
            if self.max_aspect_ratio is not None and (
                tf.divide(tf.maximum(width, height), tf.minimum(width, height))
                > self.max_aspect_ratio
            ):
                return False
            return True

        def _parse_image(image, width, height):
            return tfio.image.decode_webp(image)[..., :3]

        def _augment(image, seed):
            # create a new seed
            new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
            # apply random crop
            return tf.image.stateless_random_crop(
                image, size=[self.image_size, self.image_size, 3], seed=new_seed
            )

        # augmentation wrapper
        def _augment_wrapper(image):
            seed = self.rng.make_seeds(2)[0]
            return _augment(image, seed)

        # normalization
        def _normalize(image):
            if self.format == "rgb":
                # project to [-1, 1]
                return tf.cast(image, tf.float32) / 255.0 * 2.0 - 1.0
            elif self.format == "lab":
                image = tf.cast(image, tf.float32) / 255.0
                # convert to lab
                image = tfio.experimental.color.rgb_to_lab(image)
                # project to [-1, 1]
                return image / tf.constant([50.0, 128.0, 128.0]) + tf.constant(
                    [-1.0, 0.0, 0.0]
                )

        for folder, dataset, augment in zip(
            [self.train_folder, self.valid_folder],
            ["train", "valid"],
            [True, False],
        ):
            if folder is not None:
                # load files
                files = [f"{Path(f)}" for f in Path(folder).glob("*.tfrecord")]
                assert len(files) > 0, f"No files found at folder: {folder}"

                # load dataset
                ds = tf.data.TFRecordDataset(files)

                if self.multi_hosts and dataset == "train":
                    # repeat indefinitely
                    ds = ds.repeat()

                # parse dataset
                ds = ds.map(
                    _parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
                )

                # filter dataset
                ds = ds.filter(_filter_function)

                # parse image
                ds = ds.map(
                    _parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
                )

                if augment:
                    ds = ds.shuffle(1000)
                    # note: validation set must already be downloaded at target size
                    ds = ds.map(
                        _augment_wrapper,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    )

                # batch, normalize and prefetch
                ds = ds.batch(self.batch_size, drop_remainder=True)
                ds = ds.map(
                    _normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
                ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                setattr(self, dataset, ds)

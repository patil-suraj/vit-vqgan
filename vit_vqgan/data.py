from dataclasses import dataclass, field
from pathlib import Path

import tensorflow as tf
import tensorflow_io as tfio


@dataclass
class Dataset:
    train_folder: str = None
    valid_folder: str = None
    batch_size: int = 64
    image_size: int = 256
    dataset_seed: int = 42
    format: str = "rgb"  # rgb or lab
    train: tf.data.Dataset = field(init=False)
    valid: tf.data.Dataset = field(init=False)
    rng: tf.random.Generator = field(init=False)

    def __post_init__(self):
        # verify valid args
        assert self.format in ["rgb", "lab"], f"Invalid format: {self.format}"

        # define parsing function
        features = {"webp": tf.io.FixedLenFeature([], tf.string)}

        def _parse_function(example_proto):
            parsed_features = tf.io.parse_single_example(example_proto, features)
            return tfio.image.decode_webp(parsed_features["webp"])[..., :3]

        # define augmentation function
        self.rng = tf.random.Generator.from_seed(self.dataset_seed, alg="philox")

        def _augment(image, seed):
            # create a new seed
            new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
            # apply random crop
            return tf.image.stateless_random_crop(image, size=[self.image_size, self.image_size, 3], seed=new_seed)

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
                return image / tf.constant([50.0, 128.0, 128.0]) + tf.constant([-1.0, 0.0, 0.0])

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

                # parse dataset
                ds = ds.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

                if augment:
                    ds = ds.shuffle(1000)
                    # note: validation set must already be downloaded at target size
                    ds = ds.map(
                        _augment_wrapper,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    )

                # batch, normalize and prefetch
                ds = ds.batch(self.batch_size)
                ds = ds.map(_normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                setattr(self, dataset, ds)

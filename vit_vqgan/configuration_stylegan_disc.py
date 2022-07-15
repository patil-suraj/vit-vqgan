from typing import Tuple, Union

from transformers import PretrainedConfig

from .utils import PretrainedFromWandbMixin


class StyleGANDiscriminatorConfig(PretrainedFromWandbMixin, PretrainedConfig):
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = (256, 256),
        base_features: int = 32,
        max_hidden_feature_size: int = 512,
        mbstd_group_size: int = None,
        mbstd_num_features: int = 1,
    ):
        super().__init__()
        self.image_size = image_size
        self.base_features = base_features
        self.max_hidden_feature_size = max_hidden_feature_size
        self.mbstd_group_size = mbstd_group_size
        self.mbstd_num_features = mbstd_num_features

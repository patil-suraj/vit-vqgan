from transformers import PretrainedConfig


class ViTVQGANConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=768,
        codebook_embed_dim=32,
        n_embed=8192,
        # commitment_cost should be set appropriately. It's often useful to try a couple
        # of values. It mostly depends on the scale of the reconstruction cost
        # (log p(x|z)). So if the reconstruction cost is 100x higher, the
        # commitment_cost should also be multiplied with the same amount.
        commitment_cost=0.25, 
        intermediate_size=768 * 4,
        num_hidden_layers=12,
        num_attention_heads=12,
        image_size=256,
        num_channels=3,
        patch_size=8,
        use_conv_patches=False,
        hidden_act="relu",
        extra_feed_forward_act="tanh",
        layer_norm_eps=0.00001,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.codebook_embed_dim = codebook_embed_dim
        self.n_embed = n_embed
        self.commitment_cost = commitment_cost
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.use_conv_patches = use_conv_patches
        self.image_size = image_size
        self.num_channels = num_channels
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.extra_feed_forward_act = extra_feed_forward_act

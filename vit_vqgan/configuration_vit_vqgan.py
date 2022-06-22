from transformers import PretrainedConfig

class ViTVQGANConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=768,
        embed_dim=32,
        n_embed=8192,
        intermediate_size=768*4,
        num_hidden_layers=12,
        num_attention_heads=12,
        image_size=256,
        num_channels=3,
        patch_size=8,
        hidden_act="tanh",
        layer_norm_eps=0.00001,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

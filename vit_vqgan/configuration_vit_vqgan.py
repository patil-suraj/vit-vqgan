from transformers import PretrainedConfig


class ViTVQConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=768,
        codebook_embed_dim=32,
        n_embed=16384,
        # commitment_cost should be set appropriately. It's often useful to try a couple
        # of values. It mostly depends on the scale of the reconstruction cost
        # (log p(x|z)). So if the reconstruction cost is 100x higher, the
        # commitment_cost should also be multiplied with the same amount.
        cost_recon=1.0,
        cost_q_latent=1.0,
        cost_e_latent=0.25,
        intermediate_size=768 * 4,
        num_hidden_layers=12,
        num_attention_heads=12,
        image_size=256,
        num_channels=3,
        patch_size=16,
        use_conv_patches=False,
        post_attention_conv=False,
        use_glu=False,
        hidden_act="relu",
        extra_projection=True,
        extra_feed_forward_act="tanh",
        layer_norm_eps=0.00001,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.01,
        use_bias=False,
        ln_positions="preln",  # preln, normformer
        **kwargs
    ):
        super().__init__(**kwargs)

        self.cost_recon = cost_recon
        self.cost_q_latent = cost_q_latent
        self.cost_e_latent = cost_e_latent
        self.hidden_size = hidden_size
        self.codebook_embed_dim = codebook_embed_dim
        self.n_embed = n_embed
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.use_conv_patches = use_conv_patches
        self.post_attention_conv = post_attention_conv
        self.use_glu = use_glu
        self.image_size = image_size
        self.num_channels = num_channels
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.extra_projection = extra_projection
        self.extra_feed_forward_act = extra_feed_forward_act
        self.use_bias = use_bias
        self.ln_positions = ln_positions

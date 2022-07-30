from transformers import PretrainedConfig

from .utils import PretrainedFromWandbMixin


class ViTVQConfig(PretrainedFromWandbMixin, PretrainedConfig):
    def __init__(
        self,
        hidden_size=768,
        codebook_embed_dim=32,
        n_embed=16384,
        # cost_e_latent is commitment_cost and should be set appropriately. It's often useful to try a couple
        # of values. It mostly depends on the scale of the reconstruction cost
        # (log p(x|z)). So if the reconstruction cost is 100x higher, the
        # commitment_cost should also be multiplied with the same amount.
        cost_l1=1.0,
        cost_l2=1.0,
        cost_q_latent=1.0,
        cost_e_latent=0.25,
        cost_lpips=0.1,
        intermediate_size=768 * 4,
        num_hidden_layers=12,
        num_attention_heads=12,
        image_size=256,
        num_channels=3,
        patch_size=16,
        use_conv_patches=False,
        post_attention_conv=False,
        mid_ffn_conv=False,
        use_glu=False,
        hidden_act="relu",
        layer_norm_eps=0.00001,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        use_bias=False,
        ln_positions="preln",  # preln, normformer
        gradient_checkpointing=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.cost_l1 = cost_l1
        self.cost_l2 = cost_l2
        self.cost_q_latent = cost_q_latent
        self.cost_e_latent = cost_e_latent
        self.cost_lpips = cost_lpips
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
        self.mid_ffn_conv = mid_ffn_conv
        self.use_glu = use_glu
        self.image_size = image_size
        self.num_channels = num_channels
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.use_bias = use_bias
        self.ln_positions = ln_positions
        self.gradient_checkpointing = gradient_checkpointing

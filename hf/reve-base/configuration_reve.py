from transformers import PretrainedConfig


class ReveConfig(PretrainedConfig):
    model_type = "reve"

    def __init__(
        self,
        embed_dim=512,
        depth=22,
        heads=8,
        head_dim=64,
        mlp_dim_ratio=2.66,
        use_geglu=True,
        freqs=4,
        noise_ratio=0.0025,
        patch_size=200,
        patch_overlap=20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        self.head_dim = head_dim
        self.mlp_dim_ratio = mlp_dim_ratio
        self.use_geglu = use_geglu
        self.freqs = freqs
        self.noise_ratio = noise_ratio
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap

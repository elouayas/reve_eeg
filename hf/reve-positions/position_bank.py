import torch
from transformers import PreTrainedModel

from .configuration_bank import RevePositionBankConfig


class RevePositionBank(PreTrainedModel):
    config_class = RevePositionBankConfig

    def __init__(self, config: RevePositionBankConfig):
        super().__init__(config)

        self.position_names = config.position_names
        self.mapping = {name: i for i, name in enumerate(self.position_names)}
        self.register_buffer("embedding", torch.randn(len(self.position_names), 3))
        self.post_init()

    def forward(self, channel_names: list[str]):
        indices = [self.mapping[q] for q in channel_names if q in self.mapping]

        if len(indices) < len(channel_names):
            print(f"Found {len(indices)} positions out of {len(channel_names)} channels")

        indices = torch.tensor(indices, device=self.embedding.device)

        return self.embedding[indices]

    def get_all_positions(self):
        return self.position_names

from transformers import PretrainedConfig


class RevePositionBankConfig(PretrainedConfig):
    model_type = "reve-position-bank"

    def __init__(self, position_names: list[str] = [], **kwargs):
        super().__init__(**kwargs)
        self.position_names = position_names

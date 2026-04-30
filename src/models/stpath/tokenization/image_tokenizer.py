import torch
import torch.nn.functional as F

from .tokenizer_base import TokenizerBase


class ImageTokenizer(TokenizerBase):
    def __init__(
        self,
        feature_dim: int,
    ):
        super().__init__()
        self.feature_dim = feature_dim

    @property
    def n_tokens(self):
        return 2

    @property
    def mask_token(self) -> str:
        return F.one_hot(torch.tensor(1), num_classes=self.feature_dim).float()

    @property
    def mask_token_id(self) -> int:
        return 1

    @property
    def pad_token(self) -> str:
        return F.one_hot(torch.tensor(0), num_classes=self.feature_dim).float()

    @property
    def pad_token_id(self) -> int:
        return 0

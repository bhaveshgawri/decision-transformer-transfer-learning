import torch
import torch.nn.functional as F

from transformers import DecisionTransformerModel, DecisionTransformerConfig

from src.utils import DEVICE

from typing import Dict, Any

class DecisionTransformer(DecisionTransformerModel):
    def __init__(self, config: DecisionTransformerConfig) -> None:
        super().__init__(config)

        self.action_dim = config.act_dim
        self.to(DEVICE)

    def forward(self, **x: Dict[str, Any]):
        return super().forward(**x)

    def calc_loss(self, inp: torch.Tensor, out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        i = inp.reshape(-1, self.action_dim)
        o = out.reshape(-1, self.action_dim)
        m = mask.reshape(-1)
        return F.mse_loss(i[m == 1], o[m == 1])

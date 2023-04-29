import torch
from typing import Dict, Any
from transformers import Trainer
from src.DTModel import DecisionTransformer

class DecisionTransformerTrainer(Trainer):
    def compute_loss(self, model: DecisionTransformer, inputs: Dict[str, torch.Tensor], 
                    return_outputs: bool=False) -> (tuple[Any, Any | dict] | Any):
        outputs = model.forward(**inputs)

        action_inp, action_out, mask = inputs['actions'], outputs[1], inputs['attention_mask']
        loss = model.calc_loss(action_inp, action_out, mask)
        return (loss, outputs) if return_outputs else loss 

from transformers import Trainer

class DecisionTransformerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model.forward(**inputs)

        action_inp, action_out, mask = inputs['actions'], outputs[1], inputs['attention_mask']
        loss = model.calc_loss(action_inp, action_out, mask)
        return (loss, outputs) if return_outputs else loss 

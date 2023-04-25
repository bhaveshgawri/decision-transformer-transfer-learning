from classes import CheetahTransformer, CheetahDataLoader

from transformers import DecisionTransformerConfig

import torch
from torch.optim import AdamW
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR

# state_dimension, action_dimension = 17, 6
# max_ep_len, train_ep_len = 1000, 20
# epochs, itr, batch_size = 40, 45, 64
# warmup_steps, grad_clip = 10000, 0.25
# drop_out, gamma = 0.1, 0.99
# lr, weight_decay = 0.0001, 0.0001



def train(max_ep_len, train_ep_len, gamma, state_dimension, action_dimension, drop_out, lr, weight_decay, warmup_steps, epochs, itr, batch_size, grad_clip):
    cdl = CheetahDataLoader(max_ep_len, train_ep_len, gamma)
    cfg = DecisionTransformerConfig(state_dim=state_dimension, act_dim=action_dimension, max_ep_len=max_ep_len, drop_out=drop_out)
    model = CheetahTransformer(cfg)
    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = LambdaLR(optim, lambda steps: min((steps + 1) / warmup_steps, 1))

    for epoch in range(epochs):
        for _ in range(itr):        
            x = cdl.get_batch(batch_size=batch_size)
            y_pred = model(**x)

            action_inp, action_out, mask = x['actions'], y_pred[1], x['attention_mask']
            loss = model.calc_loss(action_inp, action_out, mask)

            optim.zero_grad()
            loss.backward()

            clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            sched.step()

        print(f'On epoch: {epoch + 1}, loss: {loss}')
    return model


def save_model(model, env, type_, time):
    """
    env: cheetah / walker / hopper
    type_: ft (for a fine-tuned model) / sc (if trained from scratch)
    time: int(time.time())
    """
    torch.save(model.state_dict(), f'./models/{env}_{type_}_{time}.pt')
    return True

def load_weights(model, file_path):
    model.load_state_dict(torch.load(file_path))
    return model
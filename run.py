from classes import CheetahTransformer, CheetahDataLoader
from transformers import DecisionTransformerConfig
from torch.optim import AdamW
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR

state_dimension, action_dimension = 17, 6
max_ep_len, train_ep_len = 1000, 20
epochs, itr, batch_size = 40, 45, 64
warmup_steps, grad_clip = 10000, 0.25
drop_out = 0.1
lr, weight_decay = 0.0001, 0.0001

cdl = CheetahDataLoader(max_ep_len=max_ep_len, train_ep_len=train_ep_len, gamma=0.99)
cfg = DecisionTransformerConfig(state_dim=state_dimension, act_dim=action_dimension, max_ep_len=max_ep_len, drop_out=drop_out)
model = CheetahTransformer(cfg)
optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
sched = LambdaLR(optim, lambda steps: min((steps + 1) / warmup_steps, 1))

for epoch in range(epochs):
    for i in range(itr):        
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

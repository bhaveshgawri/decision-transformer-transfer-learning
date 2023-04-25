from run_utils import train, save_model

import time

runtime_env='dev'

state_dimension, action_dimension = 17, 6
max_ep_len, train_ep_len = 1000, 20
epochs, itr, batch_size = 40, 45, 64
warmup_steps, grad_clip = 10000, 0.25
drop_out, gamma = 0.1, 0.99
lr, weight_decay = 0.0001, 0.0001

if runtime_env == 'dev':
    epochs = 5
    itr = 1

curr_time = int(time.time())
model = train(max_ep_len, train_ep_len, gamma, state_dimension, action_dimension, 
                drop_out, lr, weight_decay, warmup_steps, epochs, itr, batch_size, 
                grad_clip)

save_model(model, env='cheetah', type_='sc', time=curr_time)




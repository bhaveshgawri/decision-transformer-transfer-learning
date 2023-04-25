import time

from classes import DecisionTransformerUtils

from transformers import DecisionTransformerConfig

runtime_env='dev'
# runtime_env='prod'

state_dimension, action_dimension = 17, 6
max_ep_len, train_ep_len = 1000, 20
epochs, itr, batch_size = 40, 45, 64
warmup_steps, grad_clip = 10000, 0.25
drop_out, gamma = 0.1, 0.99
lr, weight_decay = 0.0001, 0.0001

if runtime_env == 'dev':
    epochs = 10
    itr = 45

curr_time = int(time.time())

cheetah_cfg = DecisionTransformerConfig(state_dim=17, act_dim=6, max_ep_len=max_ep_len, drop_out=drop_out)
cheetahTransformer = DecisionTransformerUtils(None, cheetah_cfg, max_ep_len, train_ep_len, gamma, lr, weight_decay, warmup_steps, True, 
                             dataset_path="edbeeching/decision_transformer_gym_replay", dataset_name="halfcheetah-expert-v2")

cheetahTransformer.train(epochs, itr, batch_size, grad_clip)
cheetahTransformer.save_model(env='cheetah', type_='sc', time=curr_time)




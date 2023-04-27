import time

from decision_transformer import DecisionTransformer
from DecisionTransformerEvaluator import DecisionTransformerEvaluator

from transformers import DecisionTransformerConfig

runtime_env='dev'
runtime_env='prod'

platform = 'pt'

max_ep_len = 1000
train_ep_len = test_ep_len = 20 # K
train_epochs, batch_size, test_epochs = 100, 64, 1
warmup_steps, warmup_ratio, grad_clip = 300, 0.1, 0.25
drop_out, gamma = 0.1, 1
lr, weight_decay = 0.0001, 0.0001
reward_scale = 1000
save_steps, logging_steps = 500, 5
if runtime_env == 'dev':
    train_epochs = 4


def train():
    curr_time = int(time.time())
    cheetah_cfg = DecisionTransformerConfig(state_dim=17, act_dim=6, max_ep_len=max_ep_len, drop_out=drop_out)
    cheetahTransformerRunner = DecisionTransformer(platform, None, cheetah_cfg, max_ep_len, train_ep_len, gamma, lr, 
                                                        weight_decay, warmup_steps, warmup_ratio, True, 
                                                        "edbeeching/decision_transformer_gym_replay", 
                                                        "halfcheetah-expert-v2", reward_scale, grad_clip)

    cheetahTransformerRunner.train_and_save(train_epochs, batch_size, 'cheetah', 'sc', curr_time, save_steps, logging_steps)    

def eval():
    cheetahEval = DecisionTransformerEvaluator.load_weights('hf', './cache/hf/cheetah_sc_1682574846/config.json', './cache/hf/cheetah_sc_1682574846/checkpoint-75')
    cheetahEval.evaluate('HalfCheetah-v4', test_epochs, test_ep_len, './cache/hf/cheetah_sc_1682574846/output', reward_scale, target_reward=12000)


if __name__ == '__main__':
    # train()
    eval()
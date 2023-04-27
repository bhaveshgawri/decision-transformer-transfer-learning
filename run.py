import time

from decision_transformer import DecisionTransformerTrainer
from DecisionTransformerEvaluator import DecisionTransformerEvaluator

from transformers import DecisionTransformerConfig

runtime_env='dev'
runtime_env='prod'

max_ep_len = 1000
train_ep_len = test_ep_len = 20 # K
train_itr, batch_size, test_itr = 100, 64, 1
warmup_steps, grad_clip = 10000, 0.25
drop_out, gamma = 0.1, 1
lr, weight_decay = 0.0001, 0.0001
reward_scale = 1000
if runtime_env == 'dev':
    train_itr = 4


def train():
    curr_time = int(time.time())

    cheetah_cfg = DecisionTransformerConfig(state_dim=17, act_dim=6, max_ep_len=max_ep_len, drop_out=drop_out)
    cheetahTransformerRunner = DecisionTransformerTrainer(None, cheetah_cfg, max_ep_len, train_ep_len, gamma, lr, weight_decay, warmup_steps, True, 
                                dataset_path="edbeeching/decision_transformer_gym_replay", dataset_name="halfcheetah-expert-v2", return_scale=reward_scale)

    cheetahTransformerRunner.train(train_itr, batch_size, grad_clip)
    cheetahTransformerRunner.save_model(env='cheetah', type_='sc', time=curr_time)

def eval():
    cheetahEval = DecisionTransformerEvaluator.load_weights('cheetah_sc_1682544426')
    cheetahEval.evaluate('HalfCheetah-v4', test_itr, test_ep_len, reward_scale, target_reward=12000)


if __name__ == '__main__':
    train()
    # eval()
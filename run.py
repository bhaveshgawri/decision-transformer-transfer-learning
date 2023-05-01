import time

from src.utils import Properties
from src.DTRunner import DecisionTransformerRunner
from src.DTEvaluator import DecisionTransformerEvaluator

from transformers import DecisionTransformerConfig

runtime_env='dev'
runtime_env='prod'

platform = 'pt'

max_ep_len = 1000
train_ep_len = test_ep_len = 20 # K
train_epochs, batch_size, test_epochs = 100, 64, 1
warmup_steps, warmup_ratio, grad_clip = 320, 0.1, 0.25
drop_out, gamma = 0.1, 1
lr, weight_decay = 0.0001, 0.0001
reward_scale = 1000
save_steps, logging_steps = 801, 5
if runtime_env == 'dev':
    train_epochs = 1
    logging_steps = 1
    warmup_steps = 1
    save_steps=11


def train(props: Properties) -> None:
    curr_time = int(time.time())
    cfg = DecisionTransformerConfig(state_dim=props.get_state_dim(), act_dim=props.get_action_dim(), max_ep_len=max_ep_len, drop_out=drop_out)
    transformerRunner = DecisionTransformerRunner(platform, None, cfg, max_ep_len, train_ep_len, gamma, lr, 
                                                        weight_decay, warmup_steps, warmup_ratio, False, False,
                                                        "edbeeching/decision_transformer_gym_replay", 
                                                        props.get_dataset_name(), reward_scale, grad_clip, props)
    transformerRunner.train_and_save(train_epochs, batch_size, props.get_env(), props.get_type(), curr_time, save_steps, logging_steps)    


def eval(props: Properties, src_cfg_path: str, src_mdl_path: str, out_path: str, target: int) -> None:
    evaluator = DecisionTransformerEvaluator.load_weights(platform, src_cfg_path, src_mdl_path)
    evaluator.evaluate(props.get_gym_env(), test_epochs, test_ep_len, out_path, reward_scale, target_reward=target, render=True)


def finetune(props: Properties, src_cfg_path: str, src_mdl_path: str, only_final_layer: bool) -> None:
    curr_time = int(time.time())
    evaluator = DecisionTransformerEvaluator.load_weights(platform, src_cfg_path, src_mdl_path)
    transformerRunner = DecisionTransformerRunner(platform, evaluator.model, None, max_ep_len, train_ep_len, gamma, lr, 
                                                        weight_decay, warmup_steps, warmup_ratio, True, only_final_layer,
                                                        "edbeeching/decision_transformer_gym_replay", 
                                                        props.get_dataset_name(), reward_scale, grad_clip, props)
    transformerRunner.train_and_save(train_epochs, batch_size, props.get_env(), props.get_type(), curr_time, save_steps, logging_steps) 


if __name__ == '__main__':
    # train(Properties('cheetah', 'sc'))
    # eval(Properties('cheetah', 'sc'), './cache/pt/configs/cheetah_sc_1682575937_1500.json', './cache/pt/models/cheetah_sc_1682575937_1500.pt', './cache/pt/outputs/cheetah_sc_1682575937_1500', 10000)    #pt
    # eval(Properties('cheetah', 'sc'), 'cache/hf/cheetah_sc_1682577402/config.json', 'cache/hf/cheetah_sc_1682577402/checkpoint-1600', 'cache/hf/cheetah_sc_1682577402/output', 10000)                     #hf
    # finetune(Properties('hopper', 'ft'), './cache/pt/configs/cheetah_sc_1682575937_1500.json', './cache/pt/models/cheetah_sc_1682575937_1500.pt', False)
    # train(Properties('cheetah', 'sc'))
    # train(Properties('hopper', 'sc'))
    # train(Properties('walker', 'sc'))
    finetune(Properties('cheetah', 'embed_ft_cheetah_sc_1682905510_1500'), './cache/pt/configs/cheetah_sc_1682905510_1500.json', './cache/pt/models/cheetah_sc_1682905510_1500.pt', True)
    finetune(Properties('hopper', 'embed_ft_cheetah_sc_1682905510_1500'), './cache/pt/configs/cheetah_sc_1682905510_1500.json', './cache/pt/models/cheetah_sc_1682905510_1500.pt', True)
    finetune(Properties('walker', 'embed_ft_cheetah_sc_1682905510_1500'), './cache/pt/configs/cheetah_sc_1682905510_1500.json', './cache/pt/models/cheetah_sc_1682905510_1500.pt', True)
    finetune(Properties('cheetah', 'embed_ft_hopper_sc_1682909019_1600'), './cache/pt/configs/hopper_sc_1682909019_1600.json', './cache/pt/models/hopper_sc_1682909019_1600.pt', True)
    finetune(Properties('hopper', 'embed_ft_hopper_sc_1682909019_1600'), './cache/pt/configs/hopper_sc_1682909019_1600.json', './cache/pt/models/hopper_sc_1682909019_1600.pt', True)
    finetune(Properties('walker', 'embed_ft_hopper_sc_1682909019_1600'), './cache/pt/configs/hopper_sc_1682909019_1600.json', './cache/pt/models/hopper_sc_1682909019_1600.pt', True)
    finetune(Properties('cheetah', 'embed_ft_walker_sc_1682910403_1500'), './cache/pt/configs/walker_sc_1682910403_1500.json', './cache/pt/models/walker_sc_1682910403_1500.pt', True)
    finetune(Properties('hopper', 'embed_ft_walker_sc_1682910403_1500'), './cache/pt/configs/walker_sc_1682910403_1500.json', './cache/pt/models/walker_sc_1682910403_1500.pt', True)
    finetune(Properties('walker', 'embed_ft_walker_sc_1682910403_1500'), './cache/pt/configs/walker_sc_1682910403_1500.json', './cache/pt/models/walker_sc_1682910403_1500.pt', True)
    pass


"""
Cheetah scratch train config:
    target_reward = 12000
    platform = 'pt'
        max_ep_len = 1000
        train_ep_len = test_ep_len = 20 # K
        train_epochs, batch_size, test_epochs = 100, 64, 1
        warmup_steps, warmup_ratio, grad_clip = 300, 0.1, 0.25
        drop_out, gamma = 0.1, 1
        lr, weight_decay = 0.0001, 0.0001
        reward_scale = 1000
        save_steps, logging_steps = 500, 5
    platform = 'hf'
        max_ep_len = 1000
        train_ep_len = test_ep_len = 20 # K
        train_epochs, batch_size, test_epochs = 100, 64, 1
        warmup_steps, warmup_ratio, grad_clip = 320, 0.1, 0.25
        drop_out, gamma = 0.1, 1
        lr, weight_decay = 0.0001, 0.0001
        reward_scale = 1000
        save_steps, logging_steps = 800, 5

"""
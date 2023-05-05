import time
from typing import List

from src.utils import Properties
from src.DTRunner import DecisionTransformerRunner
from src.DTEvaluator import DecisionTransformerEvaluator
from src.plotter import plot

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
save_steps, logging_steps = 1700, 5
if runtime_env == 'dev':
    train_epochs = 1
    logging_steps = 1
    warmup_steps = 1
    save_steps=11


def train(props: Properties) -> None:
    curr_time = int(time.time())
    cfg = DecisionTransformerConfig(state_dim=props.get_state_dim(), act_dim=props.get_action_dim(), max_ep_len=max_ep_len, drop_out=drop_out)
    transformerRunner = DecisionTransformerRunner(platform, None, cfg, max_ep_len, train_ep_len, gamma, lr, 
                                                        weight_decay, warmup_steps, warmup_ratio, False, None,
                                                        "edbeeching/decision_transformer_gym_replay", 
                                                        props.get_dataset_name(), reward_scale, grad_clip, props)
    transformerRunner.train_and_save(train_epochs, batch_size, props.get_env(), props.get_type(), curr_time, save_steps, logging_steps)    


def eval(props: Properties, src_cfg_path: str, src_mdl_path: str, out_path: str, target: int) -> None:
    evaluator = DecisionTransformerEvaluator.load_weights(platform, src_cfg_path, src_mdl_path)
    evaluator.evaluate(props.get_gym_env(), test_epochs, test_ep_len, out_path, reward_scale, target_reward=target, render=True)


def finetune(props: Properties, src_cfg_path: str, src_mdl_path: str, encoder_ft_layers: List, curr_time: int) -> None:
    evaluator = DecisionTransformerEvaluator.load_weights(platform, src_cfg_path, src_mdl_path)
    transformerRunner = DecisionTransformerRunner(platform, evaluator.model, None, max_ep_len, train_ep_len, gamma, lr, 
                                                        weight_decay, warmup_steps, warmup_ratio, True, encoder_ft_layers,
                                                        "edbeeching/decision_transformer_gym_replay", 
                                                        props.get_dataset_name(), reward_scale, grad_clip, props)
    transformerRunner.train_and_save(train_epochs, batch_size, props.get_env(), props.get_type(), curr_time, save_steps, logging_steps) 


if __name__ == '__main__':
    # training
    # train(Properties('cheetah', 'sc'))
    # train(Properties('hopper', 'sc'))
    # train(Properties('walker', 'sc'))

    # evaluation
    # eval(Properties('walker', 'sc'), './cache/pt/configs/walker_sc_1682910403_1500.json', './cache/pt/models/walker_sc_1682910403_1500.pt', './cache/pt/outputs/walker_sc_1682910403_1500', 5000)
    
    # finetuning
    # curr_time = int(time.time())
    # for ft_layers in [[], [2], [1, 2], [0, 1, 2]]:
    #     finetune(Properties('cheetah', f'h{len(ft_layers)}_ft_cheetah_sc_1682905510_1500'), './cache/pt/configs/cheetah_sc_1682905510_1500.json', './cache/pt/models/cheetah_sc_1682905510_1500.pt', ft_layers, curr_time)
    #     finetune(Properties('hopper', f'h{len(ft_layers)}_ft_cheetah_sc_1682905510_1500'), './cache/pt/configs/cheetah_sc_1682905510_1500.json', './cache/pt/models/cheetah_sc_1682905510_1500.pt', ft_layers, curr_time)
    #     finetune(Properties('walker', f'h{len(ft_layers)}_ft_cheetah_sc_1682905510_1500'), './cache/pt/configs/cheetah_sc_1682905510_1500.json', './cache/pt/models/cheetah_sc_1682905510_1500.pt', ft_layers, curr_time)
    #     finetune(Properties('cheetah', f'h{len(ft_layers)}_ft_hopper_sc_1682909019_1600'), './cache/pt/configs/hopper_sc_1682909019_1600.json', './cache/pt/models/hopper_sc_1682909019_1600.pt', ft_layers, curr_time)
    #     finetune(Properties('hopper', f'h{len(ft_layers)}_ft_hopper_sc_1682909019_1600'), './cache/pt/configs/hopper_sc_1682909019_1600.json', './cache/pt/models/hopper_sc_1682909019_1600.pt', ft_layers, curr_time)
    #     finetune(Properties('walker', f'h{len(ft_layers)}_ft_hopper_sc_1682909019_1600'), './cache/pt/configs/hopper_sc_1682909019_1600.json', './cache/pt/models/hopper_sc_1682909019_1600.pt', ft_layers, curr_time)
    #     finetune(Properties('cheetah', f'h{len(ft_layers)}_ft_walker_sc_1682910403_1500'), './cache/pt/configs/walker_sc_1682910403_1500.json', './cache/pt/models/walker_sc_1682910403_1500.pt', ft_layers, curr_time)
    #     finetune(Properties('hopper', f'h{len(ft_layers)}_ft_walker_sc_1682910403_1500'), './cache/pt/configs/walker_sc_1682910403_1500.json', './cache/pt/models/walker_sc_1682910403_1500.pt', ft_layers, curr_time)
    #     finetune(Properties('walker', f'h{len(ft_layers)}_ft_walker_sc_1682910403_1500'), './cache/pt/configs/walker_sc_1682910403_1500.json', './cache/pt/models/walker_sc_1682910403_1500.pt', ft_layers, curr_time)

    # plotting
    # plot_files = [{'cheetah_h0_ft_cheetah_sc_1682905510_1500_1683000901': 'finetune hidden layers: 0', 
    #               'cheetah_h1_ft_cheetah_sc_1682905510_1500_1683000901': 'finetune hidden layers: 1', 
    #               'cheetah_h2_ft_cheetah_sc_1682905510_1500_1683000901': 'finetune hidden layers: 2', 
    #               'cheetah_h3_ft_cheetah_sc_1682905510_1500_1683000901': 'finetune hidden layers: 3 (full model)', 
    #               'cheetah_sc_1682905510': 'scratch'},
    #               {'cheetah_h0_ft_hopper_sc_1682909019_1600_1683000901': 'finetune hidden layers: 0', 
    #               'cheetah_h1_ft_hopper_sc_1682909019_1600_1683000901': 'finetune hidden layers: 1', 
    #               'cheetah_h2_ft_hopper_sc_1682909019_1600_1683000901': 'finetune hidden layers: 2', 
    #               'cheetah_h3_ft_hopper_sc_1682909019_1600_1683000901': 'finetune hidden layers: 3 (full model)', 
    #               'cheetah_sc_1682905510': 'scratch'},
    #               {'cheetah_h0_ft_walker_sc_1682910403_1500_1683000901': 'finetune hidden layers: 0', 
    #               'cheetah_h1_ft_walker_sc_1682910403_1500_1683000901': 'finetune hidden layers: 1', 
    #               'cheetah_h2_ft_walker_sc_1682910403_1500_1683000901': 'finetune hidden layers: 2', 
    #               'cheetah_h3_ft_walker_sc_1682910403_1500_1683000901': 'finetune hidden layers: 3 (full model)', 
    #               'cheetah_sc_1682905510': 'scratch'},

    #               {'hopper_h0_ft_cheetah_sc_1682905510_1500_1683000901': 'finetune hidden layers: 0', 
    #               'hopper_h1_ft_cheetah_sc_1682905510_1500_1683000901': 'finetune hidden layers: 1', 
    #               'hopper_h2_ft_cheetah_sc_1682905510_1500_1683000901': 'finetune hidden layers: 2', 
    #               'hopper_h3_ft_cheetah_sc_1682905510_1500_1683000901': 'finetune hidden layers: 3 (full model)', 
    #               'hopper_sc_1682909019': 'scratch'},
    #               {'hopper_h0_ft_hopper_sc_1682909019_1600_1683000901': 'finetune hidden layers: 0', 
    #               'hopper_h1_ft_hopper_sc_1682909019_1600_1683000901': 'finetune hidden layers: 1', 
    #               'hopper_h2_ft_hopper_sc_1682909019_1600_1683000901': 'finetune hidden layers: 2', 
    #               'hopper_h3_ft_hopper_sc_1682909019_1600_1683000901': 'finetune hidden layers: 3 (full model)', 
    #               'hopper_sc_1682909019': 'scratch'},
    #               {'hopper_h0_ft_walker_sc_1682910403_1500_1683000901': 'finetune hidden layers: 0', 
    #               'hopper_h1_ft_walker_sc_1682910403_1500_1683000901': 'finetune hidden layers: 1', 
    #               'hopper_h2_ft_walker_sc_1682910403_1500_1683000901': 'finetune hidden layers: 2', 
    #               'hopper_h3_ft_walker_sc_1682910403_1500_1683000901': 'finetune hidden layers: 3 (full model)', 
    #               'hopper_sc_1682909019': 'scratch'},

    #               {'walker_h0_ft_cheetah_sc_1682905510_1500_1683000901': 'finetune hidden layers: 0', 
    #               'walker_h1_ft_cheetah_sc_1682905510_1500_1683000901': 'finetune hidden layers: 1', 
    #               'walker_h2_ft_cheetah_sc_1682905510_1500_1683000901': 'finetune hidden layers: 2', 
    #               'walker_h3_ft_cheetah_sc_1682905510_1500_1683000901': 'finetune hidden layers: 3 (full model)', 
    #               'walker_sc_1682910403': 'scratch'},
    #               {'walker_h0_ft_hopper_sc_1682909019_1600_1683000901': 'finetune hidden layers: 0', 
    #               'walker_h1_ft_hopper_sc_1682909019_1600_1683000901': 'finetune hidden layers: 1', 
    #               'walker_h2_ft_hopper_sc_1682909019_1600_1683000901': 'finetune hidden layers: 2', 
    #               'walker_h3_ft_hopper_sc_1682909019_1600_1683000901': 'finetune hidden layers: 3 (full model)', 
    #               'walker_sc_1682910403': 'scratch'},
    #               {'walker_h0_ft_walker_sc_1682910403_1500_1683000901': 'finetune hidden layers: 0', 
    #               'walker_h1_ft_walker_sc_1682910403_1500_1683000901': 'finetune hidden layers: 1', 
    #               'walker_h2_ft_walker_sc_1682910403_1500_1683000901': 'finetune hidden layers: 2', 
    #               'walker_h3_ft_walker_sc_1682910403_1500_1683000901': 'finetune hidden layers: 3 (full model)', 
    #               'walker_sc_1682910403': 'scratch'}
    # ]
    # titles =['cheetah_fine_tuned_on_cheetah', 
    #          'cheetah_fine_tuned_on_hopper',
    #          'cheetah_fine_tuned_on_walker',

    #          'hopper_fine_tuned_on_cheetah', 
    #          'hopper_fine_tuned_on_hopper',
    #          'hopper_fine_tuned_on_walker',

    #          'walker_fine_tuned_on_cheetah', 
    #          'walker_fine_tuned_on_hopper',
    #          'walker_fine_tuned_on_walker']

    # for file_grp, title in zip(plot_files, titles):
    #     plot(file_grp, title)
    pass
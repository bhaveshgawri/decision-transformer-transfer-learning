import numpy as np

import time, csv
from typing import List, Dict, Any

import torch
from torch.optim import AdamW
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from transformers import DecisionTransformerConfig, TrainingArguments

from src.DTModel import DecisionTransformer
from src.DTDataset import DecisionTransformerDataset
from src.DTTrainer import DecisionTransformerTrainer
from src.utils import Properties, DEVICE
from src.DTEvaluator import DecisionTransformerEvaluator

class DecisionTransformerRunner:
    def __init__(self, platform: str, model: DecisionTransformer, config: DecisionTransformerConfig, 
                max_ep_len: int, train_ep_len: int, gamma: float, lr: float, weight_decay: float, 
                warmup_steps: int, warmup_ratio: float, fine_tune: bool, dataset_path: str, dataset_name: str, 
                return_scale: int, grad_clip: float, props: Properties) -> None:

        if model is not None:
            self.model = model
            self.config = model.config
        elif config is not None:
            self.config = config
            self.model = DecisionTransformer(self.config)
        else:
            raise 'Both model and config can\'t be None!'
        
        if fine_tune:
            self.modify_model_arch(props)
            self.config = self.model.config

        self.grad_clip = grad_clip
        self.platform = platform
        self.props = props

        self.model.train()
        self.dtds = DecisionTransformerDataset(max_ep_len, train_ep_len, gamma, 
                                               dataset_path, dataset_name, return_scale)
        self.evaluator = DecisionTransformerEvaluator(None, self.config, self.dtds.mean, self.dtds.std)

        if self.platform == 'pt':
            self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            self.sched = LambdaLR(self.optim, lambda steps: min((steps + 1) / warmup_steps, 1))

        elif self.platform == 'hf':
            self.training_args = TrainingArguments(
                output_dir='./',
                remove_unused_columns=False,
                learning_rate=lr,
                weight_decay=weight_decay,
                warmup_steps=warmup_steps,
                # warmup_ratio=warmup_ratio,
                max_grad_norm=self.grad_clip
            )
        else:
            raise 'Unknown platform {platform}.'
        
    def train_and_save(self, epochs: int, batch_size: int, env: str, type_: str, 
                       curr_time: int, save_steps: int, logging_steps: int) -> None:
        if self.platform == 'hf':
            self.trainHF(epochs, batch_size, env, type_, curr_time, save_steps, logging_steps)
        elif self.platform == 'pt':
            self.trainPT(epochs, batch_size, env, type_, curr_time, save_steps, logging_steps)

    def trainPT(self, epochs: int, batch_size: int, env: str, type_: str, 
                curr_time: int, save_steps: int, logging_steps: int) -> None:
        init_time, step_ct = time.time(), 0
        logs = []
        for epoch in range(1, epochs + 1):
            dl = DataLoader(self.dtds, batch_size=batch_size, drop_last=True, shuffle=True, pin_memory=True)
            iterator, batch_ct = iter(dl), len(dl)
            for i, x in enumerate(iterator):
                y_pred = self.model(**x)

                action_inp, action_out, mask = x['actions'], y_pred[1], x['attention_mask']
                loss = self.model.calc_loss(action_inp, action_out, mask)

                self.optim.zero_grad()
                loss.backward()

                clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optim.step()
                self.sched.step()
                step_ct += 1

                if step_ct % save_steps == 0:
                    self.save_model_pt(env, type_, curr_time, step_ct)

                if step_ct % logging_steps == 0:
                    rewards = self.evaluator.evaluate(self.props.get_gym_env(), 5, 
                                                      self.dtds.train_ep_len, None, 
                                                      self.dtds.return_scale, 
                                                      self.props.get_target_reward(), 
                                                      False, self.model).mean().item()
                    ep, tim = epoch + i/float(batch_ct), time.time() - init_time
                    print(f'Step: {step_ct}/{epochs*batch_ct}, loss: {loss}, rewards: {rewards}, time passed: {tim}s, epoch: {ep}')
                    logs.append({'loss': loss.item(), 'epochs': ep, 'steps': step_ct, 'time': tim, 'rewards': rewards})

        self.save_model_pt(env, type_, curr_time, step_ct)
        self.save_logs(logs, f'./cache/pt/logs/{env}_{type_}_{curr_time}.csv')
        print('Training complete!')

    def save_model_pt(self, env: str, type_: str, curr_time: int, step_ct: int) -> bool:
        """
        env: cheetah / walker / hopper
        type_: ft (for a fine-tuned model) / sc (if trained from scratch)
        time: int(time.time())
        """
        torch.save(self.model.cpu().state_dict(), f'./cache/pt/models/{env}_{type_}_{curr_time}_{step_ct}.pt')
        self.save_config(f'./cache/pt/configs/{env}_{type_}_{curr_time}_{step_ct}.json')
        
        return True

    def trainHF(self, epochs: int, batch_size: int, env: str, type_: str, 
                curr_time: int, save_steps: int, logging_steps: int) -> None:
        self.training_args.num_train_epochs = epochs
        self.training_args.per_device_train_batch_size = batch_size
        self.training_args.save_steps = save_steps
        self.training_args.logging_steps = logging_steps
        self.training_args.output_dir = f"./cache/hf/{env}_{type_}_{curr_time}/"

        trainer = DecisionTransformerTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dtds.data,
            data_collator=self.dtds
        )
        trainer.train()
        self.save_logs(trainer.state.log_history, f'{self.training_args.output_dir}logs.csv')
        self.save_config(f'{self.training_args.output_dir}config.json')
        print('Training complete!')

    def save_config(self, path: str) -> None:
        cfg = self.config.to_dict()
        cfg['train_data_mean'], cfg['train_data_std'] = self.dtds.mean.tolist(), self.dtds.std.tolist()
        DecisionTransformerConfig().from_dict(cfg).to_json_file(path)

    def save_logs(self, logs: List[Dict[str, Any]], path: str) -> bool:
        keys = logs[-1].keys() | logs[0].keys()
        logs_rounded = [{k: np.round(v, 4) if isinstance(v, float) else v for k, v in d.items()} for d in logs]

        with open(path, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(logs_rounded)

        return True
    
    def modify_model_arch(self, props: Properties) -> None:
        ft_act_dim = props.get_action_dim()
        ft_obs_dim = props.get_state_dim()
        if ft_act_dim != self.config.act_dim:
            self.model.embed_action = torch.nn.Linear(ft_act_dim, self.config.hidden_size, device=DEVICE)
            self.model.predict_action[0] = torch.nn.Linear(self.config.hidden_size, ft_act_dim, device=DEVICE)
            self.model.config.act_dim = ft_act_dim
            self.model.action_dim = ft_act_dim
        if ft_obs_dim != self.config.state_dim:
            self.model.embed_state = torch.nn.Linear(ft_obs_dim, self.config.hidden_size, device=DEVICE)
            self.model.predict_state = torch.nn.Linear(self.config.hidden_size, ft_obs_dim, device=DEVICE)
            self.model.config.state_dim = ft_obs_dim
import numpy as np

import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import DecisionTransformerModel, DecisionTransformerConfig, TrainingArguments, Trainer

import time, csv
from typing import Any, Dict, Tuple
from dataclasses import dataclass

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class DecisionTransformerDataset(Dataset):
    def __init__(self, max_ep_len: int, train_ep_len: int, gamma: float, path: str, name: str, return_scale: int) -> None:
        self.train_ep_len = train_ep_len
        self.max_ep_len = max_ep_len

        self.gamma = gamma
        self.return_scale = return_scale

        self.data = load_dataset(path, name, split='train')
        self.mean, self.std, self.p = self.get_obs_stats()

        self.state_dim = len(self.data[0]['observations'][0])
        self.action_dim = len(self.data[0]['actions'][0])

    def __len__(self) -> int:
        return len(self.data)

    # callable bec @dataclass
    def __call__(self, samples) -> Dict[str, Any]:
        batch_data_indices = np.random.choice(np.arange(len(self.p)), size=len(samples), p=self.p)

        states, actions, returns_to_go, timesteps, attn_masks = [], [], [], [], []
        for batch_data_index in batch_data_indices:
            sample = self.data[int(batch_data_index)]
            ep_len = len(sample['dones'])
            random_idx = np.random.choice(np.arange(ep_len))
            
            states.append(self.get_state(sample, random_idx))
            actions.append(self.get_action(sample, random_idx))
            returns_to_go.append(self.get_returns(sample, random_idx))
            timesteps.append(self.get_timestep(ep_len, random_idx))
            attn_masks.append(self.get_mask(ep_len, random_idx))
        
        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0)
        returns_to_go = torch.stack(returns_to_go, dim=0)
        timesteps = torch.stack(timesteps, dim=0)
        attn_masks = torch.stack(attn_masks, dim=0)

        return {
            "states": states.float(),
            "actions": actions.float(),
            "rewards": torch.FloatTensor([], device=DEVICE),
            "returns_to_go": returns_to_go.float(),
            "timesteps": timesteps.long(),
            "attention_mask": attn_masks.float(),
        }
    
    # iterable because inherits from Dataset
    def __getitem__(self, index) -> Any:
        sample = self.data[int(index)]
        ep_len = len(sample['dones'])
        random_idx = np.random.choice(np.arange(ep_len))

        state = self.get_state(sample, random_idx)
        action = self.get_action(sample, random_idx)
        returns_to_go = self.get_returns(sample, random_idx)
        timesteps = self.get_timestep(ep_len, random_idx)
        attn_mask = self.get_mask(ep_len, random_idx)

        return {
            "states": state.float(),
            "actions": action.float(),
            "rewards": torch.FloatTensor([], device=DEVICE),
            "returns_to_go": returns_to_go.float(),
            "timesteps": timesteps.long(),
            "attention_mask": attn_mask.float(),
        }
    

    def get_state(self, sample: Dict[str, Any], idx: int) -> torch.Tensor:
        state = torch.tensor(sample['observations'][idx: idx + self.train_ep_len], dtype=float, device=DEVICE)
        
        zeros = torch.zeros(self.train_ep_len - state.shape[0], self.state_dim, dtype=float, device=DEVICE)
        state = torch.cat([zeros, state], dim=0)
        state = (state - self.mean) / self.std
        return state

    def get_action(self, sample: Dict[str, Any], idx: int) -> torch.Tensor:
        action = torch.tensor(sample['actions'][idx: idx + self.train_ep_len], dtype=float, device=DEVICE)

        zeros = torch.ones(self.train_ep_len - action.shape[0], self.action_dim, dtype=float, device=DEVICE) * -10
        action = torch.cat([zeros, action], dim=0)
        return action

    def get_returns(self, sample: Dict[str, Any], idx: int) -> torch.Tensor:
        returns = sample['rewards']
        r = 0
        for i, reward in reversed(list(enumerate(returns))):
            r = reward + self.gamma * r
            returns[i] = r
        returns = torch.tensor(returns, dtype=float, device=DEVICE)[idx: idx + self.train_ep_len]

        zeros = torch.zeros(self.train_ep_len - returns.shape[0], dtype=float, device=DEVICE)
        returns = torch.cat([zeros, returns], dim=0) / self.return_scale
        return returns.unsqueeze(1)

    def get_timestep(self, ep_len: int, idx: int) -> torch.Tensor:
        time = torch.arange(start=idx, end=min(ep_len, idx + self.train_ep_len), dtype=torch.long, device=DEVICE)

        zeros = torch.zeros(self.train_ep_len - time.shape[0], dtype=float, device=DEVICE)
        time = torch.cat([zeros, time], dim=0)
        return time

    def get_mask(self, ep_len: int, idx: int) -> torch.Tensor:
        one_ct = min(ep_len, idx + self.train_ep_len) - idx
        ones = torch.ones(one_ct, dtype=float, device=DEVICE)

        zeros = torch.zeros(self.train_ep_len - one_ct, dtype=float, device=DEVICE)
        mask = torch.cat([zeros, ones], dim=0)
        return mask

    def get_obs_stats(self) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
        obs, ep_len = [], []
        for observation in self.data['observations']:
            obs.extend(observation)
            ep_len.append(len(observation))
        obs, ep_len = np.array(obs), np.array(ep_len)
        return obs.mean(axis=0), obs.std(axis=0) + 1e-6, ep_len / ep_len.sum()


class DT(DecisionTransformerModel):
    def __init__(self, config: DecisionTransformerConfig) -> None:
        super().__init__(config)

        self.action_dim = config.act_dim
        self.to(DEVICE)

    def forward(self, **x: Dict[str, Any]):
        return super().forward(**x)

    def calc_loss(self, inp: torch.Tensor, out: torch.Tensor, mask: torch.Tensor):
        i = inp.reshape(-1, self.action_dim)
        o = out.reshape(-1, self.action_dim)
        m = mask.reshape(-1)
        return F.mse_loss(i[m == 1], o[m == 1])


class DecisionTransformerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model.forward(**inputs)

        action_inp, action_out, mask = inputs['actions'], outputs[1], inputs['attention_mask']
        loss = model.calc_loss(action_inp, action_out, mask)
        return (loss, outputs) if return_outputs else loss 


class DecisionTransformer:
    def __init__(self, platform: str, model: DT, config: DecisionTransformerConfig, 
                max_ep_len: int, train_ep_len: int, gamma: float, lr: float, weight_decay: float, 
                warmup_steps: int, warmup_ratio: float, train: bool, dataset_path: str, dataset_name: str, 
                return_scale: int, grad_clip: float) -> None:
        
        if model is not None:
            self.model == model
        elif config is not None:
            self.config = config
            self.model = DT(self.config)
        else:
            raise 'Both model and config can\'t be None!'
        
        self.grad_clip = grad_clip
        self.platform = platform

        self.model.train()
        self.dtds = DecisionTransformerDataset(max_ep_len, train_ep_len, gamma, dataset_path, dataset_name, return_scale)

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
        
    def train_and_save(self, epochs: int, batch_size: int, env: str, type_: str, curr_time: int, save_steps: int, logging_steps: int):
        if self.platform == 'hf':
            self.trainHF(epochs, batch_size, env, type_, curr_time, save_steps, logging_steps)
        elif self.platform == 'pt':
            self.trainPT(epochs, batch_size, env, type_, curr_time, save_steps, logging_steps)

    def trainPT(self, epochs: int, batch_size: int, env: str, type_: str, curr_time: int, save_steps: int, logging_steps: int) -> None:
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
                    ep, tim = epoch + i/float(batch_ct), time.time() - init_time
                    print(f'Step: {step_ct}, Epoch: {ep}, loss: {loss}, seconds passed: {tim}')
                    logs.append({'loss': loss.item(), 'epochs': ep, 'steps': step_ct, 'timestep': tim})

        self.save_logs(logs, f'./cache/pt/logs/{env}_{type_}_{curr_time}.csv')
        print('Training complete!')

    def save_model_pt(self, env, type_, curr_time, step_ct) -> bool:
        """
        env: cheetah / walker / hopper
        type_: ft (for a fine-tuned model) / sc (if trained from scratch)
        time: int(time.time())
        """
        torch.save(self.model.cpu().state_dict(), f'./cache/pt/models/{env}_{type_}_{curr_time}_{step_ct}.pt')
        self.save_config(f'./cache/pt/configs/{env}_{type_}_{curr_time}_{step_ct}.json')
        
        return True

    def trainHF(self, epochs: int, batch_size: int, env: str, type_: str, curr_time: int, save_steps: int, logging_steps: int):
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

    def save_config(self, path):
        cfg = self.config.to_dict()
        cfg['train_data_mean'], cfg['train_data_std'] = self.dtds.mean.tolist(), self.dtds.std.tolist()
        DecisionTransformerConfig().from_dict(cfg).to_json_file(path)

    def save_logs(self, logs, path):
        keys = logs[-1].keys() | logs[0].keys()
        logs_rounded = [{k: np.round(v, 4) if isinstance(v, float) else v for k, v in d.items()} for d in logs]

        with open(path, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(logs_rounded)

        return True
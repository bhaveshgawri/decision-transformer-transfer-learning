import numpy as np

import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import DecisionTransformerModel, DecisionTransformerConfig

import time
from typing import Any, Dict, Tuple

from DecisionTransformerEvaluator import DecisionTransformerEvaluator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.data[index]
        ep_len = len(sample['dones'])
        
        state = self.get_state(sample, index)
        action = self.get_action(sample, index)
        returns_to_go = self.get_returns(sample, index)
        timesteps = self.get_timestep(ep_len, index)
        attn_mask = self.get_mask(ep_len, index)

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


class DecisionTransformer(DecisionTransformerModel):
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


class DecisionTransformerTrainer:
    def __init__(self, model, config: DecisionTransformerConfig, max_ep_len: int, train_ep_len: int, 
                gamma: float, lr: float, weight_decay: float, warmup_steps: int, train: bool,
                dataset_path: str, dataset_name: str, return_scale: int) -> None:
        
        if model is not None:
            self.model == model
        elif config is not None:
            self.config = config
            self.model = DecisionTransformer(self.config)
        else:
            raise 'Both model and config can\'t be None!'
        
        self.model.train()
        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.sched = LambdaLR(self.optim, lambda steps: min((steps + 1) / warmup_steps, 1))
        self.dtds = DecisionTransformerDataset(max_ep_len, train_ep_len, gamma, dataset_path, dataset_name, return_scale)

    def train(self, epochs: int, batch_size: int, grad_clip: float) -> None:
        for epoch in range(1, epochs + 1):
            iterator = iter(DataLoader(self.dtds, batch_size=batch_size, drop_last=True, shuffle=True, pin_memory=True))
            for x in iterator:
                y_pred = self.model.forward(**x)

                action_inp, action_out, mask = x['actions'], y_pred[1], x['attention_mask']
                loss = self.model.calc_loss(action_inp, action_out, mask)

                self.optim.zero_grad()
                loss.backward()

                clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optim.step()
                self.sched.step()

            if epoch % 20 == 0:
                e = DecisionTransformerEvaluator(self.model, self.config, self.dtds.mean, self.dtds.std, f'cheetah_sc_{int(time.time())}')
                e.evaluate('HalfCheetah-v4', 1, self.dtds.train_ep_len, self.dtds.return_scale, target_reward=12000)
                print(f'Completed epoch: {epoch}, loss: {loss}')
        print('Training complete.')

    def save_model(self, env: str, type_: str, time: int) -> bool:
        """
        env: cheetah / walker / hopper
        type_: ft (for a fine-tuned model) / sc (if trained from scratch)
        time: int(time.time())
        """
        torch.save(self.model.cpu().state_dict(), f'./cache/models/{env}_{type_}_{time}.pt')
        
        cfg = self.config.to_dict()
        mean, std, _ = self.dtds.get_obs_stats()
        cfg['train_data_mean'], cfg['train_data_std'] = mean.tolist(), std.tolist()
        DecisionTransformerConfig().from_dict(cfg).to_json_file(f'./cache/configs/{env}_{type_}_{time}.json')

        return True


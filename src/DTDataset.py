from src.utils import DEVICE

import numpy as np

import torch
from torch.utils.data import Dataset

from datasets import load_dataset

from dataclasses import dataclass
from typing import Any, Dict, Tuple, List


@dataclass
class DecisionTransformerDataset(Dataset):
    def __init__(self, max_ep_len: int, train_ep_len: int, gamma: float, 
                 path: str, name: str, return_scale: int) -> None:
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
    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    def __getitem__(self, index: int) -> Any:
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

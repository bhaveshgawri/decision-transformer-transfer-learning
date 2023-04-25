import numpy as np

import torch
from torch.optim import AdamW
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset

from datasets import load_dataset
from transformers import DecisionTransformerModel, DecisionTransformerConfig

from typing import Any, Dict

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DecisionTransformerDataLoader(Dataset):
    def __init__(self, max_ep_len, train_ep_len, gamma, path, name):
        self.train_ep_len = train_ep_len
        self.max_ep_len = max_ep_len

        self.gamma = gamma

        self.data = load_dataset(path, name, split='train')
        self.mean, self.std, self.p = self.get_obs_stats()

        self.state_dim = len(self.data[0]['observations'][0])
        self.action_dim = len(self.data[0]['actions'][0])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, Any]:
        return self.data[index]

    def get_batch(self, batch_size=8) -> Dict[str, Any]:
        batch_data_indices = np.random.choice(np.arange(len(self.p)), size=batch_size, p=self.p)

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

    def get_state(self, sample, idx):
        state = torch.tensor(sample['observations'][idx: idx + self.train_ep_len], dtype=float, device=DEVICE)
        
        zeros = torch.zeros(self.train_ep_len - state.shape[0], self.state_dim, dtype=float, device=DEVICE)
        state = torch.cat([zeros, state], dim=0)
        state = state - self.mean / self.std
        return state

    def get_action(self, sample, idx):
        action = torch.tensor(sample['actions'][idx: idx + self.train_ep_len], dtype=float, device=DEVICE)

        zeros = torch.zeros(self.train_ep_len - action.shape[0], self.action_dim, dtype=float, device=DEVICE)
        action = torch.cat([zeros, action], dim=0)
        return action

    def get_returns(self, sample, idx):
        returns = sample['rewards'][idx: idx + self.train_ep_len]
        r = 0
        for i, reward in reversed(list(enumerate(returns))):
            r = reward + self.gamma * r
            returns[i] = r
        returns = torch.tensor(returns, dtype=float, device=DEVICE)

        zeros = torch.zeros(self.train_ep_len - returns.shape[0], dtype=float, device=DEVICE)
        returns = torch.cat([zeros, returns], dim=0)
        return returns.unsqueeze(1)

    def get_timestep(self, ep_len, idx):
        time = torch.arange(start=idx, end=min(ep_len, idx + self.train_ep_len), dtype=torch.long, device=DEVICE)

        zeros = torch.zeros(self.train_ep_len - time.shape[0], dtype=float, device=DEVICE)
        time = torch.cat([zeros, time], dim=0)
        return time

    def get_mask(self, ep_len, idx):
        one_ct = min(ep_len, idx + self.train_ep_len) - idx
        ones = torch.ones(one_ct, dtype=float, device=DEVICE)

        zeros = torch.zeros(self.train_ep_len - one_ct, dtype=float, device=DEVICE)
        mask = torch.cat([zeros, ones], dim=0)
        return mask

    def get_obs_stats(self):
        obs, ep_len = [], []
        for observation in self.data['observations']:
            obs.extend(observation)
            ep_len.append(len(observation))
        obs, ep_len = np.array(obs), np.array(ep_len)
        return obs.mean(axis=0), obs.std(axis=0) + 1e-6, ep_len / ep_len.sum()


class DecisionTransformer(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

        self.action_dim = config.act_dim
        self.loss = torch.nn.MSELoss()
        self.to(DEVICE)

    def forward(self, **x):
        return super().forward(**x)

    def calc_loss(self, inp, out, mask):
        i = inp.reshape(-1, self.action_dim)
        o = out.reshape(-1, self.action_dim)
        m = mask.reshape(-1)
        return self.loss(i[m == 1], o[m == 1])


class DecisionTransformerTrainer:
    def __init__(self, model, config: DecisionTransformerConfig, max_ep_len: int, train_ep_len: int, 
                gamma: float, lr: float, weight_decay: float, warmup_steps: int, train: bool,
                dataset_path: str, dataset_name: str):
        
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
        self.cdl = DecisionTransformerDataLoader(max_ep_len, train_ep_len, gamma, dataset_path, dataset_name)

    def train(self, epochs, itr_per_epoch, batch_size, grad_clip):
        for epoch in range(epochs):
            for itr in range(itr_per_epoch):        
                x = self.cdl.get_batch(batch_size=batch_size)
                y_pred = self.model(**x)

                action_inp, action_out, mask = x['actions'], y_pred[1], x['attention_mask']
                loss = self.model.calc_loss(action_inp, action_out, mask)

                self.optim.zero_grad()
                loss.backward()

                clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optim.step()
                self.sched.step()

            print(f'Completed epoch: {epoch + 1}, loss: {loss}')
        print('Training complete.')

    def save_model(self, env, type_, time):
        """
        env: cheetah / walker / hopper
        type_: ft (for a fine-tuned model) / sc (if trained from scratch)
        time: int(time.time())
        """
        torch.save(self.model.cpu().state_dict(), f'./cache/models/{env}_{type_}_{time}.pt')
        
        cfg = self.config.to_dict()
        mean, std, _ = self.cdl.get_obs_stats()
        cfg['train_data_mean'], cfg['train_data_std'] = mean.tolist(), std.tolist()
        DecisionTransformerConfig().from_dict(cfg).to_json_file(f'./cache/configs/{env}_{type_}_{time}.json')

        return True

class DecisionTransformerEval:
    def __init__(self, model, config, train_data_mean, train_data_std):
        self.model = model
        self.config = config

        self.train_data_mean = train_data_mean
        self.train_data_std = train_data_std

    def evaluate(self, epochs, itr_per_epoch, batch_size):
        print(self.config, self.train_data_mean, self.train_data_std)

    @staticmethod
    def load_weights(file_name):
        config = DecisionTransformerConfig().from_json_file(f'./cache/configs/{file_name}.json').to_dict()
        mean, std = torch.tensor(config.pop('train_data_mean', None), device=DEVICE).float(), \
                    torch.tensor(config.pop('train_data_std', None), device=DEVICE).float()

        config = DecisionTransformerConfig().from_dict(config)
        model = DecisionTransformer(config)
        model.load_state_dict(torch.load(f'./cache/models/{file_name}.pt'))
        dt_eval = DecisionTransformerEval(model, config, mean, std)

        return dt_eval
        
        



        

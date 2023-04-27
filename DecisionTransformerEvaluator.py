import numpy as np

import torch
from transformers import DecisionTransformerConfig
import gymnasium as gym

from gif import GIFMaker

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DecisionTransformerEvaluator:
    def __init__(self, model, config: DecisionTransformerConfig, 
                train_data_mean: torch.Tensor, train_data_std: torch.Tensor, file_name: str) -> None:
        self.config = config
        self.model = model
        self.model.eval()

        self.obs_mean = train_data_mean
        self.obs_std = train_data_std

        self.file_name = file_name

        self.env = None
        self.g = GIFMaker()

    def reset_env(self):
        state = self.env.reset(seed = np.random.randint(low=1, high=100000))
        return torch.tensor(state[0].tolist(), device=DEVICE).float()

    def pad_inputs(self, observations, actions, returns_to_go, timesteps, timestep, test_ep_len):
        pad_len = max(0, test_ep_len - (timestep + 1))

        st, en = timestep + 1 - test_ep_len, timestep + 1
        if pad_len != 0:
            st = 0

        zeros = torch.zeros(1, pad_len, self.config.state_dim, device=DEVICE).float()
        obs = torch.cat([zeros, observations[:, st: en]], dim=1)

        zeros = torch.zeros(1, pad_len, self.config.act_dim, device=DEVICE).float()
        act = torch.cat([zeros, actions[:, st: en]], dim=1)

        zeros = torch.zeros(1, pad_len, 1, device=DEVICE).float()
        rtg = torch.cat([zeros, returns_to_go[:, st: en]], dim=1)

        zeros = torch.zeros(1, pad_len, device=DEVICE).long()
        tim = torch.cat([zeros, timesteps[:, st: en]], dim=1)

        zeros, ones = torch.zeros(1, pad_len, device=DEVICE).float(), torch.ones(1, test_ep_len - pad_len, device=DEVICE).float()
        msk = torch.cat([zeros, ones], dim=1)

        return {
            "states": obs,
            "actions": act,
            "rewards": torch.FloatTensor([], device=DEVICE),
            "returns_to_go": rtg,
            "timesteps": tim,
            "attention_mask": msk,
        }

    def evaluate(self, env: str, iterations: int, test_ep_len: int, reward_scale: float, target_reward: int=5000) -> None:
        self.env = gym.make(env, render_mode = "rgb_array")

        with torch.no_grad():
            for iteration in range(1, iterations + 1):
                render = True #if iteration == iterations else False
                
                observation = self.reset_env()

                observations = torch.zeros(1, self.config.max_ep_len, self.config.state_dim, device=DEVICE).float()
                actions = torch.zeros(1, self.config.max_ep_len, self.config.act_dim, device=DEVICE).float()
                returns_to_go = torch.zeros(1, self.config.max_ep_len, 1, device=DEVICE).float()
                timesteps = torch.arange(start=0, end=self.config.max_ep_len, device=DEVICE).unsqueeze(0).long()
                
                reward, ep_reward = 0, 0
                target_reward /= reward_scale

                for timestep in range(self.config.max_ep_len):
                    observations[0][timestep] = (observation - self.obs_mean) / self.obs_std
                    
                    returns_to_go[0][timestep] = target_reward - reward / reward_scale
                    target_reward = returns_to_go[0][timestep]

                    params = self.pad_inputs(observations, actions, returns_to_go, timesteps, timestep, test_ep_len)
                    y_pred = self.model(**params)
                    action = y_pred[1][0][-1]

                    observation, reward, terminated, truncated, _ = self.env.step(action.detach().cpu().numpy())
                    
                    observation = torch.tensor(observation.tolist(), device=DEVICE).float()
                    actions[0][timestep] = action

                    ep_reward += reward

                    if render: self.g.append(self.env.render())
                    if terminated or truncated: break

                if render:
                    self.g.save(f'{self.file_name}_{iteration}')
                    self.g.reset()
            self.env.close()

    # @staticmethod
    # def load_weights(file_name: str) -> 'DecisionTransformerEvaluator':
    #     config = DecisionTransformerConfig().from_json_file(f'./cache/configs/{file_name}.json').to_dict()
    #     mean, std = torch.tensor(config.pop('train_data_mean', None), device=DEVICE).float(), \
    #                 torch.tensor(config.pop('train_data_std', None), device=DEVICE).float()

    #     config = DecisionTransformerConfig().from_dict(config)
    #     model = DecisionTransformer(config)
    #     model.load_state_dict(torch.load(f'./cache/models/{file_name}.pt'))
    #     dt_eval = DecisionTransformerEvaluator(model, config, mean, std, file_name)

    #     return dt_eval

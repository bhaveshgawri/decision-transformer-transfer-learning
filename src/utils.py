import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Properties:
    def __init__(self, env, type_):
        self.env = env
        self.type_ = type_ # sc / ft

    def get_dataset_name(self):
        if self.env == 'cheetah':
            return 'halfcheetah-expert-v2'
        elif self.env == 'walker':
            return 'walker2d-expert-v2'
        elif self.env == 'hopper':
            return 'hopper-expert-v2'
        else: return None
    
    def get_action_dim(self):
        if self.env == 'cheetah':
            return 6
        elif self.env == 'walker':
            return 6
        elif self.env == 'hopper':
            return 3
        else: return -1

    def get_state_dim(self):
        if self.env == 'cheetah':
            return 17
        elif self.env == 'walker':
            return 17
        elif self.env == 'hopper':
            return 11
        else: return -1

    def get_target_reward(self):
        if self.env == 'cheetah':
            return 12000
        elif self.env == 'walker':
            return 5000
        elif self.env == 'hopper':
            return 5000
        else: return -1

    def get_gym_env(self):
        if self.env == 'cheetah':
            return 'HalfCheetah-v4'
        elif self.env == 'walker':
            return 'Walker2d-v4'
        elif self.env == 'hopper':
            return 'Hopper-v4'
        else: return None

    def get_env(self):
        return self.env

    def get_type(self):
        return self.type_
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self,observation_space,features_dim):
        super().__init__(observation_space,features_dim)
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(41,256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128,features_dim),
            torch.nn.Tanh()
        )

    def forward(self,state):
        return self.sequential(state)

class CustomNetWork(torch.nn.Module):
    def __init__(self,
                 feature_dim: int,
                 last_layer_dim_pi: int = 128,
                 last_layer_dim_vf: int = 64):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 1024),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, last_layer_dim_pi),
            torch.nn.Tanh(),
        )
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 1024),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, last_layer_dim_vf),
            torch.nn.Tanh(),
        )

    def forward(self,features):
        return self.forward_actor(features),self.forward_critic(features)

    def forward_actor(self,features):
        return self.policy_net(features)

    def forward_critic(self,features):
        return self.value_net(features)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self,
                 observation_space,
                 action_space,
                 lr_schedule,*args,**kwargs):
        super().__init__(observation_space,
                         action_space,
                         lr_schedule,
                         *args,**kwargs)
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetWork(self.features_dim)









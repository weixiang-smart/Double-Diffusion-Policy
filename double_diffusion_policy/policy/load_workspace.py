# import torch

# a = torch.tensor([[[1,2,3],[1,2,3],[1,2,3]],[[4,5,6],[4,5,6],[4,5,6]]])
# b = torch.tensor([[11,12,13,14],[11,12,13,14]])

# print(a.shape)

# a = a.reshape(a.shape[0], -1)

# a = torch.cat([a, b], dim=-1)

# print(a)


import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import hydra
import torch


class load_human_workspace():
    def __init__(self, 
        checkpoint
        ):
        super().__init__()
        self.checkpoint = checkpoint
        self.payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        self.cfg = self.payload['cfg']
        self.cls = hydra.utils.get_class(self.cfg._target_)
        self.workspace = self.cls(self.cfg, output_dir="data/lift_eval_output")
        self.workspace: BaseWorkspace
        self.workspace.load_payload(self.payload, exclude_keys=None, include_keys=None)
        self.policy = None

    def get_model(self):
        self.policy = self.workspace.model
        if self.cfg.training.use_ema:
            self.policy = self.workspace.ema_model
        print("------------------loading human model--------------------")
        return self.policy

if __name__ == '__main__':

    human_workspace = load_human_workspace("data/outputs/2024.02.29/20.53.33_train_diffusion_unet_hybrid_lift_image/checkpoints/epoch=0250-test_mean_score=1.000.ckpt")

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    human_policy = human_workspace.get_model()
    human_policy.to(device)
    human_policy.eval()

    a = human_policy()

# checkpoint = "data/outputs/2024.02.29/20.53.33_train_diffusion_unet_hybrid_lift_image/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
# output_dir =  "data/lift_eval_output"

# payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
# cfg = payload['cfg']
# cls = hydra.utils.get_class(cfg._target_)
# workspace = cls(cfg, output_dir=output_dir)
# workspace: BaseWorkspace
# workspace.load_payload(payload, exclude_keys=None, include_keys=None)


# policy = workspace.model
# if cfg.training.use_ema:
#     policy = workspace.ema_model


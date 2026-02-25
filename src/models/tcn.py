import torch
import torch.nn as nn
from src.models.encoders import BoneMLPEncoder
from src.models.pooling import PersonPooling


class MLP_POOL(nn.Module):
    def __init__(self, input_dim=97, num_classes=1,
                 mlp_out_dim=32, pool_mode="mean_max_std"):
        super().__init__()

        self.mlp = BoneMLPEncoder(out_dim=mlp_out_dim)
        self.pool = PersonPooling(mode=pool_mode)

        pool_mult = {
            "mean": 1,
            "max": 1,
            "mean_max": 2,
            "mean_max_std": 3,
        }
        if pool_mode not in pool_mult:
            raise ValueError(f"Unsupported pool_mode: {pool_mode}")

        in_ch = mlp_out_dim * pool_mult[pool_mode] + 1
        self.fc = nn.Linear(in_ch, 1)

    def forward(self, x):
        # x: (B, T, C)
        emb, person_mask = self.mlp(x)                       # (B,T,K,E), (B,T,K)
        scene = self.pool(emb, mask=person_mask)             # (B,T,E_scene)

        count = person_mask.sum(dim=2).float()               # (B,T)
        count_norm = count / person_mask.size(2)             # (B,T)

        scene_count = torch.cat(
            [scene, count_norm.unsqueeze(-1)], dim=-1
        )                                                    # (B,T,E_scene+1)

        last_scene = scene_count[:, -1, :]                   # (B,E_scene+1)
        logits = self.fc(last_scene).squeeze(-1)             # (B,)

        return logits
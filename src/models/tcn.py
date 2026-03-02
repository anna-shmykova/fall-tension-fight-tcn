import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.encoders import BoneMLPEncoder
from src.models.pooling import PersonPooling

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: (B, C_in, T)
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        return out

class EventTCNSimple(nn.Module):
    def __init__(self, input_dim, num_classes=1,
                 hidden_dim=64, num_layers=3):
        super().__init__()

        layers = []
        in_ch = input_dim
        dilation = 1
        for _ in range(num_layers):
            layers.append(
                TemporalConvBlock(
                    in_channels=in_ch,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    dilation=dilation,
                )
            )
            in_ch = hidden_dim
            dilation *= 2  # 1,2,4,...

        self.tcn = nn.Sequential(*layers)
        #self.head = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)
        self.fc = nn.Linear(hidden_dim, 1)   # <--- ONE logit

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)          # (B, C, T)
        feat = self.tcn(x)             # (B, hidden_dim, T)
        #logits = self.head(feat)       # (B, num_classes, T)
        #logits = logits.transpose(1, 2)  # (B, T, num_classes)
        mid = feat.size(-1) // 2
        last_feat = feat[:, :, mid]         # (B, hidden_dim)
        logits = self.fc(last_feat).squeeze(-1)  # (B,)  raw score
        return logits

class EventTCN(nn.Module):
    def __init__(self, input_dim=97, num_classes=1,
                 hidden_dim=64, num_layers=3, kernel_size=3,
                 mlp_out_dim=32, pool_mode="mean_max_std"):
        super().__init__()

        layers = []
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

        # TCN input channels must match pooled scene features (+1 for count_norm)
        in_ch = mlp_out_dim * pool_mult[pool_mode] + 1

        #print(in_ch)
        dilation = 1
        for _ in range(num_layers):
            layers.append(
                TemporalConvBlock(
                    in_channels=in_ch,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
            )
            in_ch = hidden_dim
            dilation *= 2  # 1,2,4,...
        self.tcn = nn.Sequential(*layers)
        #self.head = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)
        self.fc = nn.Linear(hidden_dim, 1)   # <--- ONE logit

    def forward(self, x):
        # x: (B, T, C)
        # pose_xyc: (B,T,K,V,3) or (B,T,K,V,2)
        emb, person_mask = self.mlp(x) # B,T,K,out_dim
        
        scene = self.pool(emb, mask=person_mask)              # (B,T,C)
        
        count = person_mask.sum(dim=2).float()                # (B,T)
        count_norm = count / person_mask.size(2)              # (B,T) for K fixed
        #print(count_norm.unsqueeze(-1).shape)
        scene_count = torch.cat([scene, count_norm.unsqueeze(-1)], dim=-1)  # (B,T,C+1)
        #print(scene_count.shape)
        #B, T, K, E = x.shape

        '''scene_per = (
            scene.permute(0, 2, 1)      # (B, K, E, T)
             .contiguous()
             #.view(B * K, E, T)        # (B*K, E, T)
        )'''
        tcn_in = scene_count.transpose(1,2).contiguous()
        feat = self.tcn(tcn_in)             # (B, hidden_dim, T)
        #logits = self.head(feat)       # (B, num_classes, T)
        #logits = logits.transpose(1, 2)  # (B, T, num_classes)
        #last_feat = feat[:, :, -1]         # (B, hidden_dim)
        mid = feat.size(-1) // 2
        mid_feat = feat[:, :, mid]              # (B, hidden)
        logits = self.fc(mid_feat).squeeze(-1)  # (B,)
        #logits = self.fc(feat)#.squeeze(-1)  # (B,)  raw score
        return logits

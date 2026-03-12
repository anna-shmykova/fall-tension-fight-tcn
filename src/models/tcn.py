import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.encoders import BoneMLPEncoder
from src.models.pooling import PersonPooling, AttentionReadout
from src.models.graph import InterpersonalGraph

class CausalConv1d(nn.Module):
    """Conv1d with left-only padding so output[t] depends only on input[:t]."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, bias=True):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.left_pad = (self.kernel_size - 1) * self.dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,   # IMPORTANT: no symmetric padding
            bias=bias,
        )

    def forward(self, x):
        # x: (B, C, T)
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, causal=False, norm="group"):
        super().__init__()
        self.causal = bool(causal)

        if self.causal:
            self.conv = CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        else:
            padding = (kernel_size - 1) * dilation // 2
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            )

        # NOTE: BatchNorm1d mixes statistics across time during training.
        # For strict causality (no training-time leakage), prefer GroupNorm.
        if norm == "batch":
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class EventTCN(nn.Module):
    def __init__(
                    self,
                    input_dim=97,
                    num_classes=1,
                    hidden_dim=64,
                    num_layers=3,
                    kernel_size=3,
                    mlp_out_dim=32,
                    pool_mode="attn",
                    causal=True,
                    norm="group",
                ):
        super().__init__()
        self.mlp = BoneMLPEncoder(out_dim=mlp_out_dim)
        self.graph = InterpersonalGraph(
                                            dim=mlp_out_dim,
                                            k_nn=2,
                                            radius=2.5,
                                            hidden=hidden_dim,
                                            dropout=0.1,
                                        )
        if not pool_mode == "attn":
            self.pool = PersonPooling(
                                    mode=pool_mode,
                                    emb_dim=mlp_out_dim,
                                    attn_hidden_dim=mlp_out_dim,
                                    dropout=0.1,
                                )
        else: 
            self.pool = AttentionReadout(
                                        emb_dim=mlp_out_dim,
                                        hidden_dim=32,
                                        dropout=0.1
                                      )
        pool_mult = {
                        "mean": 1,
                        "max": 1,
                        "mean_max": 2,
                        "mean_max_std": 3,
                        "attn": 1,
                    }
        in_ch = mlp_out_dim * pool_mult[pool_mode] + 1

        layers = []
        dilation = 1
        for _ in range(num_layers):
            layers.append(
                TemporalConvBlock(
                    in_channels=in_ch,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    causal=causal,
                    norm=norm,
                )
            )
            in_ch = hidden_dim
            dilation *= 2

        self.tcn = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, T, C)  (your BoneMLPEncoder already handles your format)
        emb, person_mask, person_bboxes = self.mlp(x)          # (B,T,N,E), (B,T,N), (B,T,N,4)
        emb = self.graph(emb, person_bboxes, person_mask)      # (B,T,N,E)
        scene,_ = self.pool(emb, mask=person_mask)               # (B,T,E_scene)
        #print(scene.shape)
        count = person_mask.sum(dim=2).float()              # (B,T)
        count_norm = count / person_mask.size(2)            # (B,T)
        scene_count = torch.cat([scene, count_norm.unsqueeze(-1)], dim=-1)  # (B,T,E_scene+1)

        feat = self.tcn(scene_count.transpose(1, 2))        # (B, hidden, T)
        logits = self.head(feat)                            # (B,1,T)

        return logits.squeeze(1)                            # (B,T)
"""Temporal models used by training and inference.

`EventTCN` is the main model:
1. encode each person in each frame,
2. optionally refine people with an interpersonal graph,
3. pool people into one scene embedding per frame,
4. run a temporal convolution network over the sequence.

`MotionTCN` is the simpler baseline that works directly on motion features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.encoders import BoneMLPEncoder, IntrapersonGraphEncoder, infer_max_persons_from_input_dim
from src.models.pooling import PersonPooling
from src.models.graph import InterpersonalGraph


VALID_POOL_MODES = {"mean", "max", "mean_max", "mean_max_std", "attn"}
DEFAULT_NON_ATTN_POOL_MODE = "mean_max_std"
VALID_ENCODER_TYPES = {"mlp", "intraperson_graph"}
DEFAULT_ENCODER_TYPE = "intraperson_graph"


def resolve_pool_mode(pool_mode="attn", use_attention_readout=None):
    """Resolve the effective pooling mode from legacy and new config flags."""
    pool_mode = str(pool_mode)

    if use_attention_readout is True:
        effective_pool_mode = "attn"
    elif use_attention_readout is False:
        effective_pool_mode = DEFAULT_NON_ATTN_POOL_MODE if pool_mode == "attn" else pool_mode
    else:
        effective_pool_mode = pool_mode

    if effective_pool_mode not in VALID_POOL_MODES:
        raise ValueError(f"Unsupported pool_mode: {effective_pool_mode}")
    return effective_pool_mode


def state_dict_has_prefix(state_dict, prefix):
    """Return True when any checkpoint key starts with `prefix`."""
    return any(key.startswith(prefix) for key in state_dict)


def resolve_encoder_type(encoder_type=None):
    """Normalize the requested person encoder type."""
    if encoder_type is None:
        return DEFAULT_ENCODER_TYPE

    resolved = str(encoder_type).lower()
    if resolved not in VALID_ENCODER_TYPES:
        raise ValueError(f"Unsupported encoder_type: {resolved}")
    return resolved


def infer_encoder_type(state_dict=None, configured=None):
    """Infer which encoder produced a checkpoint.

    Old checkpoints store the person encoder under `mlp.*`.
    New checkpoints store it under `encoder.*`, with graph-specific submodules.
    """
    if configured is not None:
        return resolve_encoder_type(configured)

    state_dict = state_dict or {}
    if (
        state_dict_has_prefix(state_dict, "encoder.node_proj.")
        or state_dict_has_prefix(state_dict, "encoder.layers.")
        or state_dict_has_prefix(state_dict, "encoder.joint_embed.")
    ):
        return "intraperson_graph"
    if state_dict_has_prefix(state_dict, "mlp.") or state_dict_has_prefix(state_dict, "encoder.mlp."):
        return "mlp"
    if state_dict_has_prefix(state_dict, "encoder."):
        return DEFAULT_ENCODER_TYPE
    return DEFAULT_ENCODER_TYPE


def normalize_event_state_dict(state_dict):
    """Rewrite legacy checkpoint keys into the current naming scheme.

    Currently handled:
    - `pool.score.*` -> `pool.attn.score.*`
    - `mlp.*` -> `encoder.*`
    """
    remapped = dict(state_dict)

    if state_dict_has_prefix(remapped, "pool.score.") and not state_dict_has_prefix(remapped, "pool.attn.score."):
        updated = {}
        for key, value in remapped.items():
            if key.startswith("pool.score."):
                updated[key.replace("pool.score.", "pool.attn.score.", 1)] = value
            else:
                updated[key] = value
        remapped = updated

    if state_dict_has_prefix(remapped, "mlp.") and not state_dict_has_prefix(remapped, "encoder."):
        updated = {}
        for key, value in remapped.items():
            if key.startswith("mlp."):
                updated[key.replace("mlp.", "encoder.", 1)] = value
            else:
                updated[key] = value
        remapped = updated

    return remapped


def resolve_dilations(num_layers=3, dilations=None):
    """Return the dilation schedule for the temporal convolution stack."""
    if dilations is None:
        resolved = []
        dilation = 1
        for _ in range(int(num_layers)):
            resolved.append(dilation)
            dilation *= 2
        return resolved

    resolved = [int(dilation) for dilation in dilations]
    if not resolved:
        raise ValueError("dilations must contain at least one positive integer")
    if any(dilation <= 0 for dilation in resolved):
        raise ValueError(f"dilations must be positive, got {resolved}")

    expected_layers = int(num_layers)
    if expected_layers != len(resolved):
        raise ValueError(f"num_layers={expected_layers} does not match len(dilations)={len(resolved)}")
    return resolved

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
    """One temporal conv block: conv -> norm -> ReLU."""

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
        """Apply one temporal block to `[B, C, T]` features."""
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class EventTCN(nn.Module):
    """Main event model combining pose encoding, pooling, and temporal context.

    Tensor flow:
        raw frame vector
            -> per-person encoder
            -> optional interpersonal graph
            -> person pooling
            -> temporal convolution stack
            -> per-time-step event logits
    """

    def __init__(
                    self,
                    input_dim=97,
                    num_classes=1,
                    hidden_dim=64,
                    num_layers=3,
                    dilations=None,
                    kernel_size=3,
                    mlp_out_dim=32,
                    person_emb_dim=None,
                    pool_mode="attn",
                    use_attention_readout=None,
                    encoder_type=DEFAULT_ENCODER_TYPE,
                    encoder_hidden_dim=128,
                    encoder_graph_dim=None,
                    encoder_num_layers=2,
                    use_graph=True,
                    causal=True,
                    norm="group",
                    dropout=0.1,
                    motion_dim=0,
                    motion_proj_dim=None,
                    tcn_input_mode="pooled_count",
                    use_person_count=True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.motion_dim = int(motion_dim)
        self.tcn_input_mode = str(tcn_input_mode)
        self.use_person_count = bool(use_person_count)
        self.use_graph = bool(use_graph)
        self.encoder_type = resolve_encoder_type(encoder_type)
        self.person_emb_dim = int(person_emb_dim if person_emb_dim is not None else mlp_out_dim)
        self.pool_mode = resolve_pool_mode(pool_mode=pool_mode, use_attention_readout=use_attention_readout)
        # The flat input width already encodes how many person slots exist.
        max_persons = infer_max_persons_from_input_dim(input_dim=self.input_dim, motion_dim=self.motion_dim)

        if self.encoder_type == "mlp":
            # Legacy handcrafted per-person encoder.
            self.encoder = BoneMLPEncoder(
                K=max_persons,
                out_dim=self.person_emb_dim,
                hidden=int(encoder_hidden_dim),
            )
        else:
            # New default: graph over joints inside each person skeleton.
            self.encoder = IntrapersonGraphEncoder(
                K=max_persons,
                out_dim=self.person_emb_dim,
                hidden=int(encoder_hidden_dim),
                graph_dim=int(encoder_graph_dim) if encoder_graph_dim is not None else None,
                num_layers=int(encoder_num_layers),
                dropout=dropout,
            )
        self.graph = (
            InterpersonalGraph(
                                dim=self.person_emb_dim,
                                k_nn=2,
                                radius=2.5,
                                hidden=hidden_dim,
                                dropout=dropout,
                            )
            if self.use_graph
            else None
        )
        # Attention pooling uses a small hidden layer; non-attention pooling does
        # not need one because it is purely statistical.
        attn_hidden_dim = 32 if self.pool_mode == "attn" else self.person_emb_dim
        self.pool = PersonPooling(
                                mode=self.pool_mode,
                                emb_dim=self.person_emb_dim,
                                attn_hidden_dim=attn_hidden_dim,
                                dropout=dropout,
                            )
        pool_mult = {
                        "mean": 1,
                        "max": 1,
                        "mean_max": 2,
                        "mean_max_std": 3,
                        "attn": 1,
                    }

        use_motion = self.tcn_input_mode == "pooled_count_motion"
        if self.tcn_input_mode not in {"pooled_count", "pooled_count_motion"}:
            raise ValueError(f"Unsupported tcn_input_mode: {self.tcn_input_mode}")
        if use_motion and self.motion_dim <= 0:
            raise ValueError("tcn_input_mode='pooled_count_motion' requires motion_dim > 0")

        if self.motion_dim > 0:
            if motion_proj_dim is None:
                self.motion_proj = nn.Identity()
                motion_out_dim = self.motion_dim
            else:
                # Optional projection for auxiliary motion features before
                # concatenating them with pooled scene features.
                motion_proj_dim = int(motion_proj_dim)
                self.motion_proj = nn.Linear(self.motion_dim, motion_proj_dim)
                motion_out_dim = motion_proj_dim
        else:
            self.motion_proj = None
            motion_out_dim = 0

        count_dim = 1 if self.use_person_count else 0
        # TCN input width depends on how we pool people and whether we append
        # person count and/or motion features.
        in_ch = self.person_emb_dim * pool_mult[self.pool_mode] + count_dim + (motion_out_dim if use_motion else 0)

        layers = []
        for dilation in resolve_dilations(num_layers=num_layers, dilations=dilations):
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

        self.tcn = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x):
        """Return per-time-step logits with shape `[B, T]`."""
        motion = None
        pose_x = x
        if self.motion_dim > 0:
            if x.size(-1) <= self.motion_dim:
                raise ValueError("Input does not contain the expected static pose channels")
            # Split the raw feature vector back into pose features and optional
            # motion features appended by the dataset builder.
            pose_x = x[..., :-self.motion_dim]
            motion = x[..., -self.motion_dim:]

        # Encode each person in each frame.
        emb, person_mask, person_bboxes = self.encoder(pose_x)  # (B,T,N,E), (B,T,N), (B,T,N,4)
        if self.graph is not None:
            # Optionally let nearby people exchange information inside a frame.
            emb = self.graph(emb, person_bboxes, person_mask)   # (B,T,N,E)

        # Collapse the person dimension to get one frame-level scene embedding.
        scene, _ = self.pool(emb, mask=person_mask, return_attn=True)
        tcn_inputs = [scene]
        if self.use_person_count:
            count = person_mask.sum(dim=2).float()                  # (B,T)
            count_norm = count / person_mask.size(2)                # (B,T)
            tcn_inputs.append(count_norm.unsqueeze(-1))

        if self.tcn_input_mode == "pooled_count_motion":
            if motion is None or self.motion_proj is None:
                raise ValueError("Motion features are enabled in the model but missing at runtime")
            tcn_inputs.append(self.motion_proj(motion))

        # Temporal conv expects channels-first `[B, C, T]`.
        scene_count = torch.cat(tcn_inputs, dim=-1)             # (B,T,E_scene+1[+motion])

        feat = self.tcn(scene_count.transpose(1, 2))        # (B, hidden, T)
        logits = self.head(feat)                            # (B,1,T)

        return logits.squeeze(1)                            # (B,T)


class MotionTCN(nn.Module):
    """Simpler baseline that runs a TCN directly on motion features only."""

    def __init__(
                    self,
                    input_dim=25,
                    num_classes=1,
                    hidden_dim=64,
                    num_layers=4,
                    dilations=None,
                    kernel_size=3,
                    causal=True,
                    norm="group",
                    input_proj_dim=0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)

        proj_dim = int(input_proj_dim or 0)
        if proj_dim > 0:
            self.input_proj = nn.Linear(self.input_dim, proj_dim)
            in_ch = proj_dim
        else:
            self.input_proj = None
            in_ch = self.input_dim

        layers = []
        for dilation in resolve_dilations(num_layers=num_layers, dilations=dilations):
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

        self.tcn = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x):
        """Return per-time-step logits from motion-only inputs `[B, T, C]`."""
        if self.input_proj is not None:
            x = self.input_proj(x)

        feat = self.tcn(x.transpose(1, 2))
        logits = self.head(feat)
        return logits.squeeze(1)

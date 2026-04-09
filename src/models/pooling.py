"""Pooling layers that collapse person embeddings into one scene embedding."""

import torch
import torch.nn as nn

def masked_mean(x, mask, dim):
    """Mean over `dim` while ignoring invalid entries indicated by `mask`."""
    w = mask.float().unsqueeze(-1)
    s = (x * w).sum(dim=dim)
    d = w.sum(dim=dim).clamp_min(1.0)
    return s / d

def masked_std(x, mask, dim):
    """Standard deviation over valid entries only."""
    mu = masked_mean(x, mask, dim=dim)
    w = mask.float().unsqueeze(-1)
    var = (w * (x - mu.unsqueeze(dim))**2).sum(dim=dim) / w.sum(dim=dim).clamp_min(1.0)
    return torch.sqrt(var + 1e-6)

def masked_max(x, mask, dim):
    """Max over valid entries only, returning 0 when everything is invalid."""
    neg = -1e9
    x2 = x.masked_fill(~mask.bool().unsqueeze(-1), neg)
    mx = x2.max(dim=dim).values
    all_invalid = (mask.sum(dim=dim) == 0)
    if all_invalid.any():
        mx = mx.masked_fill(all_invalid.unsqueeze(-1), 0.0)
    return mx


class AttentionReadout(nn.Module):
    """
    Attention-based pooling over people.

    The readout learns a scalar score for each person in the frame and then
    uses a softmax over people to compute a weighted average scene embedding.
    """
    def __init__(self, emb_dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = emb_dim if hidden_dim is None else hidden_dim

        self.score = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, emb: torch.Tensor, mask: torch.Tensor | None = None):
        """Pool `[B, T, K, E]` person embeddings into `[B, T, E]` scene vectors."""
        B, T, K, E = emb.shape
        if mask is None:
            mask = torch.ones((B, T, K), device=emb.device, dtype=torch.bool)
        else:
            mask = mask.bool()

        # (B,T,K,1) -> (B,T,K)
        logits = self.score(emb).squeeze(-1)

        # Invalid padded people must not get attention mass.
        neg = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~mask, neg)

        # softmax over people
        alpha = torch.softmax(logits, dim=2)  # (B,T,K)

        # If a frame has no valid people, force the attention vector to zeros.
        all_invalid = (mask.sum(dim=2) == 0)  # (B,T)
        if all_invalid.any():
            alpha = alpha.masked_fill(all_invalid.unsqueeze(-1), 0.0)

        out = (emb * alpha.unsqueeze(-1)).sum(dim=2)  # (B,T,E)
        return out, alpha


class PersonPooling(nn.Module):
    """
    Reduce per-person embeddings into one scene-level embedding per frame.

    Supported modes:
        mean          simple average over people
        max           max over people
        mean_max      concatenate mean and max
        mean_max_std  concatenate mean, max, and std
        attn          learned attention over people
    """
    def __init__(self, mode="mean_max_std", emb_dim: int | None = None, attn_hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.mode = mode

        if self.mode == "attn":
            if emb_dim is None:
                raise ValueError("PersonPooling(mode='attn') requires emb_dim")
            self.attn = AttentionReadout(
                emb_dim=emb_dim,
                hidden_dim=attn_hidden_dim,
                dropout=dropout,
            )

    def forward(self, emb, mask=None, return_attn: bool = False):
        """Pool person embeddings for each frame and optionally return attention."""
        B, T, K, E = emb.shape
        if mask is None:
            mask = torch.ones((B, T, K), device=emb.device, dtype=torch.bool)

        if self.mode == "mean":
            out = masked_mean(emb, mask, dim=2)
            return (out, None) if return_attn else out

        if self.mode == "max":
            out = masked_max(emb, mask, dim=2)
            return (out, None) if return_attn else out

        if self.mode == "mean_max":
            mu = masked_mean(emb, mask, dim=2)
            mx = masked_max(emb, mask, dim=2)
            out = torch.cat([mu, mx], dim=-1)
            return (out, None) if return_attn else out

        if self.mode == "mean_max_std":
            mu = masked_mean(emb, mask, dim=2)
            mx = masked_max(emb, mask, dim=2)
            sd = masked_std(emb, mask, dim=2)
            out = torch.cat([mu, mx, sd], dim=-1)
            return (out, None) if return_attn else out

        if self.mode == "attn":
            out, alpha = self.attn(emb, mask)
            return (out, alpha) if return_attn else out

        raise ValueError(self.mode)

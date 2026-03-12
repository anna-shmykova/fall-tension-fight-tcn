import torch
import torch.nn as nn

def masked_mean(x, mask, dim):
    w = mask.float().unsqueeze(-1)
    s = (x * w).sum(dim=dim)
    d = w.sum(dim=dim).clamp_min(1.0)
    return s / d

def masked_std(x, mask, dim):
    mu = masked_mean(x, mask, dim=dim)
    w = mask.float().unsqueeze(-1)
    var = (w * (x - mu.unsqueeze(dim))**2).sum(dim=dim) / w.sum(dim=dim).clamp_min(1.0)
    return torch.sqrt(var + 1e-6)

def masked_max(x, mask, dim):
    neg = -1e9
    x2 = x.masked_fill(~mask.bool().unsqueeze(-1), neg)
    mx = x2.max(dim=dim).values
    all_invalid = (mask.sum(dim=dim) == 0)
    if all_invalid.any():
        mx = mx.masked_fill(all_invalid.unsqueeze(-1), 0.0)
    return mx


class AttentionReadout(nn.Module):
    """
    emb:  (B,T,K,E)
    mask: (B,T,K) boolean
    out:  (B,T,E)
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
        B, T, K, E = emb.shape
        if mask is None:
            mask = torch.ones((B, T, K), device=emb.device, dtype=torch.bool)
        else:
            mask = mask.bool()

        # (B,T,K,1) -> (B,T,K)
        logits = self.score(emb).squeeze(-1)

        # mask invalid persons before softmax
        neg = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~mask, neg)

        # softmax over people
        alpha = torch.softmax(logits, dim=2)  # (B,T,K)

        # if a frame has no valid persons, softmax would be undefined numerically
        all_invalid = (mask.sum(dim=2) == 0)  # (B,T)
        if all_invalid.any():
            alpha = alpha.masked_fill(all_invalid.unsqueeze(-1), 0.0)

        out = (emb * alpha.unsqueeze(-1)).sum(dim=2)  # (B,T,E)
        return out, alpha


class PersonPooling(nn.Module):
    """
    emb: (B,T,K,E)
    mask: (B,T,K) boolean, True for valid persons
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
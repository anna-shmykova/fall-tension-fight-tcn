import torch
import torch.nn as nn

def masked_mean(x, mask, dim):
    # x: (..., K, E), mask: (..., K) in {0,1}
    w = mask.float().unsqueeze(-1)                # (..., K, 1)
    s = (x * w).sum(dim=dim)                      # (..., E)
    d = w.sum(dim=dim).clamp_min(1.0)             # (..., 1)
    return s / d

def masked_std(x, mask, dim):
    mu = masked_mean(x, mask, dim=dim)            # (..., E)
    w = mask.float().unsqueeze(-1)
    var = (w * (x - mu.unsqueeze(dim))**2).sum(dim=dim) / w.sum(dim=dim).clamp_min(1.0)
    return torch.sqrt(var + 1e-6)

def masked_max(x, mask, dim):
    # set masked-out to very negative so they never win max
    #neg_inf = torch.finfo(x.dtype).min
    neg = -1e9
    x2 = x.masked_fill(~mask.bool().unsqueeze(-1), neg)
    mx = x2.max(dim=dim).values  # (..., E)

    # if all invalid, set output to 0
    all_invalid = (mask.sum(dim=dim) == 0)  # (...) boolean
    if all_invalid.any():
        mx = mx.masked_fill(all_invalid.unsqueeze(-1), 0.0)
    return mx

class PersonPooling(nn.Module):
    """
    emb: (B,T,K,E)
    mask: (B,T,K) boolean, True for valid persons
    """
    def __init__(self, mode="mean_max_std"):
        super().__init__()
        self.mode = mode

    def forward(self, emb, mask=None):
        B, T, K, E = emb.shape
        if mask is None:
            # assume all K are valid
            mask = torch.ones((B, T, K), device=emb.device, dtype=torch.bool)

        if self.mode == "mean":
            return masked_mean(emb, mask, dim=2)                         # (B,T,E)
        if self.mode == "max":
            return masked_max(emb, mask, dim=2)                          # (B,T,E)
        if self.mode == "mean_max":
            mu = masked_mean(emb, mask, dim=2)
            mx = masked_max(emb, mask, dim=2)
            return torch.cat([mu, mx], dim=-1)                           # (B,T,2E)
        if self.mode == "mean_max_std":
            mu = masked_mean(emb, mask, dim=2)
            mx = masked_max(emb, mask, dim=2)
            sd = masked_std(emb, mask, dim=2)
            
            return torch.cat([mu, mx, sd], dim=-1)                       # (B,T,3E)

        raise ValueError(self.mode)
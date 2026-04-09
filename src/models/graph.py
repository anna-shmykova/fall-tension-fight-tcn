"""Graph layers that model interactions between different people in a frame."""

import torch
import torch.nn as nn


class InterpersonalGraph(nn.Module):
    """
    Message-passing graph over people within the same frame.

    Nodes:
        people

    Edges:
        dynamic spatial neighbors selected with KNN over bbox centers

    Input:
        emb:         [B, T, N, D] per-person embeddings from the encoder
        bboxes:      [B, T, N, 4] person boxes in (cx, cy, w, h)
        person_mask: [B, T, N] valid-person mask

    Output:
        refined person embeddings with the same shape [B, T, N, D]
    """
    def __init__(self, dim, k_nn=4, radius=2.5, hidden=64, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.k_nn = k_nn
        self.radius = radius

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * dim + 3, hidden),   # x_i, x_j, [dx,dy,dist]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(2 * dim, hidden),       # x_i + aggregated messages
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, emb, bboxes, person_mask):
        """Exchange information between nearby people independently per frame."""
        B, T, N, D = emb.shape
        assert D == self.dim

        # Merge batch and time so each frame becomes one small person graph.
        x = emb.reshape(B * T, N, D)                 # [BT, N, D]
        boxes = bboxes.reshape(B * T, N, 4)          # [BT, N, 4]
        mask = person_mask.reshape(B * T, N).bool()  # [BT, N]

        cx = boxes[..., 0]
        cy = boxes[..., 1]
        h  = boxes[..., 3].clamp_min(1e-6)

        dx = cx[:, :, None] - cx[:, None, :]         # [BT, N, N]
        dy = cy[:, :, None] - cy[:, None, :]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6)

        # Normalize geometry by the source person's height so distances are
        # roughly scale-invariant across near/far people.
        scale = h[:, :, None]
        dx_norm = dx / scale
        dy_norm = dy / scale
        dist_norm = dist / scale

        # Only valid, non-self person pairs may exchange messages.
        pair_valid = mask[:, :, None] & mask[:, None, :]   # [BT, N, N]
        eye = torch.eye(N, device=emb.device, dtype=torch.bool).unsqueeze(0)
        pair_valid = pair_valid & (~eye)

        # Keep only the K nearest people for each source node.
        big = torch.full_like(dist_norm, 1e6)
        dist_for_knn = torch.where(pair_valid, dist_norm, big)

        K = min(self.k_nn, max(N - 1, 1))
        knn_dist, knn_idx = torch.topk(dist_for_knn, k=K, dim=-1, largest=False)  # [BT,N,K]

        nbr_valid = torch.gather(pair_valid, dim=2, index=knn_idx)  # [BT,N,K]
        if self.radius is not None:
            # Optional radius gate: nearby neighbors only.
            nbr_valid = nbr_valid & (knn_dist < self.radius)

        # Gather the source/neighbor node states for each selected edge.
        x_i = x[:, :, None, :].expand(B * T, N, K, D)

        x_expand = x[:, None, :, :].expand(B * T, N, N, D)
        idx_feat = knn_idx[..., None].expand(B * T, N, K, D)
        x_j = torch.gather(x_expand, dim=2, index=idx_feat)

        # Gather relative geometry for the selected person-person edges.
        edge_all = torch.stack([dx_norm, dy_norm, dist_norm], dim=-1)   # [BT,N,N,3]
        idx_edge = knn_idx[..., None].expand(B * T, N, K, 3)
        e_ij = torch.gather(edge_all, dim=2, index=idx_edge)            # [BT,N,K,3]

        # Build messages from source state, neighbor state, and relative pose.
        msg_in = torch.cat([x_i, x_j, e_ij], dim=-1)    # [BT,N,K,2D+3]
        msg = self.edge_mlp(msg_in)                      # [BT,N,K,D]
        msg = msg * nbr_valid[..., None].float()

        denom = nbr_valid.float().sum(dim=2, keepdim=True).clamp_min(1.0)
        agg = msg.sum(dim=2) / denom                     # [BT,N,D]

        # Update each person's embedding with the aggregated neighbor message.
        upd_in = torch.cat([x, agg], dim=-1)
        delta = self.node_mlp(upd_in)                    # [BT,N,D]

        # If a person has no valid neighbors, keep its original embedding.
        has_nbr = nbr_valid.any(dim=2, keepdim=True)     # [BT,N,1]
        delta = delta * has_nbr.float()

        out = self.norm(x + delta)
        out = out * mask[..., None].float()

        return out.reshape(B, T, N, D)

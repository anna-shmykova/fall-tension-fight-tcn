"""Pose encoders that turn a frame's person detections into person embeddings.

The dataset stores each frame as one flat vector:
`[person_1_features, ..., person_K_features, normalized_person_count]`.

Each person block contains:
`[cx, cy, w, h, joint_0(x,y,conf), ..., joint_11(x,y,conf)]`.

This file provides two encoder families that consume that layout and return the
same interface:
`(person_embeddings, person_mask, person_bboxes)`.
"""

import torch
import torch.nn as nn


DEFAULT_NUM_JOINTS = 12
PERSON_BBOX_DIM = 4
PERSON_COUNT_DIM = 1
DEFAULT_SKELETON_EDGES = [
    (0, 1),
    (0, 2),
    (2, 4),
    (1, 3),
    (3, 5),
    (0, 6),
    (1, 7),
    (6, 7),
    (6, 8),
    (8, 10),
    (7, 9),
    (9, 11),
]


def person_feature_dim(num_joints=DEFAULT_NUM_JOINTS):
    """Return the width of one flattened person block inside the input tensor."""
    return PERSON_BBOX_DIM + 3 * int(num_joints)


def infer_max_persons_from_input_dim(input_dim, motion_dim=0, num_joints=DEFAULT_NUM_JOINTS):
    """Infer how many people were packed into the flat frame vector.

    The static pose part of the vector is:
    `K * person_feature_dim + PERSON_COUNT_DIM`.
    """
    static_dim = int(input_dim) - int(motion_dim)
    if static_dim <= PERSON_COUNT_DIM:
        raise ValueError(f"input_dim={input_dim} motion_dim={motion_dim} leaves no pose features")

    per_person_dim = person_feature_dim(num_joints=num_joints)
    remainder = static_dim - PERSON_COUNT_DIM
    if remainder % per_person_dim != 0:
        raise ValueError(
            f"Could not infer max persons from input_dim={input_dim}, motion_dim={motion_dim}: "
            f"static pose width {static_dim} is not 1 + K*{per_person_dim}"
        )
    return remainder // per_person_dim


def unpack_pose_tensor(pose_xyc, num_persons, num_joints):
    """Split the flat pose tensor into boxes and per-joint tensors.

    Input:
        pose_xyc: [B, T, K * (4 + 3V) + 1]

    Returns:
        bboxes:   [B, T, K, 4]
        keypoints:[B, T, K, V, 3]
        xy:       [B, T, K, V, 2]
        conf:     [B, T, K, V]
    """
    per_person_dim = person_feature_dim(num_joints=num_joints)
    expected_dim = int(num_persons) * per_person_dim + PERSON_COUNT_DIM
    if pose_xyc.size(-1) != expected_dim:
        raise ValueError(
            f"Expected pose tensor width {expected_dim} for K={num_persons}, V={num_joints}, "
            f"got {pose_xyc.size(-1)}"
        )

    people = pose_xyc[..., :-PERSON_COUNT_DIM].contiguous().view(*pose_xyc.shape[:-1], int(num_persons), per_person_dim)
    bboxes = people[..., :PERSON_BBOX_DIM]
    keypoints = people[..., PERSON_BBOX_DIM:].contiguous().view(*people.shape[:-1], int(num_joints), 3)
    xy = keypoints[..., :2]
    conf = keypoints[..., 2]
    return bboxes, keypoints, xy, conf


def compute_pose_masks(xy, conf, bboxes, conf_thr=0.2, min_visible_joints=3):
    """Build validity masks for joints and people.

    A person is considered valid when:
    1. its bbox exists, and
    2. it has at least `min_visible_joints` visible joints.
    """
    if conf is not None:
        joint_mask = conf > float(conf_thr)
    else:
        joint_mask = xy.abs().sum(dim=-1) > 0

    w = bboxes[..., 2]
    h = bboxes[..., 3]
    person_mask = (w > 0) & (h > 0)
    person_mask = person_mask & (joint_mask.sum(dim=-1) >= int(min_visible_joints))
    joint_mask = joint_mask & person_mask.unsqueeze(-1)
    return joint_mask, person_mask


class BoneMLPEncoder(nn.Module):
    """Legacy per-person encoder based on handcrafted bone features + an MLP.

    This is the old path that existed before the intraperson graph encoder.
    It summarizes each person independently by:
    1. extracting bone vectors from the 12-joint skeleton,
    2. adding a pelvis-like anchor and validity flags,
    3. projecting the handcrafted feature vector with an MLP.
    """

    def __init__(
        self,
        K=25,
        V=DEFAULT_NUM_JOINTS,
        edges=DEFAULT_SKELETON_EDGES,
        out_dim=32,
        hidden=128,
        use_conf=True,
        add_bbox=True,
        add_masks=True,
        conf_thr=0.2,
        min_visible_joints=3,
    ):
        super().__init__()
        self.K = int(K)
        self.V = int(V)
        self.edges = list(edges)
        self.use_conf = bool(use_conf)
        self.add_bbox = bool(add_bbox)
        self.add_masks = bool(add_masks)
        self.conf_thr = float(conf_thr)
        self.min_visible_joints = int(min_visible_joints)

        num_bones = len(self.edges)
        in_dim = 2 + (2 * num_bones)
        if self.add_bbox:
            in_dim += PERSON_BBOX_DIM
        if self.add_masks:
            in_dim += num_bones + 2

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, pose_xyc, bbox_cxcywh=None):
        """Encode each person independently into a fixed-width embedding."""
        bboxes, _keypoints, xy, conf = unpack_pose_tensor(pose_xyc, num_persons=self.K, num_joints=self.V)
        joint_mask, person_mask = compute_pose_masks(
            xy,
            conf if self.use_conf else None,
            bboxes,
            conf_thr=self.conf_thr,
            min_visible_joints=self.min_visible_joints,
        )

        bones = []
        bone_masks = []
        for u, v in self.edges:
            # A bone is valid only if both endpoint joints are visible.
            bone_valid = joint_mask[..., u] & joint_mask[..., v]
            delta = (xy[..., v, :] - xy[..., u, :]) * bone_valid.unsqueeze(-1).float()
            bones.append(delta)
            bone_masks.append(bone_valid)

        bones = torch.stack(bones, dim=-2)
        bone_mask = torch.stack(bone_masks, dim=-1)

        both_hips = joint_mask[..., 6] & joint_mask[..., 7]
        midhip = 0.5 * (xy[..., 6, :] + xy[..., 7, :])
        # Use the midpoint between hips as a coarse body anchor when available.
        anchor_xy = torch.where(both_hips[..., None], midhip, torch.zeros_like(midhip))
        anchor_valid = both_hips

        # Concatenate all handcrafted features into one per-person vector.
        feat = [bones.reshape(*bones.shape[:-2], -1), anchor_xy]
        if self.add_masks:
            feat.append(bone_mask.float())
            feat.append(person_mask.float().unsqueeze(-1))
            feat.append(anchor_valid.float().unsqueeze(-1))
        if self.add_bbox:
            feat.append(bboxes)

        x = torch.cat(feat, dim=-1)
        emb = self.mlp(x)
        return emb, person_mask, bboxes


class IntrapersonGraphLayer(nn.Module):
    """One message-passing step over the joints of a single person.

    Nodes are joints.
    Edges are the fixed human skeleton edges from `DEFAULT_SKELETON_EDGES`.

    For each joint i, the layer:
    1. gathers messages from neighboring joints j,
    2. conditions each message on the source/target states and relative offset,
    3. averages valid incoming messages,
    4. updates the joint state with a residual MLP.
    """

    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * dim + 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, xy, joint_mask, edge_index):
        """Run one graph layer on flattened `[B*T*K, V, D]` joint tensors."""
        if edge_index.numel() == 0:
            return h * joint_mask.unsqueeze(-1).float()

        dst_idx = edge_index[:, 0]
        src_idx = edge_index[:, 1]

        # Only gather real skeleton edges instead of materializing all VxV
        # joint pairs and masking them afterwards.
        target = h.index_select(1, dst_idx)
        source = h.index_select(1, src_idx)
        # Relative geometry is part of the edge feature, so messages know
        # whether a neighbor is above/below/left/right of the current joint.
        rel = xy.index_select(1, src_idx) - xy.index_select(1, dst_idx)

        edge_valid = joint_mask.index_select(1, dst_idx) & joint_mask.index_select(1, src_idx)
        msg_in = torch.cat([target, source, rel], dim=-1)
        msg = self.edge_mlp(msg_in) * edge_valid.unsqueeze(-1).float()

        # Average only over valid skeleton neighbors.
        agg = torch.zeros_like(h)
        agg.index_add_(1, dst_idx, msg)
        denom = h.new_zeros(h.size(0), h.size(1), 1)
        denom.index_add_(1, dst_idx, edge_valid.unsqueeze(-1).float())
        agg = agg / denom.clamp_min(1.0)

        delta = self.node_mlp(torch.cat([h, agg], dim=-1))
        has_neighbors = denom > 0
        out = self.norm(h + delta * has_neighbors.float())
        return out * joint_mask.unsqueeze(-1).float()


class IntrapersonGraphEncoder(nn.Module):
    """Encode one person's skeleton with graph message passing over joints.

    This is the new default encoder.

    High-level idea:
    1. unpack the flat pose vector into bbox + joints,
    2. create one node per joint,
    3. run several message-passing layers over the fixed skeleton graph,
    4. pool the joint states back into one embedding for the whole person.
    """

    def __init__(
        self,
        K=25,
        V=DEFAULT_NUM_JOINTS,
        edges=DEFAULT_SKELETON_EDGES,
        out_dim=32,
        hidden=128,
        graph_dim=None,
        num_layers=2,
        dropout=0.1,
        use_conf=True,
        add_bbox=True,
        conf_thr=0.2,
        min_visible_joints=3,
    ):
        super().__init__()
        self.K = int(K)
        self.V = int(V)
        self.edges = list(edges)
        self.out_dim = int(out_dim)
        self.hidden = int(hidden)
        self.graph_dim = int(graph_dim if graph_dim is not None else max(out_dim, 64))
        self.num_layers = int(num_layers)
        self.use_conf = bool(use_conf)
        self.add_bbox = bool(add_bbox)
        self.conf_thr = float(conf_thr)
        self.min_visible_joints = int(min_visible_joints)

        # Joint identity embedding lets the model distinguish "left wrist"
        # from "right ankle" even if their coordinates look similar.
        joint_id_dim = max(4, min(16, self.graph_dim // 4))
        node_in_dim = 2 + 2 + 1 + joint_id_dim + (1 if self.use_conf else 0)

        self.joint_embed = nn.Embedding(self.V, joint_id_dim)
        self.node_proj = nn.Sequential(
            nn.Linear(node_in_dim, self.graph_dim),
            nn.ReLU(),
            nn.Linear(self.graph_dim, self.graph_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(
            IntrapersonGraphLayer(dim=self.graph_dim, hidden=self.hidden, dropout=dropout)
            for _ in range(self.num_layers)
        )

        readout_in_dim = 2 * self.graph_dim + 2 + (PERSON_BBOX_DIM if self.add_bbox else 0)
        self.readout = nn.Sequential(
            nn.Linear(readout_in_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, self.out_dim),
            nn.ReLU(),
        )

        adjacency = torch.zeros(self.V, self.V, dtype=torch.bool)
        for u, v in self.edges:
            adjacency[u, v] = True
            adjacency[v, u] = True
        edge_index = adjacency.nonzero(as_tuple=False)
        self.register_buffer("edge_index", edge_index, persistent=False)

    def forward(self, pose_xyc, bbox_cxcywh=None):
        """Return one embedding per person plus the validity mask and bboxes."""
        bboxes, _keypoints, xy, conf = unpack_pose_tensor(pose_xyc, num_persons=self.K, num_joints=self.V)
        joint_mask, person_mask = compute_pose_masks(
            xy,
            conf if self.use_conf else None,
            bboxes,
            conf_thr=self.conf_thr,
            min_visible_joints=self.min_visible_joints,
        )

        joint_mask_f = joint_mask.float()
        both_hips = joint_mask[..., 6] & joint_mask[..., 7]
        midhip = 0.5 * (xy[..., 6, :] + xy[..., 7, :])
        joint_count = joint_mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        centroid = (xy * joint_mask_f.unsqueeze(-1)).sum(dim=-2) / joint_count
        # Prefer the hip midpoint as a stable body anchor; fall back to the
        # centroid of visible joints if one of the hips is missing.
        anchor_xy = torch.where(both_hips[..., None], midhip, centroid)
        rel_xy = (xy - anchor_xy.unsqueeze(-2)) * joint_mask_f.unsqueeze(-1)

        # Learnable joint IDs tell the graph which anatomical joint each node is.
        joint_ids = torch.arange(self.V, device=pose_xyc.device)
        joint_ids = self.joint_embed(joint_ids).view(*([1] * (xy.dim() - 2)), self.V, -1)
        joint_ids = joint_ids.expand(*xy.shape[:-2], self.V, -1)

        # Node features combine absolute position, position relative to body
        # anchor, validity, joint identity, and optional confidence.
        node_feat = [xy, rel_xy, joint_mask_f.unsqueeze(-1), joint_ids]
        if self.use_conf:
            node_feat.append((conf * joint_mask_f).unsqueeze(-1))
        node_feat = torch.cat(node_feat, dim=-1)
        node_feat = node_feat * joint_mask_f.unsqueeze(-1)

        h = self.node_proj(node_feat)

        # Collapse [B, T, K] into one batch axis so each person skeleton is
        # processed independently by the same joint graph layers.
        flat_shape = (-1, self.V)
        h = h.reshape(*flat_shape, self.graph_dim)
        xy_flat = xy.reshape(*flat_shape, 2)
        joint_mask_flat = joint_mask.reshape(*flat_shape)
        person_mask_flat = person_mask.reshape(-1)
        bboxes_flat = bboxes.reshape(-1, PERSON_BBOX_DIM)
        both_hips_flat = both_hips.reshape(-1, 1).float()

        for layer in self.layers:
            h = layer(h, xy_flat, joint_mask_flat, self.edge_index)

        # Read out one person embedding from the joint states using mean + max
        # pooling plus two small summary signals about skeleton quality.
        denom = joint_mask_flat.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_readout = (h * joint_mask_flat.unsqueeze(-1).float()).sum(dim=1) / denom

        # Invalid joints are suppressed before max-pooling so they never win.
        masked_h = h.masked_fill(~joint_mask_flat.unsqueeze(-1), -1e9)
        max_readout = masked_h.max(dim=1).values
        max_readout = torch.where(joint_mask_flat.any(dim=1, keepdim=True), max_readout, torch.zeros_like(max_readout))

        readout_parts = [mean_readout, max_readout, denom / float(self.V), both_hips_flat]
        if self.add_bbox:
            readout_parts.append(bboxes_flat)

        person_emb = self.readout(torch.cat(readout_parts, dim=-1))
        person_emb = person_emb * person_mask_flat.unsqueeze(-1).float()

        out_shape = (*person_mask.shape, self.out_dim)
        return person_emb.view(out_shape), person_mask, bboxes

import torch
import torch.nn as nn

class BoneMLPEncoder(nn.Module):
    def __init__(self, K=25, V=12, edges=[(0,1),(0,2),(2,4),(1,3),(3,5),(0,6),(1,7),(6,7),(6,8),(8,10),(7,9),(9,11)],
                 out_dim=32, hidden=128, use_conf=True, add_bbox=True,
                 add_masks=True, conf_thr=0.2, min_visible_joints=3):
        super().__init__()
        self.K = K
        self.V = V
        self.edges = edges
        self.use_conf = use_conf
        self.add_bbox = add_bbox
        self.add_masks = add_masks
        self.conf_thr = conf_thr
        self.min_visible_joints = min_visible_joints

        nbones = len(edges)

        in_dim = 2 #anchor centroid
        in_dim += (2 * nbones)           # bone dxdy
        if add_bbox:
            in_dim += 4

        # add explicit masks as features (recommended)
        if add_masks:
            #in_dim += (1 * V)            # joint_mask
            in_dim += (1 * nbones)       # bone_mask
            in_dim += 1+1                  # person_mask anchor_validate

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, pose_xyc, bbox_cxcywh=None):
        """
        pose_xyc: (B,T,K,V,3) preferred (x,y,conf). If (B,T,K,V,2), conf is unavailable.
        bbox_cxcywh: (B,T,K,4) where padded persons have w=h=0 (recommended).
        Returns:
            emb: (B,T,K,out_dim)
            person_mask: (B,T,K) bool
        """
        num_persons = pose_xyc[..., -1]
        persons = pose_xyc[..., :-1].contiguous().view(*pose_xyc.shape[:-1], self.K, -1)
        #print(persons.shape)
        pers_keypoints = persons[..., 4:].contiguous().view(*persons.shape[:-1], self.V, -1)
        #print(pers_keypoints.shape)
        pers_bboxes = persons[..., :4]
        #print(pers_bboxes.shape)
        xy = pers_keypoints[..., :2]  # (B,T,K,V,2)
        #print(xy.shape)
        # --- joint mask ---
        '''if pose_xyc.size(-1) >= 3:
            conf = pose_xyc[..., 2]  # (B,T,K,V)
            joint_mask = conf > self.conf_thr
        else:
            conf = None
            # WARNING: only valid if (0,0) truly indicates missing joints
            joint_mask = (xy.abs().sum(dim=-1) > 0)  # (B,T,K,V)
    
        # --- person mask ---
        if bbox_cxcywh is not None:
            w = bbox_cxcywh[..., 2]  # (B,T,K)
            h = bbox_cxcywh[..., 3]  # (B,T,K)
            person_mask = (w > 0) & (h > 0)
        else:
            person_mask = joint_mask.any(dim=-1)  # (B,T,K)'''
    
        if pers_keypoints.size(-1) >= 3:
            conf = pers_keypoints[..., 2]  # (B,T,K,V)
            joint_mask = conf > self.conf_thr
        else:
            conf = None
            # WARNING: only valid if (0,0) truly indicates missing joints
            joint_mask = (xy.abs().sum(dim=-1) > 0)  # (B,T,K,V)
    
        # --- person mask ---
        '''if bbox_cxcywh is not None:
            w = bbox_cxcywh[..., 2]  # (B,T,K)
            h = bbox_cxcywh[..., 3]  # (B,T,K)
            person_mask = (w > 0) & (h > 0)
        else:
            person_mask = joint_mask.any(dim=-1)  # (B,T,K)'''
    
        w = pers_bboxes[..., 2]  # (B,T,K)
        h = pers_bboxes[..., 3]  # (B,T,K)
        person_mask = (w > 0) & (h > 0)
    
        #person_mask = joint_mask.any(dim=-1)  # (B,T,K)
    
        # require enough visible joints (optional but usually helpful)
        person_mask = person_mask & (joint_mask.sum(dim=-1) >= self.min_visible_joints)
    
        # if person invalid, force all its joints invalid
        joint_mask = joint_mask & person_mask.unsqueeze(-1)  # (B,T,K,V)
    
        # --- bones + bone_mask ---
        bones = []
        bone_masks = []
        for (u, v) in self.edges:
            bm = joint_mask[..., u] & joint_mask[..., v]  #(B,T,K)                                                                                                      
            dv = xy[..., v, :] - xy[..., u, :]                    # (B,T,K,2)
    
            #if self.use_conf and conf is not None:
                #wgt = torch.sqrt(torch.clamp(conf[..., u] * conf[..., v], 0.0, 1.0))  # (B,T,K)
                #dv = dv * wgt.unsqueeze(-1)
    
            dv = dv * bm.unsqueeze(-1).float()                    # gate invalid bones to zero
            bones.append(dv)
            bone_masks.append(bm)
    
        bones = torch.stack(bones, dim=-2)                         # (B,T,K,Ebones,2)
        bone_mask = torch.stack(bone_masks, dim=-1)                # (B,T,K,Ebones)
    
        both_hips = joint_mask[...,6] & joint_mask[...,7]     # (B,T,K)
        midhip = 0.5*(xy[...,6,:] + xy[...,7,:])              # (B,T,K,2)
        
        anchor_xy = torch.where(both_hips[..., None], midhip, torch.zeros_like(midhip))
        anchor_valid = both_hips
    
        # --- anchor: mid-hip if possible else centroid of visible joints ---
        # hips in your reindexed 12-joint set: left_hip=6, right_hip=7
        '''lh_ok = joint_mask[..., 6]                                 # (B,T,K)
        rh_ok = joint_mask[..., 7]                                 # (B,T,K)
    
        midhip = 0.5 * (xy[..., 6, :] + xy[..., 7, :])             # (B,T,K,2)
    
        # centroid over joints (robust fallback)
        jm_f = joint_mask.unsqueeze(-1).float()                    # (B,T,K,V,1)
        sum_xy = (xy * jm_f).sum(dim=-2)                           # (B,T,K,2)
        denom = joint_mask.sum(dim=-1).clamp_min(1).unsqueeze(-1).float()  # (B,T,K,1)
        centroid = sum_xy / denom                                  # (B,T,K,2)
    
        both_hips = lh_ok & rh_ok
        only_lh   = lh_ok & ~rh_ok
        only_rh   = rh_ok & ~lh_ok
    
        anchor_xy = torch.where(
            both_hips.unsqueeze(-1),
            midhip,
            torch.where(
                only_lh.unsqueeze(-1),
                xy[..., 6, :],
                torch.where(
                    only_rh.unsqueeze(-1),
                    xy[..., 7, :],
                    centroid,
                ),
            ),
        )                                                          # (B,T,K,2)
    
        anchor_valid = (lh_ok | rh_ok | joint_mask.any(dim=-1))     # (B,T,K) bool'''
    
        # --- build feature vector (NOTE: must match your in_dim in __init__) ---
        feat = []
    
        # bones only (flatten Ebones*2)
        feat.append(bones.reshape(*bones.shape[:-2], -1))           # (B,T,K,2*Ebones)
    
        # anchor
        feat.append(anchor_xy)                                      # (B,T,K,2)
    
        if self.add_masks:
            feat.append(bone_mask.float())                          # (B,T,K,Ebones)
            feat.append(person_mask.float().unsqueeze(-1))          # (B,T,K,1)
            feat.append(anchor_valid.float().unsqueeze(-1))         # (B,T,K,1)
    
        if self.add_bbox:
            '''if bbox_cxcywh is None:
                raise ValueError("add_bbox=True but bbox_cxcywh is None")
            feat.append(bbox_cxcywh)                                # (B,T,K,4)'''
            feat.append(pers_bboxes)                                # (B,T,K,4)
        x = torch.cat(feat, dim=-1)  # (B,T,K,in_dim)
        #print(x.shape)
        emb = self.mlp(x) # (B,T,K,out_dim)
        #print(emb.shape)
        return emb, person_mask, pers_bboxes
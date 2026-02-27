#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt


def group_gaussians_by_semantics(
    language_features: torch.Tensor,
    method: str = 'kmeans',
    n_clusters: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    debug: bool = True,
    **kwargs
) -> torch.Tensor:
    if debug:
        _print_grouping_debug(language_features, method, n_clusters, similarity_threshold, kwargs)
    
    if not isinstance(language_features, torch.Tensor):
        raise ValueError(f"language_features must be torch.Tensor, got {type(language_features)}")
    if method == 'similarity':
        if similarity_threshold is None:
            raise ValueError("similarity_threshold must be provided for similarity method")
        groups, C = group_by_similarity_fast(
            language_features,
            similarity_threshold,
            prototype_size=kwargs.get("prototype_size", 2048),
            seed=kwargs.get("seed", 0)
        )
        return groups
    elif method == 'kmeans':
        if n_clusters is None or int(n_clusters) <= 0:
            raise ValueError("kmeans requires a positive n_clusters")
        feats = language_features.to(torch.float32)
        n = feats.shape[0]
        k = int(n_clusters)
        device = feats.device
        max_iters = int(kwargs.get("max_iters", 50))
        tol = float(kwargs.get("tol", 1e-4))
        idx = torch.randperm(n, device=device)[:k]
        centroids = feats[idx].clone()
        prev_assign = None
        for _ in range(max_iters):
            dists = torch.cdist(feats, centroids, p=2)
            assign = torch.argmin(dists, dim=1)
            if prev_assign is not None and torch.equal(assign, prev_assign):
                break
            prev_assign = assign.clone()
            new_centroids = centroids.clone()
            for c in range(k):
                mask = (assign == c)
                if mask.any():
                    new_centroids[c] = feats[mask].mean(dim=0)
                else:
                    ridx = torch.randint(0, n, (1,), device=device)
                    new_centroids[c] = feats[ridx]
            shift = torch.norm(new_centroids - centroids, p=2, dim=1).max().item()
            centroids = new_centroids
            if shift < tol:
                break
        return assign
    else:
        raise ValueError(f"Unknown method: {method}")

@torch.no_grad()
def group_by_similarity_fast(
    language_features: torch.Tensor,
    similarity_threshold: float,
    prototype_size: int = 2048,
    seed: int = 0,
):
    feats = F.normalize(language_features.float(), dim=1)  # (N,D)
    N, D = feats.shape
    device = feats.device
    thr = float(similarity_threshold)

    # 1) sample prototypes
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    M = min(prototype_size, N)
    idx = torch.randperm(N, generator=g, device=device)[:M]
    protos = feats[idx]  # (M,D)

    # 2) build centers from prototypes (still greedy but M << N)
    centers = []
    counts = []
    for i in range(M):
        f = protos[i]
        if len(centers) == 0:
            centers.append(f)
            counts.append(1)
            continue
        C = torch.stack(centers, dim=0)           # (K,D), K small
        sims = C @ f                               # (K,)
        mv, mi = sims.max(dim=0)
        if mv.item() >= thr:
            k = int(mi.item())
            centers[k] = F.normalize(centers[k] * counts[k] + f, dim=0)
            counts[k] += 1
        else:
            centers.append(f)
            counts.append(1)
    C = torch.stack(centers, dim=0)  # (K,D)

    # 3) assign all points in one shot
    sims = feats @ C.t()             # (N,K)
    max_val, max_idx = sims.max(dim=1)
    groups = torch.where(max_val >= thr, max_idx, torch.full_like(max_idx, -1))
    return groups, C

def export_groups_to_ply(
    xyz: torch.Tensor,
    groups: torch.Tensor,
    out_path: str,
    colormap: str = "tab20"
):
    pts = xyz.detach().cpu().numpy() if isinstance(xyz, torch.Tensor) else np.asarray(xyz)
    # according to groups, assign colors
    g = groups.detach().cpu().numpy().astype(np.int64) if isinstance(groups, torch.Tensor) else np.asarray(groups).astype(np.int64)
    n = pts.shape[0]
    cmap = matplotlib.colormaps[colormap]
    colors = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        cid = int(g[i]) if g[i] >= 0 else 0
        base = cmap(cid % cmap.N)
        colors[i] = np.array(base[:3], dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_path, pcd)

def _print_grouping_debug(
    language_features: torch.Tensor,
    method: str,
    n_clusters: Optional[int],
    similarity_threshold: Optional[float],
    kwargs: Dict
):
    print("=" * 60)
    print("[group_gaussians_by_semantics] Debug Info")
    print("=" * 60)
    print("\n1. language_features")
    print(f"   - type: {type(language_features)}")
    if isinstance(language_features, torch.Tensor):
        print(f"   - shape: {language_features.shape}")
        print(f"   - dtype: {language_features.dtype}")
        print(f"   - device: {language_features.device}")
        print(f"   - is_cuda: {language_features.is_cuda}")
        print(f"   - min: {language_features.min().item():.6f}")
        print(f"   - max: {language_features.max().item():.6f}")
        print(f"   - mean: {language_features.mean().item():.6f}")
        print(f"   - std: {language_features.std().item():.6f}")
        if language_features.dim() == 2:
            norms = torch.norm(language_features, dim=1)
            print(f"   - L2-norm range: [{norms.min().item():.6f}, {norms.max().item():.6f}]")
            print(f"   - L2-norm mean: {norms.mean().item():.6f}")
            if torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
                print(f"   - normalized: True (L2≈1)")
            else:
                print(f"   - normalized: False")
        print(f"   - samples (first 5, first 3 dims):")
        for i in range(min(5, language_features.shape[0])):
            feat_sample = language_features[i]
            if feat_sample.shape[0] >= 3:
                print(f"     sample {i}: [{feat_sample[0].item():.4f}, {feat_sample[1].item():.4f}, {feat_sample[2].item():.4f}, ...]")
            else:
                print(f"     sample {i}: {feat_sample.cpu().numpy()}")
    else:
        print(f"   - warning: language_features is not torch.Tensor")
    print("\n2. method")
    print(f"   - value: '{method}'")
    print(f"   - type: {type(method)}")
    valid_methods = ['kmeans', 'similarity', 'hierarchical', 'dbscan']
    if method in valid_methods:
        print(f"   - valid: True")
    else:
        print(f"   - valid: False, expected one of {valid_methods}")
    print("\n3. n_clusters")
    print(f"   - value: {n_clusters}")
    print(f"   - type: {type(n_clusters)}")
    if n_clusters is not None and isinstance(language_features, torch.Tensor):
        n_points = language_features.shape[0]
        if n_clusters > n_points:
            print(f"   - warning: n_clusters ({n_clusters}) > points ({n_points})")
        elif n_clusters <= 0:
            print(f"   - warning: n_clusters ({n_clusters}) <= 0")
        else:
            print(f"   - valid: True")
    print("\n4. similarity_threshold")
    print(f"   - value: {similarity_threshold}")
    print(f"   - type: {type(similarity_threshold)}")
    if similarity_threshold is not None:
        if 0 <= similarity_threshold <= 1:
            print(f"   - valid: True")
        else:
            print(f"   - warning: threshold out of [0, 1]")
    print("\n5. kwargs")
    if kwargs:
        for key, value in kwargs.items():
            print(f"   - {key}: {value} (type: {type(value)})")
    else:
        print(f"   - none")
    print("=" * 60)
    print("[group_gaussians_by_semantics] Debug End")
    print("=" * 60)


def compute_group_centers(xyz, opacity, group_indices):
    device, dtype = xyz.device, xyz.dtype
    N = xyz.shape[0]
    w = opacity.reshape(N, 1).to(dtype)
    g = group_indices.to(device=device, dtype=torch.long)
    # some gaussians may not be assigned to any group
    valid = g >= 0
    if not valid.any():
        return torch.empty((0, 3), device=device, dtype=dtype)

    xyz2 = xyz[valid]
    w2 = w[valid]
    g2 = g[valid]

    M = int(g2.max().item()) + 1
    sum_w = torch.zeros((M, 1), device=device, dtype=dtype)
    sum_wxyz = torch.zeros((M, 3), device=device, dtype=dtype)

    sum_w.index_add_(0, g2, w2)
    sum_wxyz.index_add_(0, g2, w2 * xyz2)

    eps = torch.finfo(dtype).eps
    centers = sum_wxyz / sum_w.clamp_min(eps)
    centers[sum_w.squeeze(1) <= eps] = 0
    return centers

def project_points(
    points3d: torch.Tensor,
    full_proj_transform: torch.Tensor,
    width: int,
    height: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D points to 2D screen coordinates using full projection matrix.
    
    Args:
        points3d: (N, 3) 3D points
        full_proj_transform: (4, 4) World-to-Clip matrix (transposed).
                             Usually stored in camera.full_proj_transform.
                             v_clip = v_world @ full_proj_transform
        width: Image width
        height: Image height
        
    Returns:
        points2d: (N, 2) 2D screen coordinates (x, y)
        valid_mask: (N,) Boolean mask indicating points in front of camera
    """
    # 1. World to Clip
    N = points3d.shape[0]
    # Ensure points3d and full_proj_transform are on the same device
    device = full_proj_transform.device
    points3d = points3d.to(device)
    
    ones = torch.ones((N, 1), device=device, dtype=points3d.dtype)
    points_hom = torch.cat([points3d, ones], dim=1) # (N, 4)
    
    points_clip = points_hom @ full_proj_transform
    
    # 2. Clip to NDC
    # p_ndc = p_clip / w
    w = points_clip[:, 3:4]
    valid_mask = (w > 0.01).squeeze(1) # Check if point is in front of camera
    
    # Add epsilon to w to avoid NaN/Inf
    points_ndc = points_clip[:, :3] / (w + 1e-6)
    
    # 3. NDC to Screen
    # Assuming standard OpenGL NDC (y up) and Image coordinates (y down)
    points2d = torch.zeros((N, 2), device=device, dtype=points3d.dtype)
    points2d[:, 0] = (points_ndc[:, 0] + 1) * width / 2.0
    points2d[:, 1] = (points_ndc[:, 1] + 1) * height / 2.0
    
    return points2d, valid_mask

"""
Quality metrics for evaluating TRELLIS.2 generated 3D models.

Compares input images to rendered views of the output mesh using:
- DINO feature similarity (pose-agnostic, semantic)
- LPIPS perceptual similarity
- SSIM structural similarity
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple

from .render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics, render_frames
from .loss_utils import ssim, lpips


def render_standard_views(
    mesh,
    resolution: int = 512,
    views: List[str] = None,
    envmap=None,
) -> Dict[str, np.ndarray]:
    """
    Render mesh from standard viewpoints (front, right, back, left).

    Args:
        mesh: MeshWithVoxel or similar mesh object
        resolution: Output image resolution
        views: List of view names to render (default: front, right, back, left)

    Returns:
        Dict mapping view name to RGB image as numpy array (H, W, 3) uint8
    """
    if views is None:
        views = ['front', 'right', 'back', 'left']

    # Standard yaw angles for each view (radians)
    view_yaws = {
        'front': np.pi / 2,      # 90 degrees
        'right': 0,              # 0 degrees
        'back': -np.pi / 2,      # -90 degrees (or 270)
        'left': np.pi,           # 180 degrees
    }

    yaws = [view_yaws[v] for v in views]
    pitchs = [0.25] * len(views)  # Slight elevation

    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitchs, 2, 40
    )

    result = render_frames(
        mesh, extrinsics, intrinsics,
        options={'resolution': resolution, 'bg_color': (1, 1, 1)},
        verbose=False,
        envmap=envmap,
    )

    rendered = {}
    for i, view_name in enumerate(views):
        # Use 'shaded' if available (PBR mesh), else 'color'
        if 'shaded' in result:
            rendered[view_name] = result['shaded'][i]
        else:
            rendered[view_name] = result['color'][i]

    return rendered


def pil_to_tensor(img: Image.Image, size: int = 512) -> torch.Tensor:
    """Convert PIL image to normalized tensor (C, H, W) in [0, 1]."""
    img = img.convert('RGB').resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    return tensor


def numpy_to_tensor(arr: np.ndarray, size: int = 512) -> torch.Tensor:
    """Convert numpy array (H, W, 3) uint8 to normalized tensor."""
    img = Image.fromarray(arr).resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def compute_dino_similarity(
    input_images: List[Image.Image],
    rendered_views: Dict[str, np.ndarray],
    feature_extractor,
    resolution: int = 512,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute DINO feature similarity between inputs and renders.

    Uses cosine similarity between DINO features, which is pose-agnostic
    and captures semantic similarity.

    Args:
        input_images: List of input PIL images
        rendered_views: Dict of rendered views as numpy arrays
        feature_extractor: DINO feature extractor (pipeline.image_cond_model)
        resolution: Resolution for feature extraction

    Returns:
        Tuple of (overall_similarity, per_view_similarities)
    """
    # Set resolution for feature extraction
    old_size = getattr(feature_extractor, 'image_size', 512)
    feature_extractor.image_size = resolution

    # Extract features for input images
    with torch.no_grad():
        input_features = feature_extractor(input_images)  # (N, patches, dim)
        # Average pool over patches to get global feature
        input_global = input_features.mean(dim=1)  # (N, dim)
        input_global = F.normalize(input_global, dim=-1)

    # Extract features for rendered views
    render_pils = [
        Image.fromarray(arr) for arr in rendered_views.values()
    ]
    with torch.no_grad():
        render_features = feature_extractor(render_pils)
        render_global = render_features.mean(dim=1)
        render_global = F.normalize(render_global, dim=-1)

    # Restore original size
    feature_extractor.image_size = old_size

    # Compute similarity matrix (inputs x renders)
    sim_matrix = torch.mm(input_global, render_global.t())  # (N_in, N_render)

    # For each input, find best matching render
    best_matches = sim_matrix.max(dim=1).values  # (N_in,)
    overall_sim = best_matches.mean().item()

    # Per-view similarities (average across inputs for each render)
    per_view = {}
    view_names = list(rendered_views.keys())
    for i, view_name in enumerate(view_names):
        per_view[view_name] = sim_matrix[:, i].mean().item()

    return overall_sim, per_view


def compute_perceptual_metrics(
    input_images: List[Image.Image],
    rendered_views: Dict[str, np.ndarray],
    size: int = 256,
) -> Dict[str, float]:
    """
    Compute LPIPS and SSIM between inputs and best-matching renders.

    Args:
        input_images: List of input PIL images
        rendered_views: Dict of rendered views as numpy arrays
        size: Image size for comparison

    Returns:
        Dict with 'lpips' and 'ssim' values
    """
    # Convert all to tensors
    input_tensors = torch.stack([
        pil_to_tensor(img, size) for img in input_images
    ]).cuda()  # (N, 3, H, W)

    render_tensors = torch.stack([
        numpy_to_tensor(arr, size) for arr in rendered_views.values()
    ]).cuda()  # (M, 3, H, W)

    # For each input, find best matching render by SSIM
    lpips_scores = []
    ssim_scores = []

    for i in range(len(input_tensors)):
        input_t = input_tensors[i:i+1]  # (1, 3, H, W)

        # Find best SSIM match
        best_ssim = -1
        best_idx = 0
        for j in range(len(render_tensors)):
            render_t = render_tensors[j:j+1]
            s = ssim(input_t, render_t).item()
            if s > best_ssim:
                best_ssim = s
                best_idx = j

        # Compute metrics with best match
        best_render = render_tensors[best_idx:best_idx+1]
        lpips_val = lpips(input_t, best_render).item()

        lpips_scores.append(lpips_val)
        ssim_scores.append(best_ssim)

    return {
        'lpips': np.mean(lpips_scores),
        'ssim': np.mean(ssim_scores),
    }


def compute_quality_metrics(
    input_images: List[Image.Image],
    mesh,
    feature_extractor,
    render_resolution: int = 512,
    metric_resolution: int = 256,
    envmap=None,
) -> Dict[str, any]:
    """
    Compute all quality metrics comparing inputs to rendered mesh.

    Args:
        input_images: List of input PIL images
        mesh: Generated mesh object
        feature_extractor: DINO feature extractor
        render_resolution: Resolution for rendering
        metric_resolution: Resolution for LPIPS/SSIM computation

    Returns:
        Dict with all metrics
    """
    # Render standard views
    rendered = render_standard_views(mesh, render_resolution, envmap=envmap)

    # Compute DINO similarity
    dino_sim, dino_per_view = compute_dino_similarity(
        input_images, rendered, feature_extractor, render_resolution
    )

    # Compute perceptual metrics
    perceptual = compute_perceptual_metrics(
        input_images, rendered, metric_resolution
    )

    return {
        'dino_similarity': dino_sim,
        'dino_per_view': dino_per_view,
        'lpips': perceptual['lpips'],
        'ssim': perceptual['ssim'],
        'rendered_views': rendered,
    }


def format_metrics_report(metrics: Dict[str, any]) -> str:
    """Format metrics as a human-readable report."""
    lines = [
        "=" * 40,
        "QUALITY METRICS REPORT",
        "=" * 40,
        "",
        f"DINO Similarity: {metrics['dino_similarity']:.3f}",
        "  (higher = better, range 0-1)",
        "",
        f"LPIPS: {metrics['lpips']:.3f}",
        "  (lower = better, range 0-1)",
        "",
        f"SSIM: {metrics['ssim']:.3f}",
        "  (higher = better, range 0-1)",
        "",
        "Per-view DINO similarities:",
    ]

    for view_name, sim in metrics['dino_per_view'].items():
        lines.append(f"  {view_name}: {sim:.3f}")

    lines.extend([
        "",
        "=" * 40,
    ])

    return "\n".join(lines)

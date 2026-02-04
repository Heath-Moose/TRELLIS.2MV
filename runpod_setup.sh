#!/bin/bash
# TRELLIS.2 Multi-View Setup for RunPod
# Paste this entire script into your RunPod terminal

set -e
cd /workspace/TRELLIS.2

echo "=== Setting up TRELLIS.2 Multi-View ==="

# 1. Create multiview pipeline
cat > trellis2/pipelines/trellis2_multiview.py << 'ENDOFFILE'
"""
Multi-view extension for TRELLIS.2 Image-to-3D Pipeline.

This module implements tuning-free multi-view 3D generation by aggregating
predictions from multiple input views during the denoising process.

Based on the approach from TRELLIS v1's run_multi_image() method.
"""

from typing import *
from contextlib import contextmanager
import torch
import numpy as np
from PIL import Image
from .trellis2_image_to_3d import Trellis2ImageTo3DPipeline
from .samplers import FlowEulerSampler
from ..representations import MeshWithVoxel


class Trellis2MultiViewPipeline(Trellis2ImageTo3DPipeline):
    """
    Extended pipeline that supports multi-view image input for improved 3D generation.

    Implements two aggregation algorithms:
    - 'stochastic': Cycles through views sequentially during denoising
    - 'multidiffusion': Averages predictions from all views at each step
    """

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Context manager that injects multi-image aggregation into a sampler.
        """
        old_inference_model = sampler._inference_model
        setattr(sampler, '_old_inference_model', old_inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images ({num_images}) is greater "
                      f"than number of steps ({num_steps}). This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()

            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)

        elif mode == 'multidiffusion':
            def _new_inference_model(self, model, x_t, t, cond, neg_cond=None, guidance_strength=1.0, **kwargs):
                guidance_interval = kwargs.pop('guidance_interval', (0.0, 1.0))
                in_guidance_interval = guidance_interval[0] <= t <= guidance_interval[1]

                preds = []
                for i in range(len(cond)):
                    cond_i = cond[i:i+1]
                    pred = FlowEulerSampler._inference_model(self, model, x_t, t, cond_i, **kwargs)
                    preds.append(pred)

                pred_avg = sum(preds) / len(preds)

                if in_guidance_interval and guidance_strength != 1 and neg_cond is not None:
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    pred = guidance_strength * pred_avg + (1 - guidance_strength) * neg_pred
                else:
                    pred = pred_avg

                return pred
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'stochastic' or 'multidiffusion'.")

        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        try:
            yield
        finally:
            sampler._inference_model = old_inference_model
            delattr(sampler, '_old_inference_model')

    def get_cond_multi(
        self,
        images: List[Image.Image],
        resolution: int,
        include_neg_cond: bool = True
    ) -> dict:
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)

        cond = self.image_cond_model(images)

        if self.low_vram:
            self.image_cond_model.cpu()

        if not include_neg_cond:
            return {'cond': cond}

        neg_cond = torch.zeros_like(cond[:1])

        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> List[MeshWithVoxel]:
        if len(images) < 2:
            print("Warning: run_multi_image called with fewer than 2 images. "
                  "Consider using run() for single-image generation.")

        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models
            assert 'tex_slat_flow_model_512' in self.models
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models
            assert 'tex_slat_flow_model_1024' in self.models
        elif pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models
            assert 'shape_slat_flow_model_1024' in self.models
            assert 'tex_slat_flow_model_1024' in self.models
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models
            assert 'shape_slat_flow_model_1024' in self.models
            assert 'tex_slat_flow_model_1024' in self.models
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")

        if preprocess_image:
            images = [self.preprocess_image(img) for img in images]

        torch.manual_seed(seed)
        num_images = len(images)

        cond_512 = self.get_cond_multi(images, 512)
        cond_1024 = self.get_cond_multi(images, 1024) if pipeline_type != '512' else None

        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps', 12)
        shape_steps = {**self.shape_slat_sampler_params, **shape_slat_sampler_params}.get('steps', 12)
        tex_steps = {**self.tex_slat_sampler_params, **tex_slat_sampler_params}.get('steps', 12)

        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]

        with self.inject_sampler_multi_image(
            self.sparse_structure_sampler, num_images, ss_steps, mode
        ):
            coords = self.sample_sparse_structure(
                cond_512, ss_res, num_samples, sparse_structure_sampler_params
            )

        if pipeline_type == '512':
            with self.inject_sampler_multi_image(
                self.shape_slat_sampler, num_images, shape_steps, mode
            ):
                shape_slat = self.sample_shape_slat(
                    cond_512, self.models['shape_slat_flow_model_512'],
                    coords, shape_slat_sampler_params
                )
            res = 512

        elif pipeline_type == '1024':
            with self.inject_sampler_multi_image(
                self.shape_slat_sampler, num_images, shape_steps, mode
            ):
                shape_slat = self.sample_shape_slat(
                    cond_1024, self.models['shape_slat_flow_model_1024'],
                    coords, shape_slat_sampler_params
                )
            res = 1024

        elif pipeline_type in ['1024_cascade', '1536_cascade']:
            target_res = 1024 if pipeline_type == '1024_cascade' else 1536

            with self.inject_sampler_multi_image(
                self.shape_slat_sampler, num_images, shape_steps * 2, mode
            ):
                shape_slat, res = self.sample_shape_slat_cascade(
                    cond_512, cond_1024,
                    self.models['shape_slat_flow_model_512'],
                    self.models['shape_slat_flow_model_1024'],
                    512, target_res,
                    coords, shape_slat_sampler_params,
                    max_num_tokens
                )

        tex_cond = cond_512 if pipeline_type == '512' else cond_1024
        tex_model = self.models['tex_slat_flow_model_512'] if pipeline_type == '512' else self.models['tex_slat_flow_model_1024']

        with self.inject_sampler_multi_image(
            self.tex_slat_sampler, num_images, tex_steps, mode
        ):
            tex_slat = self.sample_tex_slat(
                tex_cond, tex_model, shape_slat, tex_slat_sampler_params
            )

        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)

        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh
ENDOFFILE

echo "Created trellis2_multiview.py"

# 2. Update __init__.py
cat > trellis2/pipelines/__init__.py << 'ENDOFFILE'
import importlib

__attributes = {
    "Trellis2ImageTo3DPipeline": "trellis2_image_to_3d",
    "Trellis2TexturingPipeline": "trellis2_texturing",
    "Trellis2MultiViewPipeline": "trellis2_multiview",
}

__submodules = ['samplers', 'rembg']

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str):
    import os
    import json
    is_local = os.path.exists(f"{path}/pipeline.json")

    if is_local:
        config_file = f"{path}/pipeline.json"
    else:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(path, "pipeline.json")

    with open(config_file, 'r') as f:
        config = json.load(f)
    return globals()[config['name']].from_pretrained(path)


# For PyLance
if __name__ == '__main__':
    from . import samplers, rembg
    from .trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    from .trellis2_texturing import Trellis2TexturingPipeline
    from .trellis2_multiview import Trellis2MultiViewPipeline
ENDOFFILE

echo "Updated __init__.py"

# 3. Create example script
cat > example_multiview.py << 'ENDOFFILE'
"""Multi-view 3D Generation Example for TRELLIS.2"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2MultiViewPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel


def main():
    parser = argparse.ArgumentParser(description='TRELLIS.2 Multi-View 3D Generation')
    parser.add_argument('--images', nargs='+', type=str, help='Paths to view images')
    parser.add_argument('--mode', type=str, default='stochastic',
                        choices=['stochastic', 'multidiffusion'])
    parser.add_argument('--output', type=str, default='sample_multiview')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()

    if args.images:
        image_paths = args.images
    else:
        print("No images specified. Using default example image twice.")
        image_paths = ['assets/example_image/T.png', 'assets/example_image/T.png']

    print(f"Loading {len(image_paths)} images:")
    for p in image_paths:
        print(f"  - {p}")

    images = [Image.open(p) for p in image_paths]

    envmap = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device='cuda'
    ))

    print("\nLoading TRELLIS.2 Multi-View Pipeline...")
    pipeline = Trellis2MultiViewPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()

    print(f"\nGenerating 3D mesh with {len(images)} views using '{args.mode}' mode...")
    mesh = pipeline.run_multi_image(images, seed=args.seed, mode=args.mode)[0]
    mesh.simplify(16777216)

    if not args.no_video:
        print(f"\nRendering video to {args.output}.mp4...")
        video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
        imageio.mimsave(f"{args.output}.mp4", video, fps=15)

    print(f"\nExporting to {args.output}.glb...")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], decimation_target=1000000,
        texture_size=4096, remesh=True, remesh_band=1, remesh_project=0, verbose=True
    )
    glb.export(f"{args.output}.glb", extension_webp=True)
    print(f"\nDone! Output: {args.output}.glb")


if __name__ == '__main__':
    main()
ENDOFFILE

echo "Created example_multiview.py"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Now run:"
echo "  1. Activate environment:"
echo "     eval \"\$(/workspace/miniconda3/bin/conda shell.bash hook)\""
echo "     conda activate trellis2"
echo "     export CUDA_HOME=/usr/local/cuda-12.4"
echo ""
echo "  2. Fix cumesh (if needed):"
echo "     export NVCC_PREPEND_FLAGS=\"--extended-lambda\""
echo "     pip install git+https://github.com/JeffreyXiang/cumesh.git"
echo ""
echo "  3. Test single-image baseline:"
echo "     python example.py"
echo ""
echo "  4. Test multi-view:"
echo "     python example_multiview.py --mode stochastic"

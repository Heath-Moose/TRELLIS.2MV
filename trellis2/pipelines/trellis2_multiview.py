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

        Temporarily patches the sampler's _inference_model method to handle
        multiple conditioning images during the denoising loop.

        Args:
            sampler: The sampler instance to patch
            num_images: Number of conditioning images
            num_steps: Number of sampling steps
            mode: Aggregation mode - 'stochastic' or 'multidiffusion'
        """
        # Save original method (bound method)
        old_inference_model = sampler._inference_model
        setattr(sampler, '_old_inference_model', old_inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images ({num_images}) is greater "
                      f"than number of steps ({num_steps}). This may lead to performance degradation.\033[0m")

            # Pre-compute which image to use at each step (cycles through images)
            cond_indices = (np.arange(num_steps) % num_images).tolist()

            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                # Pop the next conditioning index
                cond_idx = cond_indices.pop(0)
                # Select single image's conditioning
                cond_i = cond[cond_idx:cond_idx+1]
                # Call original method with single-view conditioning
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)

        elif mode == 'multidiffusion':
            def _new_inference_model(self, model, x_t, t, cond, neg_cond=None, guidance_strength=1.0, **kwargs):
                """
                Multidiffusion: compute predictions for all views and average them.
                """
                # Get guidance interval if specified
                guidance_interval = kwargs.pop('guidance_interval', (0.0, 1.0))
                # Remove sampler-specific kwargs that model doesn't understand
                kwargs.pop('guidance_rescale', None)

                # Check if we're in the guidance interval
                in_guidance_interval = guidance_interval[0] <= t <= guidance_interval[1]

                # Compute predictions for each conditioning image using base inference
                # (bypass CFG mixin to get raw predictions)
                preds = []
                for i in range(len(cond)):
                    cond_i = cond[i:i+1]
                    # Call base FlowEulerSampler's _inference_model (without CFG)
                    pred = FlowEulerSampler._inference_model(self, model, x_t, t, cond_i, **kwargs)
                    preds.append(pred)

                # Average predictions across all views
                pred_avg = sum(preds) / len(preds)

                # Apply CFG if in guidance interval and guidance_strength != 1
                if in_guidance_interval and guidance_strength != 1 and neg_cond is not None:
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    pred = guidance_strength * pred_avg + (1 - guidance_strength) * neg_pred
                else:
                    pred = pred_avg

                return pred

        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'stochastic' or 'multidiffusion'.")

        # Bind the new function as a method and patch the sampler
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        try:
            yield
        finally:
            # Restore original method
            sampler._inference_model = old_inference_model
            delattr(sampler, '_old_inference_model')

    def get_cond_multi(
        self,
        images: List[Image.Image],
        resolution: int,
        include_neg_cond: bool = True
    ) -> dict:
        """
        Get conditioning information for multiple images.

        Args:
            images: List of PIL images
            resolution: Target resolution for feature extraction
            include_neg_cond: Whether to include negative conditioning

        Returns:
            dict with 'cond' tensor of shape (N, num_patches, feature_dim)
            and optionally 'neg_cond' tensor
        """
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)

        # Extract features for all images
        cond = self.image_cond_model(images)

        if self.low_vram:
            self.image_cond_model.cpu()

        if not include_neg_cond:
            return {'cond': cond}

        # Single negative conditioning (zeros)
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
        """
        Run the pipeline with multiple input images for improved 3D generation.

        This is a tuning-free approach that aggregates predictions from multiple
        views during the denoising process. No specialized training is required.

        Args:
            images: List of PIL images representing different views of the object
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility
            sparse_structure_sampler_params: Additional params for sparse structure sampler
            shape_slat_sampler_params: Additional params for shape latent sampler
            tex_slat_sampler_params: Additional params for texture latent sampler
            preprocess_image: Whether to preprocess images (background removal, cropping)
            return_latent: Whether to return latent codes along with mesh
            pipeline_type: Pipeline type ('512', '1024', '1024_cascade', '1536_cascade')
            max_num_tokens: Maximum number of tokens
            mode: Aggregation mode:
                - 'stochastic': Cycles through views at each denoising step (faster)
                - 'multidiffusion': Averages predictions from all views (more consistent)

        Returns:
            List of MeshWithVoxel objects (one per sample)

        Note:
            This is an experimental algorithm without training a specialized model.
            Results may vary, especially for images with different poses or
            inconsistent details across views.
        """
        if len(images) < 2:
            print("Warning: run_multi_image called with fewer than 2 images. "
                  "Consider using run() for single-image generation.")

        # Check pipeline type
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

        # Preprocess all images
        if preprocess_image:
            images = [self.preprocess_image(img) for img in images]

        torch.manual_seed(seed)
        num_images = len(images)

        # Get conditioning for all images at required resolutions
        cond_512 = self.get_cond_multi(images, 512)
        cond_1024 = self.get_cond_multi(images, 1024) if pipeline_type != '512' else None

        # Get sampling step counts from params (with defaults)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps', 12)
        shape_steps = {**self.shape_slat_sampler_params, **shape_slat_sampler_params}.get('steps', 12)
        tex_steps = {**self.tex_slat_sampler_params, **tex_slat_sampler_params}.get('steps', 12)

        # === Stage 1: Sparse Structure ===
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]

        with self.inject_sampler_multi_image(
            self.sparse_structure_sampler, num_images, ss_steps, mode
        ):
            coords = self.sample_sparse_structure(
                cond_512, ss_res, num_samples, sparse_structure_sampler_params
            )

        # === Stage 2: Shape Latent ===
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

            # For cascade, we need to inject into both LR and HR sampling
            # The cascade method does two sampling passes internally
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

        # === Stage 3: Texture Latent ===
        tex_cond = cond_512 if pipeline_type == '512' else cond_1024
        tex_model = self.models['tex_slat_flow_model_512'] if pipeline_type == '512' else self.models['tex_slat_flow_model_1024']

        with self.inject_sampler_multi_image(
            self.tex_slat_sampler, num_images, tex_steps, mode
        ):
            tex_slat = self.sample_tex_slat(
                tex_cond, tex_model, shape_slat, tex_slat_sampler_params
            )

        # === Decode ===
        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)

        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh

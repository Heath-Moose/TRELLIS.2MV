# TRELLIS.2 Multi-View Session History

## Summary

Successfully migrated TRELLIS.2 (4B model) with custom multi-view support from RunPod to Azure ML (MooseAML6 compute instance with A100 80GB GPU).

---

## Multi-View Implementation

### How It Works
The multi-view capability was ported from TRELLIS v1 to v2 using a **tuning-free** approach - no special training required.

### Key Technique
Manipulate the conditioning during the diffusion denoising loop by patching the sampler's `_inference_model` method:

**Stochastic Mode** - Cycle through views at each denoising step:
```
Step 0: denoise(x, cond=view_0_features)
Step 1: denoise(x, cond=view_1_features)
Step 2: denoise(x, cond=view_0_features)  # wraps around
```

**Multidiffusion Mode** - Average predictions from ALL views at each step:
```
Step N: pred = average(denoise(x, view_0), denoise(x, view_1), ...)
```

### Why This Works
- TRELLIS.2 uses DINOv2/v3 image encoder → flow-based diffusion → 3D latent decoding
- The conditioning is just image features passed to the denoising network
- By swapping/averaging conditioning at each step, the model sees multiple views
- Known technique: "multidiffusion" for panoramas, "stochastic conditioning" for multi-view 3D

### Files Added
- `trellis2/pipelines/trellis2_multiview.py` - Multi-view pipeline class
- `trellis2/pipelines/__init__.py` - Modified to register new pipeline
- `example_multiview.py` - CLI test script

### Usage
```bash
# 2-view
python example_multiview.py --images front.jpg back.jpg --mode multidiffusion

# 4-view
python example_multiview.py --images front.jpg back.jpg left.jpg right.jpg --mode stochastic
```

---

## Bug Fixes Applied

**CuMesh build fix** (device-annotated lambdas):
```bash
sed -i "s/nvcc_flags = \[/nvcc_flags = ['--extended-lambda', /" setup.py
```

**o-voxel Eigen fix**:
```bash
git submodule update --init --recursive
```

**transformers version fix** (breaks RMBG-2.0):
```bash
pip install transformers==4.57.3
```

**Multi-view guidance_rescale fix** (added to `trellis2_multiview.py` line 77):
```python
kwargs.pop('guidance_rescale', None)
```

---

## Generated Outputs
- `cyborg_rat_azure.glb` / `.mp4` - 2-view test
- `cyborg_rat_4view.glb` / `.mp4` - 4-view high quality (multidiffusion mode)

---

## Configurable Parameters

### Command Line (example_multiview.py)

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `--mode` | `stochastic`, `multidiffusion` | `stochastic` | Aggregation mode |
| `--pipeline-type` | `512`, `1024`, `1024_cascade`, `1536_cascade` | `1024_cascade` | Resolution pipeline |
| `--seed` | integer | `42` | Random seed |
| `--images` | paths | - | Input view images |

### Sampler Parameters (in code)

```python
mesh = pipeline.run_multi_image(
    images,
    mode='multidiffusion',
    seed=42,
    pipeline_type='1024_cascade',
    sparse_structure_sampler_params={
        'steps': 12,              # More steps = better quality (default: 12)
        'guidance_strength': 7.5, # Higher = stronger adherence to input
        'guidance_rescale': 0.7,  # Prevents variance collapse
        'rescale_t': 5.0,         # Time schedule rescaling
    },
    shape_slat_sampler_params={
        'steps': 12,
        'guidance_strength': 7.5,
        'guidance_rescale': 0.5,
        'rescale_t': 3.0,
    },
    tex_slat_sampler_params={
        'steps': 12,
        'guidance_strength': 1.0,
        'guidance_rescale': 0.0,
        'rescale_t': 3.0,
    },
)
```

### Parameter Ranges (from app.py)

| Parameter | Sparse Structure | Shape SLat | Texture SLat |
|-----------|-----------------|-----------|--------------|
| steps | 12 (1-50) | 12 (1-50) | 12 (1-50) |
| guidance_strength | 7.5 (1-10) | 7.5 (1-10) | 1.0 (1-10) |
| guidance_rescale | 0.7 (0-1) | 0.5 (0-1) | 0.0 (0-1) |
| rescale_t | 5.0 (1-6) | 3.0 (1-6) | 3.0 (1-6) |

---

## Training & Fine-Tuning Capabilities

TRELLIS.2 has **full training support** with:

### Trainable Components
1. **Shape VAE** - Encodes 3D geometry (FlexiDualGrid)
2. **Texture VAE** - Encodes PBR materials
3. **Sparse Structure Flow** - Image → structure
4. **Shape SLat Flow** - Structure → shape latent (cascade: 512→1024)
5. **Texture SLat Flow** - Shape → texture latent (cascade: 512→1024)

### Training Configs
```
configs/scvae/shape_vae_next_dc_f16c32_fp16.json
configs/scvae/shape_vae_next_dc_f16c32_fp16_ft_512.json
configs/scvae/tex_vae_next_dc_f16c32_fp16.json
configs/scvae/tex_vae_next_dc_f16c32_fp16_ft_512.json
configs/gen/ss_flow_img_dit_1_3B_64_bf16.json
configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json
configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_ft1024.json
configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json
configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_ft1024.json
```

### Training Command Example
```bash
python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json \
  --output_dir results/my_training \
  --data_dir '{"ObjaverseXL_sketchfab": {...}}'
```

### Fine-tuning Support
- Load pretrained checkpoints via `finetune_ckpt` config parameter
- Lower learning rates recommended (1e-5 vs 1e-4)
- Multi-resolution progressive training supported

---

## Image Conditioning Architecture

### Encoders
- **DinoV2**: 518x518 fixed resolution
- **DinoV3**: Configurable (512/1024/1536)

### Conditioning Flow
```
Image → DINOv3 Encoder → Patch Tokens → Cross-Attention in Flow Models
```

### Multi-View Modes
- **Stochastic**: Cycles through views at each denoising step
- **Multidiffusion**: Averages predictions from ALL views at each step

---

## TRELLIS.2 vs Hunyuan3D Comparison

| Aspect | TRELLIS.2 (Microsoft) | Hunyuan3D 2.5/3.0 (Tencent) |
|--------|----------------------|----------------------------|
| **Core Approach** | Direct 3D latent diffusion | Multi-view diffusion → 3D reconstruction |
| **Pipeline** | Image → Sparse Structure → Shape → Texture → Mesh | Image → Multi-view images → LRM → Mesh |
| **3D Representation** | FlexiDualGrid sparse voxels | Triplane NeRF / Direct mesh |
| **Diffusion Type** | Flow matching (continuous) | DDPM/DDIM (discrete) |
| **Image Encoder** | DINOv2/DINOv3 | CLIP + custom encoders |
| **Multi-view** | Inference-time aggregation (tuning-free) | Native multi-view diffusion (trained) |

---

## Azure ML Deployment

### Compute Instance
- **Name**: MooseAML6
- **SKU**: Standard_NC24ads_A100_v4 (A100 80GB)
- **SSH**: Enabled

### File Location
```
/mnt/batch/tasks/shared/LS_root/mounts/clusters/mooseaml6/code/Users/heath.saber/TRELLIS.2/
```

Access via: Azure ML Studio → Compute → mooseaml6 → JupyterLab

---

## Test Assets
- `assets/cyborg_rat/` - 4 views (front, back, left, right) for testing

---

## Next Steps / Ideas

1. **Higher quality runs**: Increase `steps` to 20-25, try `guidance_strength` 4.0-5.0
2. **More views**: Try 6-8 views for complex objects
3. **Fine-tuning**: Train on specific object categories for better results
4. **Hunyuan3D techniques**: Investigate multi-view consistency mechanisms for potential adaptation

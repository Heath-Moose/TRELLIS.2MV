# TRELLIS.2 Multi-View Implementation Status

## Goal
Add multi-view image input capability to Microsoft TRELLIS.2 (4B) for improved 3D generation. Port the tuning-free multi-view approach from TRELLIS v1 to v2.

---

## What's Been Completed

### 1. Multi-View Pipeline Code (DONE)
All code has been written and is in your local repo:

- **`trellis2/pipelines/trellis2_multiview.py`** - New pipeline class with:
  - `Trellis2MultiViewPipeline` extending the base pipeline
  - `inject_sampler_multi_image()` context manager for patching samplers
  - `get_cond_multi()` for extracting features from multiple images
  - `run_multi_image(images, mode='stochastic')` main method
  - Supports both `stochastic` and `multidiffusion` aggregation modes

- **`trellis2/pipelines/__init__.py`** - Updated to register the new pipeline

- **`example_multiview.py`** - CLI test script for multi-view generation

- **`runpod_setup.sh`** - Setup script (heredoc-based) for RunPod deployment

- **`deploy_to_runpod.sh`** - SCP-based deployment script (doesn't work with RunPod SSH proxy)

### 2. RunPod Environment Fixes Applied
These fixes were applied during the session:

```bash
# PyTorch reinstall for CUDA 12.4
pip install torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Rebuild cumesh with correct PyTorch
pip install --no-cache-dir --no-build-isolation git+https://github.com/JeffreyXiang/cumesh.git

# Rebuild o_voxel
pip install --no-cache-dir --no-build-isolation git+https://github.com/JeffreyXiang/o_voxel.git

# HuggingFace token for gated DINOv3 model
export HF_TOKEN=<your_huggingface_token>
```

---

## Current Blockers

### 1. BiRefNet/rembg Error
```
Tensor.item() cannot be called on meta tensors
```
- Occurs during background removal model initialization
- PyTorch 2.6 compatibility issue with BiRefNet
- **Workaround attempted**: Monkey-patch BiRefNet with dummy class (untested)

### 2. flash-attn Not Installed
```
ModuleNotFoundError: No module named 'flash_attn'
```
- Needs compilation which requires psutil first
- **Fix**:
  ```bash
  pip install psutil
  pip install flash-attn --no-build-isolation
  ```

### 3. SSH Connection Issues
- RunPod SSH proxy doesn't support scp/sftp
- Connection was refused at last attempt (pod may have stopped)

---

## Next Steps (On RunPod)

### Step 1: Environment Setup
```bash
# Activate conda
eval "$(/workspace/miniconda3/bin/conda shell.bash hook)"
conda activate trellis2
export CUDA_HOME=/usr/local/cuda-12.4
export HF_TOKEN=<your_huggingface_token>
cd /workspace/TRELLIS.2
```

### Step 2: Fix Dependencies
```bash
# Install flash-attn
pip install psutil
pip install flash-attn --no-build-isolation

# If flash-attn fails, try xformers instead
pip install xformers
```

### Step 3: Sync Multi-View Code
Copy the files from your local machine to RunPod. Either:
- Use RunPod web terminal file upload
- Or paste the content of `runpod_setup.sh` into terminal
- Or use `git` if you commit the changes

Files to sync:
- `trellis2/pipelines/trellis2_multiview.py`
- `trellis2/pipelines/__init__.py`
- `example_multiview.py`

### Step 4: Test Single-Image Baseline
```bash
python example.py
```
If BiRefNet error occurs, use pre-processed images (RGBA with transparent background).

### Step 5: Test Multi-View
```bash
# Test with same image twice (sanity check)
python example_multiview.py --images assets/example_image/T.png assets/example_image/T.png --mode stochastic

# Test with actual different views
python example_multiview.py --images front.png back.png --mode multidiffusion
```

---

## How Multi-View Works

### Stochastic Mode
- Cycles through views sequentially during denoising
- Step 0 uses view 0, step 1 uses view 1, etc. (wraps around)
- Faster, good for many views

### Multidiffusion Mode
- Averages predictions from ALL views at each step
- Higher quality, slower (N forward passes per step)
- Better for 2-4 views

### Architecture
```
Input: N images
    ↓
DINOv3 Feature Extraction (per image)
    ↓
Stacked conditioning tensors [N, C, H, W]
    ↓
inject_sampler_multi_image() patches the sampler to:
  - stochastic: select one view per step
  - multidiffusion: average all view predictions
    ↓
3-Stage Generation:
  1. Sparse Structure (32³ or 64³ grid)
  2. Shape Latent (geometry)
  3. Texture Latent (PBR materials)
    ↓
Output: MeshWithVoxel → GLB
```

---

## Files Reference

| File | Status | Description |
|------|--------|-------------|
| `trellis2/pipelines/trellis2_multiview.py` | ✅ Created | Multi-view pipeline |
| `trellis2/pipelines/__init__.py` | ✅ Modified | Added pipeline registration |
| `example_multiview.py` | ✅ Created | CLI test script |
| `runpod_setup.sh` | ✅ Created | Heredoc setup script |
| `deploy_to_runpod.sh` | ✅ Created | SCP deploy script |

---

## Installing Claude Code on RunPod

```bash
# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

# Install Claude Code
npm install -g @anthropic-ai/claude-code

# Run it
cd /workspace/TRELLIS.2
claude
```

Then tell Claude: "Read MULTIVIEW_STATUS.md and continue fixing TRELLIS.2"

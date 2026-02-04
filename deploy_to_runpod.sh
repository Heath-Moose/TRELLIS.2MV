#!/bin/bash
# TRELLIS.2 Multi-View Deployment Script for RunPod
# Usage: ./deploy_to_runpod.sh <runpod_ssh_host>
# Example: ./deploy_to_runpod.sh root@123.456.789.10

set -e

if [ -z "$1" ]; then
    echo "Usage: ./deploy_to_runpod.sh <runpod_ssh_host>"
    echo "Example: ./deploy_to_runpod.sh root@ssh.runpod.io -p 12345"
    exit 1
fi

RUNPOD_HOST="$1"
shift  # Remove first arg, rest are SSH options (like -p port)
SSH_OPTS="$@"

echo "=== Deploying TRELLIS.2 Multi-View to RunPod ==="
echo "Host: $RUNPOD_HOST"
echo ""

# Files to sync
FILES=(
    "trellis2/pipelines/trellis2_multiview.py"
    "trellis2/pipelines/__init__.py"
    "example_multiview.py"
)

echo "Syncing files..."
for f in "${FILES[@]}"; do
    echo "  -> $f"
    scp $SSH_OPTS "$f" "$RUNPOD_HOST:/workspace/TRELLIS.2/$f"
done

echo ""
echo "=== Files synced successfully ==="
echo ""
echo "Now SSH into RunPod and run the setup commands:"
echo ""
cat << 'COMMANDS'
# 1. Connect to RunPod
ssh <your-runpod-connection>

# 2. Activate environment
eval "$(/workspace/miniconda3/bin/conda shell.bash hook)"
conda activate trellis2
export CUDA_HOME=/usr/local/cuda-12.4
cd /workspace/TRELLIS.2

# 3. Fix cumesh (if not already done)
export NVCC_PREPEND_FLAGS="--extended-lambda"
pip install git+https://github.com/JeffreyXiang/cumesh.git

# 4. Test single-image baseline
python example.py

# 5. Test multi-view (after single-image works)
# Using same image twice for initial test:
python example_multiview.py --images assets/example_image/T.png assets/example_image/T.png --mode stochastic

# 6. Test with different views (if you have them)
# python example_multiview.py --front front.png --back back.png --mode multidiffusion
COMMANDS

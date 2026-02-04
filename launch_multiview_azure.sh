#!/bin/bash
# Launch TRELLIS.2 Multi-View Gradio UI on Azure ML
#
# Usage from Azure ML JupyterLab terminal:
#   cd /mnt/batch/tasks/shared/LS_root/mounts/clusters/mooseaml6/code/Users/heath.saber/TRELLIS.2
#   ./launch_multiview_azure.sh
#
# Or with public link:
#   ./launch_multiview_azure.sh --share

set -e

# Navigate to project directory (Azure ML path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== TRELLIS.2 Multi-View Gradio UI ==="
echo "Working directory: $(pwd)"
echo ""

# Activate conda environment if needed
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo ""
fi

# Default port for Azure ML
PORT=${PORT:-7860}

echo "Starting Gradio app..."
echo ""
echo "============================================"
echo "ACCESS OPTIONS:"
echo ""
echo "1. From Azure ML JupyterLab:"
echo "   Open a new browser tab and go to:"
echo "   https://mooseaml6-${PORT}.westus3.instances.azureml.ms/"
echo ""
echo "2. Via SSH tunnel (from your local machine):"
echo "   ssh -L ${PORT}:localhost:${PORT} azureuser@<compute-ip>"
echo "   Then open: http://localhost:${PORT}"
echo ""
echo "3. Public link (use --share flag):"
echo "   ./launch_multiview_azure.sh --share"
echo "============================================"
echo ""

# Run the app
python app_multiview.py \
    --server-name 0.0.0.0 \
    --port $PORT \
    "$@"

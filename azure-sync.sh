#!/bin/bash
# Sync local changes to Azure ML (MooseAML6)
# Usage: ./azure-sync.sh

set -e

AZURE_HOST="azureuser@20.92.74.159"
AZURE_PORT="50000"
AZURE_PATH="/mnt/batch/tasks/shared/LS_root/mounts/clusters/mooseaml6/code/Users/heath.saber/TRELLIS.2"
GITHUB_REPO="https://github.com/Heath-Moose/TRELLIS.2MV.git"

echo "=== Azure Sync ==="
echo ""

# Check for uncommitted changes (ignore untracked files)
if [[ -n $(git status -s | grep -v '^??') ]]; then
    echo "ERROR: Uncommitted changes detected."
    echo "Please commit your changes first:"
    echo "  git add -A && git commit -m 'your message'"
    exit 1
fi

# Push to GitHub
echo "[1/2] Pushing to GitHub..."
git push mv main
echo "      Done."
echo ""

# Pull on Azure
echo "[2/2] Pulling on Azure ML..."
ssh -o StrictHostKeyChecking=no -p "$AZURE_PORT" "$AZURE_HOST" \
    "cd $AZURE_PATH && \
     git config --global --add safe.directory $AZURE_PATH 2>/dev/null || true && \
     git fetch $GITHUB_REPO main && \
     git reset --hard FETCH_HEAD"
echo ""

echo "=== Sync Complete! ==="
echo ""
echo "To run the app on Azure:"
echo "  ssh -p $AZURE_PORT $AZURE_HOST"
echo "  cd $AZURE_PATH"
echo "  python app_multiview.py --port 7860"

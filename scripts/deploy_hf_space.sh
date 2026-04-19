#!/usr/bin/env bash
# Deploy gridsense-az to HuggingFace Space.
# Copies: app.py entry, src/gridsense/, data/ieee123/, requirements, README.
# Requires: HF_TOKEN in env, HF_SPACE_REPO (e.g. dc-ai-labs/gridsense-az).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_SPACE_REPO="${HF_SPACE_REPO:-dc-ai-labs/gridsense-az}"
HF_STAGE="${REPO_ROOT}/hf_space_stage"

echo "[hf-deploy] staging at ${HF_STAGE}"
rm -rf "${HF_STAGE}"
mkdir -p "${HF_STAGE}/src/gridsense" "${HF_STAGE}/data/ieee123" "${HF_STAGE}/app/components"

# Entry point (hf_space/app.py delegates to the real app via path insertion)
cat > "${HF_STAGE}/app.py" <<'PYEOF'
"""HuggingFace Space entrypoint. Delegates to the real dashboard."""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

# Run the streamlit app module
from app.streamlit_app import main  # noqa: E402
main()
PYEOF

# Pull in source + assets
cp "${REPO_ROOT}/src/gridsense/"*.py "${HF_STAGE}/src/gridsense/"
touch "${HF_STAGE}/src/__init__.py"
cp "${REPO_ROOT}/app/streamlit_app.py" "${HF_STAGE}/app/"
cp "${REPO_ROOT}/app/__init__.py" "${HF_STAGE}/app/"
cp "${REPO_ROOT}/app/components/"*.py "${HF_STAGE}/app/components/" 2>/dev/null || true
cp -r "${REPO_ROOT}/data/ieee123/"* "${HF_STAGE}/data/ieee123/"

# HF Space metadata
cp "${REPO_ROOT}/hf_space/README.md" "${HF_STAGE}/README.md"
cp "${REPO_ROOT}/hf_space/requirements.txt" "${HF_STAGE}/requirements.txt"
cp "${REPO_ROOT}/hf_space/Dockerfile" "${HF_STAGE}/Dockerfile"

# Optional: copy trained model if it exists
if [[ -f "${REPO_ROOT}/data/models/gwnet_v0.pt" ]]; then
    mkdir -p "${HF_STAGE}/data/models"
    cp "${REPO_ROOT}/data/models/"{gwnet_v0.pt,metrics.json,history.json} "${HF_STAGE}/data/models/" 2>/dev/null || true
    echo "[hf-deploy] bundled trained model"
fi

echo "[hf-deploy] staged $(find "${HF_STAGE}" -type f | wc -l) files, $(du -sh "${HF_STAGE}" | cut -f1)"
echo "[hf-deploy] to push:"
echo "  cd ${HF_STAGE} && git init && git add -A && git commit -m 'deploy' && \\"
echo "    git remote add space https://huggingface.co/spaces/${HF_SPACE_REPO} && \\"
echo "    git push --force space main"

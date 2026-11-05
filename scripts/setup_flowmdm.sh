#!/usr/bin/env bash
# Clones upstream FlowMDM at the pinned commit and creates the Python environment.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLOWMDM_DIR="${ROOT_DIR}/FlowMDM"
FLOWMDM_URL="https://github.com/BarqueroGerman/FlowMDM.git"
FLOWMDM_COMMIT="e81c67f"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if [[ ! -d "${FLOWMDM_DIR}/.git" ]]; then
  git clone "${FLOWMDM_URL}" "${FLOWMDM_DIR}"
fi
git -C "${FLOWMDM_DIR}" checkout "${FLOWMDM_COMMIT}"

uv python install "${PYTHON_VERSION}"
uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"

# chumpy needs pip/setuptools in the environment when build isolation is disabled
uv pip install --python "${PYTHON_BIN}" pip setuptools==65.5.0 wheel
uv pip install --python "${PYTHON_BIN}" --no-build-isolation chumpy==0.70
uv pip install --python "${PYTHON_BIN}" -r "${ROOT_DIR}/requirements.txt"
uv pip install --python "${PYTHON_BIN}" setuptools==65.5.0
"${PYTHON_BIN}" -m spacy download en_core_web_sm

echo "FlowMDM ${FLOWMDM_COMMIT} is at ${FLOWMDM_DIR}; environment is at ${VENV_DIR}"

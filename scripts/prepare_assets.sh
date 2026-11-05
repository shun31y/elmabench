#!/usr/bin/env bash
# Downloads SMPL and GloVe (upstream scripts), the ELMA evaluator / normalization files and
# the ELMA-trained FlowMDM checkpoints into the FlowMDM checkout.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLOWMDM_DIR="${ROOT_DIR}/FlowMDM"
GDOWN="${ROOT_DIR}/.venv/bin/gdown"

# evaluator (t2m/humanml/text_mot_match/model/finest.tar) and dataset/{t2m,HML}_{mean,std}.npy
EVALUATOR_URL="${ELMA_EVALUATOR_URL:-https://drive.google.com/file/d/1ZEHOOl3avMxgX_12ycagVj0_1oRz0r3q/view?usp=drive_link}"
# results/humanml/{FlowMDM,ELMAv3_anap_hml3d,ELMAv3_anap_hml3d_hist}/...
CHECKPOINTS_URL="${ELMA_CHECKPOINTS_URL:-https://drive.google.com/file/d/1-XqEum8HNNiMMbyiiMygIvDWylHjbpl8/view?usp=sharing}"

cd "${FLOWMDM_DIR}"
# upstream scripts: SMPL body models (needed to build the FlowMDM model) and GloVe (evaluator)
PATH="$(dirname "${GDOWN}"):${PATH}" bash runners/prepare/download_smpl_files.sh
PATH="$(dirname "${GDOWN}"):${PATH}" bash runners/prepare/download_glove.sh

for url in "${EVALUATOR_URL}" "${CHECKPOINTS_URL}"; do
  "${GDOWN}" --fuzzy "${url}" -O elma_assets.zip
  unzip -o elma_assets.zip -d .
  rm -f elma_assets.zip
done

sha256sum -c - <<'EOF'
34be6512f5137c8e5598ec0fded5392120c8a96e2108a18c586fee5cd8838b20  t2m/humanml/text_mot_match/model/finest.tar
4f4fc20dd794f1fc4c357d9a7cc4515a664a2c655566f91cffb404347dd04609  dataset/t2m_mean.npy
5cc9d91679d89be262c4090981dd2f09eca253226f8bb75be0770287abe3b15d  dataset/t2m_std.npy
cb803144a58fae9384888f4b6a4af1fbdb8e6470b8faa9215fe35bd9df22fd99  results/humanml/ELMAv3_anap_hml3d_hist/model000550000.pt
d21fa2b2f5906bc4862da10bc2dfc066c1c193c7e3a85b0ffa4dce3c4d2cf330  results/humanml/ELMAv3_anap_hml3d/model000550000.pt
EOF

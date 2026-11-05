<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/elmabench_title_dark.png">
    <img alt="ELMABench" src="assets/elmabench_title_light.png" height="54">
  </picture>
</h1>

<h3 align="center">ELMA: Benchmarking Anaphoric Compositions for Long-Term Text-to-Motion Generation</h3>

<p align="center">
  <a href="https://elma-dataset.github.io/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Page-2ea44f?logo=googlechrome&logoColor=white"></a>
  <a href="https://arxiv.org/abs/2610.13549"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2610.13549-b31b1b.svg?logo=arxiv&logoColor=white"></a>
  <a href="https://huggingface.co/datasets/shun31y/ELMA"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ELMA-ffcc4d"></a>
</p>

<br>

This repository is the official implementation of **ELMA: Benchmarking Anaphoric Compositions for Long-Term Text-to-Motion Generation**.

## 📰 News

- [2026-11] Evaluation code and the FlowMDM A2M checkpoint released.
- [2026-10] ELMA v2 released on [Hugging Face](https://huggingface.co/datasets/shun31y/ELMA), with a cleaned test set.
- [2026-10] ELMA v1.0, the data used in the paper, released on [Hugging Face](https://huggingface.co/datasets/shun31y/ELMA).
- [2026-10] Our paper (NeurIPS 2026 Evaluations and Datasets Track) is available on [arXiv](https://arxiv.org/abs/2610.13549).

## ✅ TODO List

- [x] Release the ELMA dataset.
- [x] Release the FlowMDM A2M checkpoint.
- [x] Release the A2M evaluation code.

## 🚀 Getting Started

This release evaluates FlowMDM trained on ELMA in the anaphoric text-to-motion (A2M)
setting. FlowMDM is not included here: the setup script below clones the
[official FlowMDM repository](https://github.com/BarqueroGerman/FlowMDM), and this code
imports it unmodified.

Requirements: `git`, `unzip`, [`uv`](https://docs.astral.sh/uv/), and a CUDA GPU.

1. Set up FlowMDM and the Python environment:

   ```bash
   bash scripts/setup_flowmdm.sh
   ```

   This command clones FlowMDM at commit `e81c67f`, installs Python 3.10 with `uv`,
   creates `./.venv`, installs the Python dependencies, and downloads the spaCy English
   model. The script performs all of these steps for you.

2. Prepare the assets needed for evaluation:

   ```bash
   bash scripts/prepare_assets.sh
   ```

   This command downloads the SMPL body models and GloVe using FlowMDM's scripts,
   then downloads the ELMA `v1.0` text-motion evaluator, normalization statistics,
   and FlowMDM A2M checkpoint into `./FlowMDM`. It verifies the SHA256 hashes of
   the evaluator, statistics, and checkpoint; no separate asset downloads are needed.

   These assets are for ELMA `v1.0`, the data used in the paper. To train or evaluate on
   the latest version of the dataset, you need to prepare your own evaluator,
   normalization statistics, and checkpoint.

3. Download the HumanML3D-format export of ELMA (`HML3D.tar`, ~12 GB) from
   [Hugging Face](https://huggingface.co/datasets/shun31y/ELMA):

   ```bash
   pip install -U huggingface_hub   # outside .venv, which pins an older version
   hf download shun31y/ELMA HML3D.tar --repo-type dataset --revision v1.0 --local-dir /path/to/ELMA-v1.0
   tar -xf /path/to/ELMA-v1.0/HML3D.tar -C /path/to/ELMA-v1.0
   ```

   > **ELMA v2 is now available.** v2 removes noisy motions and captions from the test set
   > (a human-reviewed subset of the v1.0 test set) and corrects the root height, so its motion
   > features and normalization statistics differ from v1.0. See the
   > [dataset page](https://huggingface.co/datasets/shun31y/ELMA) for details.

## 📊 A2M Evaluation

To evaluate in the A2M setting, run the following command:

```bash
DATA_ROOT=/path/to/ELMA-v1.0/HML3D GPU_ID=0 EVAL_MODE=final bash scripts/eval_a2m.sh
```

`EVAL_MODE` is `final` (10 replications), `fast` (5) or `debug` (1). Results are written to
`FlowMDM/results/humanml/ELMAv3_anap_hml3d_hist/elma_evaluation/` (`--output_dir` to change):
a `.log` with every replication and a `.json` with mean and 95% confidence interval of FID,
R-precision (top 1–3), matching score and diversity.

For full-generation A2M evaluation, generate all six segments (600 frames) from
`humanml_test_set_anaphora_fullgen.json` and score only the segment marked as anaphoric
in each sample ID:

```bash
DATA_ROOT=/path/to/ELMA-v1.0/HML3D GPU_ID=0 EVAL_MODE=debug bash scripts/eval_a2m.sh --scenario anaphora_fullgen
```

The other segments provide generated motion context. Their captions enter the same
history-text conditioning as above. The default scenario, `anaphora`, generates only
the 100-frame target segment.

> **Note:** Full-generation evaluation takes substantially longer than target-only
> evaluation because it generates all 600 frames for every sample. The `fast` and `final`
> modes repeat this generation 5 and 10 times, respectively.

## 🖊️ Citation

If you find ELMA useful for your research, please cite:

```bibtex
@inproceedings{kato2026elma,
  title     = {{ELMA}: Benchmarking Anaphoric Compositions for Long-Term Text-to-Motion Generation},
  author    = {Kato, Shun and Yang, Dixin and Amaya, Kotaro and Wada, Yuiga and Fukuda, Kazumi and
               Shibuya, Takashi and Mitsufuji, Yuki and Isogawa, Mariko},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026},
  eprint    = {2610.13549},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

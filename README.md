# ELMABench

Evaluation code for [ELMA: Benchmarking Anaphoric Compositions for Long-Term Text-to-Motion Generation](https://arxiv.org/abs/2610.13549).

This repository evaluates FlowMDM checkpoints trained on ELMA under the anaphoric text-to-motion
(A2M) setting. It does **not** contain FlowMDM: you clone the official
[FlowMDM repository](https://github.com/BarqueroGerman/FlowMDM) yourself (pinned to commit
`e81c67f`), and the code here imports it unmodified.

## Setup

Requirements: `git`, `unzip`, [`uv`](https://docs.astral.sh/uv/), a CUDA GPU.

1. Clone upstream FlowMDM into `./FlowMDM` and create `./.venv`:

   ```bash
   bash scripts/setup_flowmdm.sh
   ```

2. Download the assets into `./FlowMDM`:

   ```bash
   bash scripts/prepare_assets.sh
   ```

   - SMPL body models and GloVe, with FlowMDM's own download scripts
   - the ELMA text-motion evaluator and its normalization statistics
     (`t2m/humanml/text_mot_match/model/finest.tar`, `dataset/t2m_{mean,std}.npy`)
   - the ELMA-trained FlowMDM checkpoints (`results/humanml/ELMAv3_anap_hml3d{,_hist}/model000550000.pt`)

   The script checks the SHA256 of the evaluator, the statistics and the checkpoints.

3. Download the HumanML3D-format export of ELMA (`HML3D.tar`, ~12 GB) from
   [Hugging Face](https://huggingface.co/datasets/shun31y/ELMA). The checkpoints and the
   evaluator statistics above correspond to `v1.0`, the data used in the paper.

   ```bash
   pip install -U huggingface_hub   # outside .venv, which pins an older version
   hf download shun31y/ELMA HML3D.tar --repo-type dataset --revision v1.0 --local-dir /path/to/ELMA-v1.0
   tar -xf /path/to/ELMA-v1.0/HML3D.tar -C /path/to/ELMA-v1.0
   ```

   > **ELMA v2 is now available.** v2 removes noisy motions and captions from the test set
   > (a human-reviewed subset of the v1.0 test set) and corrects the root height, so its motion
   > features and normalization statistics differ from v1.0. See the
   > [dataset page](https://huggingface.co/datasets/shun31y/ELMA) for details.

## A2M evaluation

A2M → A2M: the history-text checkpoint, conditioned on each segment's caption mixed with the
captions before it (`0.5 * current + 0.5 * mean(previous)` CLIP embeddings), evaluated on
`test_anaphora`:

```bash
DATA_ROOT=/path/to/ELMA-v1.0/HML3D GPU_ID=0 EVAL_MODE=final bash scripts/eval_a2m.sh
```

`EVAL_MODE` is `final` (10 replications), `fast` (5) or `debug` (1). Results are written to
`FlowMDM/results/humanml/ELMAv3_anap_hml3d_hist/elma_evaluation/` (`--output_dir` to change):
a `.log` with every replication and a `.json` with mean and 95% confidence interval of FID,
R-precision (top 1–3), matching score and diversity.

## Repository layout

- `elmabench/evaluate.py`: evaluation entry point
- `elmabench/sampler.py`: batched multi-segment sampling with an upstream FlowMDM model
- `elmabench/history.py`: history-text conditioning
- `elmabench/gt_data.py`, `elmabench/generated_data.py`: GT and generated evaluation sets
- `elmabench/upstream.py`: checks and imports the FlowMDM checkout
- `scripts/`: environment setup, asset download and the A2M evaluation wrapper

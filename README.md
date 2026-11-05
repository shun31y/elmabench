# ELMABench

Evaluation code for [ELMA: Benchmarking Anaphoric Compositions for Long-Term Text-to-Motion Generation](https://arxiv.org/abs/2610.13549).

This repository evaluates FlowMDM checkpoints trained on ELMA under the anaphoric text-to-motion
(A2M) setting. It does **not** contain FlowMDM: you clone the official
[FlowMDM repository](https://github.com/BarqueroGerman/FlowMDM) yourself (pinned to commit
`e81c67f`), and the code here imports it unmodified. Your use of FlowMDM is governed by its own
license (academic / non-commercial research use only).

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
   [Hugging Face](https://huggingface.co/datasets/shun31y/ELMA). `v1.0` is the data used in the
   paper; `v1.1` revises captions of the anaphora test set (motions are identical).

   ```bash
   pip install -U huggingface_hub   # outside .venv, which pins an older version
   hf download shun31y/ELMA HML3D.tar --repo-type dataset --revision v1.1 --local-dir /path/to/ELMA-v1.1
   tar -xf /path/to/ELMA-v1.1/HML3D.tar -C /path/to/ELMA-v1.1
   ```

## A2M evaluation

A2M → A2M: the history-text checkpoint, conditioned on each segment's caption mixed with the
captions before it (`0.5 * current + 0.5 * mean(previous)` CLIP embeddings), evaluated on
`test_anaphora`:

```bash
DATA_ROOT=/path/to/ELMA-v1.1/HML3D GPU_ID=0 EVAL_MODE=final bash scripts/eval_a2m.sh
```

`EVAL_MODE` is `final` (10 replications), `fast` (5) or `debug` (1). Results are written to
`FlowMDM/results/humanml/ELMAv3_anap_hml3d_hist/elma_evaluation/` (`--output_dir` to change):
a `.log` with every replication and a `.json` with mean and 95% confidence interval of FID,
R-precision (top 1–3), matching score and diversity.

## T2M evaluation

T2M → T2M: the checkpoint trained without history text, conditioned only on each segment's
own caption, on the same `test_anaphora` samples:

```bash
DATA_ROOT=/path/to/ELMA-v1.1/HML3D GPU_ID=0 EVAL_MODE=final bash scripts/eval_t2m.sh
```

Results go to `FlowMDM/results/humanml/ELMAv3_anap_hml3d/elma_evaluation/`. Other
checkpoint / conditioning combinations (e.g. T2M → A2M: the T2M checkpoint with
`--use_history_text --history_current_weight 0.5`) go through `python -m elmabench.evaluate`
directly; see `--help`.

### Protocol

- **Generation**: each entry of `HML3D/humanml_test_set_anaphora.json` is one sample with six
  segments; only the target segment (`__ana_segNN_` in the id) has a non-zero length. Samples
  are generated in batches of 128 with classifier-free guidance 2.5 and 1000 denoising steps.
- **Generated set**: the target segment of each sample (1000 segments).
- **Ground truth**: all annotated segments of the `test` split between 70 and 199 frames
  (18,000 segments), as in FlowMDM's HumanML3D motion evaluation.
- **Metrics**: T2M evaluator embeddings; R-precision within batches of 32; seed 10.

## Repository layout

- `elmabench/evaluate.py`: evaluation entry point
- `elmabench/sampler.py`: batched multi-segment sampling with an upstream FlowMDM model
- `elmabench/history.py`: text conditioning (per-segment captions, history-text mix)
- `elmabench/gt_data.py`, `elmabench/generated_data.py`: GT and generated evaluation sets
- `elmabench/upstream.py`: checks and imports the FlowMDM checkout
- `scripts/`: environment setup, asset download and the A2M / T2M evaluation wrappers

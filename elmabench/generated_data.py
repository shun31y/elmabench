"""ELMA evaluation samples (humanml_test_set_anaphora.json) and the generated target segments.

Each sample id ends with __ana_segNN_XXXX; only segment NN (1-based) is evaluated. The other
segments are context: they have length 0 and only their captions enter the history-text
conditioning.
"""
import json
import re

import numpy as np
from torch.utils.data import Dataset

from elmabench.gt_data import MAX_LEN, MAX_TEXT_TOKENS
from elmabench.sampler import sample as sample_flowmdm

TARGET_RE = re.compile(r"__ana_seg(\d+)_")


def load_eval_samples(eval_file, scenario):
    with open(eval_file) as f:
        raw = json.load(f)
    samples = []
    for entry in raw:
        if scenario and entry.get("scenario") not in (None, scenario):
            continue
        match = TARGET_RE.search(entry["id"])
        if match is None:
            raise ValueError(f"cannot find the target segment in sample id {entry['id']}")
        lengths = [int(v) for v in entry["lengths"]]
        target = int(match.group(1)) - 1
        if not 0 <= target < len(lengths) or lengths[target] <= 0:
            raise ValueError(f"{entry['id']}: invalid target segment {target + 1} for lengths {lengths}")
        if len(entry["text"]) != len(lengths):
            raise ValueError(f"{entry['id']}: {len(entry['text'])} captions for {len(lengths)} segments")
        samples.append({
            "id": entry["id"],
            "texts": entry["text"],
            "lengths": lengths,
            "histories": entry.get("history_text"),
            "target": target,
        })
    if not samples:
        raise ValueError(f"no samples for scenario {scenario!r} in {eval_file}")
    return samples


def generate_targets(model, diffusion, samples, guidance_param, history_weight, batch_size):
    """Generates every sample and keeps its target segment: list of (features [T, D], caption)."""
    if history_weight is not None and any(s["histories"] is None for s in samples):
        raise ValueError("history-text conditioning needs history_text in the evaluation file")
    targets = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        motions = sample_flowmdm(model, diffusion, chunk, guidance_param, history_weight)
        for b, s in enumerate(chunk):
            offset = sum(s["lengths"][:s["target"]])
            segment = motions[b, :, 0, offset:offset + s["lengths"][s["target"]]]
            targets.append((segment.permute(1, 0).cpu().numpy(), s["texts"][s["target"]]))
        print(f"generated {min(start + batch_size, len(samples))}/{len(samples)}", flush=True)
    return targets


class GeneratedSegments(Dataset):
    """Generated segments, moved from the model's normalization to the evaluator's."""

    def __init__(self, targets, model_mean, model_std, eval_mean, eval_std, w_vectorizer, process_tokens, get_tokens):
        self.items = []
        for motion, caption in targets:
            length = len(motion)
            # zero-pad in model space (float64), as the evaluator protocol does, then renormalize
            # the whole padded window
            motion = np.concatenate([motion, np.zeros((MAX_LEN - length, motion.shape[1]))])
            motion = ((motion * model_std + model_mean) - eval_mean) / eval_std
            self.items.append((motion, length, caption))
        self.w_vectorizer = w_vectorizer
        self.process_tokens = process_tokens
        self.get_tokens = get_tokens

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        motion, length, caption = self.items[item]
        word_emb, pos_ohot, sent_len, tokens = self.process_tokens(self.get_tokens(caption), MAX_TEXT_TOKENS, self.w_vectorizer)
        return word_emb, pos_ohot, caption, sent_len, motion, length, tokens

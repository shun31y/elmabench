"""Batched multi-segment sampling with an upstream FlowMDM model.

Each batch element is one ELMA sample: a sequence of segments, each with its own text
condition. Upstream FlowMDM reads the per-frame conditioning from model_kwargs['y'], so all
we build here are those tensors; the model and the diffusion step are used unmodified.
"""
import torch

from elmabench.history import segment_embeddings


def build_conditioning(samples, guidance_param, clip_model, history_weight, device):
    seg_lengths = [s["lengths"] for s in samples]
    totals = [sum(lengths) for lengths in seg_lengths]
    bs, nframes = len(samples), max(totals)
    max_segments = max(len(lengths) for lengths in seg_lengths)

    mask = torch.zeros((bs, nframes), dtype=torch.bool, device=device)
    conditions_mask = torch.zeros((max_segments, nframes, bs), dtype=torch.bool, device=device)
    pos_pe_abs = torch.zeros((bs, nframes), dtype=torch.float32, device=device)
    # attention is restricted to each segment for the absolute-position part of FlowMDM's BPE
    pe_bias = torch.full((bs, nframes, nframes), float("-inf"), dtype=torch.float32, device=device)
    for b, lengths in enumerate(seg_lengths):
        mask[b, :totals[b]] = True
        offset = 0
        for k, length in enumerate(lengths):
            segment = slice(offset, offset + length)
            pos_pe_abs[b, segment] = torch.arange(length, dtype=torch.float32, device=device)
            pe_bias[b, segment, segment] = 0
            conditions_mask[k, segment, b] = True
            offset += length

    all_texts = [list(s["texts"]) for s in samples]
    y = {
        "all_texts": all_texts,
        "all_lengths": seg_lengths,
        "text": [" -- ".join(texts) for texts in all_texts],
        "mask": mask,
        "lengths": torch.tensor(totals, dtype=torch.int64, device=device),
        "conditions_mask": conditions_mask,
        "pos_pe_abs": pos_pe_abs,
        "pe_bias": pe_bias,
        "scale": torch.ones(bs, device=device) * guidance_param,
    }
    # always precomputed: the ELMA checkpoints use 24-token captions, upstream encode_text uses 20
    y["text_embeddings"] = segment_embeddings(clip_model, samples, history_weight)
    return {"y": y}, nframes


@torch.no_grad()
def sample(model, diffusion, samples, guidance_param, history_weight=None):
    """Returns generated features [bs, njoints, nfeats, max_total_length] (model normalization)."""
    device = next(model.parameters()).device
    model_kwargs, nframes = build_conditioning(samples, guidance_param, model.clip_model, history_weight, device)
    x = torch.randn(len(samples), model.njoints, model.nfeats, nframes, device=device)
    for step in reversed(range(diffusion.num_timesteps)):
        t = torch.full((len(samples),), step, dtype=torch.long, device=device)
        x = diffusion.p_sample(model, x, t, model_kwargs=model_kwargs, clip_denoised=False)["sample"]
    return x

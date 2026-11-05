"""A2M text conditioning: each segment is conditioned on a mix of its own caption and the
captions of the segments before it.

    e_k = w * clip(c_k) + (1 - w) * mean(clip(c_1), ..., clip(c_{k-1}))      (e_1 = clip(c_1))

The ELMA checkpoints were trained with HumanML3D captions truncated to 24 CLIP tokens.
"""
import clip
import torch

MAX_TEXT_TOKENS = 24
CLIP_CONTEXT_LENGTH = 77


def clip_encode(clip_model, texts):
    device = next(clip_model.parameters()).device
    tokens = clip.tokenize(texts, context_length=MAX_TEXT_TOKENS + 2, truncate=True).to(device)
    pad = torch.zeros((tokens.shape[0], CLIP_CONTEXT_LENGTH - tokens.shape[1]), dtype=tokens.dtype, device=device)
    return clip_model.encode_text(torch.cat([tokens, pad], dim=1)).float()


def history_embedding(clip_model, history, current_weight):
    if not history:
        raise ValueError("history must contain at least the current caption")
    encoded = clip_encode(clip_model, list(history))
    if len(history) == 1:
        return encoded[-1]
    return current_weight * encoded[-1] + (1.0 - current_weight) * encoded[:-1].mean(dim=0)


def segment_embeddings(clip_model, samples, current_weight):
    """[max_segments, batch, clip_dim] embeddings for FlowMDM's multi-text conditioning."""
    device = next(clip_model.parameters()).device
    max_segments = max(len(s["texts"]) for s in samples)
    out = torch.zeros((max_segments, len(samples), clip_model.text_projection.shape[1]),
                      device=device, dtype=torch.float32)
    for b, sample in enumerate(samples):
        if len(sample["histories"]) != len(sample["texts"]):
            raise ValueError(f"{sample['id']}: {len(sample['histories'])} histories for {len(sample['texts'])} segments")
        for k, (text, history) in enumerate(zip(sample["texts"], sample["histories"])):
            if history[-1] != text:
                raise ValueError(f"{sample['id']} segment {k}: history must end with the segment caption")
            out[k, b] = history_embedding(clip_model, history, current_weight)
    return out

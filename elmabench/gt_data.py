"""Ground-truth segments of the ELMA test split, in the T2M evaluator's input format.

Every annotated segment (caption#tokens#start#end) of a test episode becomes one GT sample
when it is between MIN_LEN and MAX_LEN frames long. Sampling at __getitem__ time follows
the HumanML3D text-to-motion evaluation protocol (random caption, random crop of the
padded length), and items are ordered by length, so batches can be compared run to run.
"""
import os
import random

import numpy as np
from torch.utils.data import Dataset

FPS = 20
MIN_LEN, MAX_LEN = 70, 200
MAX_TEXT_TOKENS = 24


def read_text_file(path):
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split("#")
            start, end = float(fields[2]), float(fields[3])
            entries.append({
                "caption": fields[0],
                "tokens": fields[1].split(" "),
                "start": 0.0 if np.isnan(start) else start,
                "end": 0.0 if np.isnan(end) else end,
            })
    return entries


def load_segments(data_root, split="test"):
    with open(os.path.join(data_root, f"{split}.txt")) as f:
        ids = [line.strip() for line in f]
    segments = []
    for episode in ids:
        motion = np.load(os.path.join(data_root, "new_joint_vecs", episode + ".npy"))
        if len(motion) < MIN_LEN:
            continue
        for entry in read_text_file(os.path.join(data_root, "texts", episode + ".txt")):
            if entry["start"] == 0.0 and entry["end"] == 0.0:
                if len(motion) >= MAX_LEN:  # whole-episode captions only for short episodes
                    continue
                raise NotImplementedError(f"{episode}: whole-motion captions on short episodes are not used by ELMA")
            clip = motion[int(entry["start"] * FPS):int(entry["end"] * FPS)]
            if MIN_LEN <= len(clip) < MAX_LEN:
                segments.append({"motion": clip, "length": len(clip), "texts": [entry]})
    segments.sort(key=lambda s: s["length"])  # stable: ties keep file order
    return segments


class GTSegments(Dataset):
    def __init__(self, segments, mean, std, w_vectorizer, process_tokens):
        self.segments = segments
        self.mean, self.std = mean, std
        self.w_vectorizer = w_vectorizer
        self.process_tokens = process_tokens

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, item):
        segment = self.segments[item]
        text = random.choice(segment["texts"])
        word_emb, pos_ohot, sent_len, tokens = self.process_tokens(text["tokens"], MAX_TEXT_TOKENS, self.w_vectorizer)
        length = max(segment["length"], MIN_LEN)
        start = random.randint(0, len(segment["motion"]) - length)
        motion = (segment["motion"][start:start + length] - self.mean) / self.std
        if length < MAX_LEN:
            motion = np.concatenate([motion, np.zeros((MAX_LEN - length, motion.shape[1]))], axis=0)
        return word_emb, pos_ohot, text["caption"], sent_len, motion, length, tokens


def load_target_segments(data_root, samples):
    """The real target segment of each evaluation sample, for a ground-truth reference row.

    Timestamps come from the sample's own text file; the motion from its episode (the
    per-sample __ana motion files are copies of the episode and some are missing).
    """
    targets = []
    for s in samples:
        entry = read_text_file(os.path.join(data_root, "texts", s["source_id"] + ".txt"))[s["target"]]
        caption = s["texts"][s["target"]]
        if entry["caption"] != caption:
            raise ValueError(f"{s['id']}: text file caption {entry['caption']!r} != evaluation file {caption!r}")
        motion = np.load(os.path.join(data_root, "new_joint_vecs", s["source_id"].split("__")[0] + ".npy"))
        targets.append((motion[int(entry["start"] * FPS):int(entry["end"] * FPS)], caption))
    return targets


class GTTargets(Dataset):
    """Real target segments scored like generated ones (captions tokenized the same way)."""

    def __init__(self, targets, mean, std, w_vectorizer, process_tokens, get_tokens):
        self.items = []
        for motion, caption in targets:
            length = len(motion)
            motion = (motion - mean) / std
            motion = np.concatenate([motion, np.zeros((MAX_LEN - length, motion.shape[1]))], axis=0)
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

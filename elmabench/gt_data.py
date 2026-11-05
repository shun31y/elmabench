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

#!/usr/bin/env python3
"""
Simplified ArtEmis preprocessing pipeline for assignments.

✔ Always loads artemis_dataset.csv
✔ Always subsamples paintings to TARGET_SUBSAMPLE
✔ Always copies images
✔ Always resizes images to 224×224
✔ Cleans + subtokenizes text
✔ Builds vocabulary
✔ Saves train/val/test CSVs
"""

import os
import os.path as osp
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
import pickle
import json
import re
from math import floor
from PIL import Image

# ---------------------------------------------------------
# FIXED CONFIGURATION
# ---------------------------------------------------------
RAW_CSV = "artemis_dataset.csv"
OUT_DIR = "data_preprocessed"
IMAGES_OUT = osp.join(OUT_DIR, "images_subset")
WIKI_ROOT = "wikiart"                    # path containing style folders + .jpg files
TARGET_SUBSAMPLE = 7500
SEED = 42

MAX_LEN = 25
MIN_WORD_FREQ = 2
SPLIT_LOADS = (0.8, 0.1, 0.1)

# ---------------------------------------------------------
# TEXT CLEANING + SUBWORD TOKENIZATION
# ---------------------------------------------------------

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def subtokenize_word(w: str):
    suffixes = ["ing", "ed", "ly", "ness", "ment", "ful", "less", "able", "ify", "ation", "s"]
    for suf in suffixes:
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return [w[:-len(suf)], suf]
    return [w]


def tokenize(s: str):
    tokens = clean_text(s).split()
    final = []
    for t in tokens:
        final.extend(subtokenize_word(t))
    return final


# ---------------------------------------------------------
# VOCAB CLASS
# ---------------------------------------------------------

class Vocab:
    PAD = "<pad>"
    START = "<start>"
    END = "<end>"
    UNK = "<unk>"

    def __init__(self, token_to_idx):
        self.token_to_idx = token_to_idx
        self.idx_to_token = {i: t for t, i in token_to_idx.items()}

    @classmethod
    def build(cls, counter: Counter, min_freq=1):
        specials = [cls.PAD, cls.START, cls.END, cls.UNK]
        token_to_idx = {t: i for i, t in enumerate(specials)}

        items = [(t, c) for t, c in counter.items() if c >= min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))

        for i, (t, _) in enumerate(items, start=len(specials)):
            token_to_idx[t] = i

        return cls(token_to_idx)

    def __len__(self):
        return len(self.token_to_idx)

    def encode(self, tokens, max_len):
        ids = [self.token_to_idx.get(t, self.token_to_idx[self.UNK]) for t in tokens]
        ids = [self.token_to_idx[self.START]] + ids[:max_len - 2] + [self.token_to_idx[self.END]]
        ids += [self.token_to_idx[self.PAD]] * (max_len - len(ids))
        return ids

    # Optional: allow vocab[token] syntax
    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.token_to_idx[self.UNK])

    # Optional: provide a property for backward compatibility
    @property
    def stoi(self):
        return self.token_to_idx


# ---------------------------------------------------------
# STRATIFIED SAMPLING
# ---------------------------------------------------------

def stratified_subsample_by_style(df, target_n, seed=SEED):
    style_to_paintings = df.groupby('art_style')['painting'].unique().to_dict()
    style_counts = {s: len(lst) for s, lst in style_to_paintings.items()}
    total = sum(style_counts.values())

    ideal = {s: target_n * (count / total) for s, count in style_counts.items()}
    alloc = {s: floor(v) for s, v in ideal.items()}
    remainder = target_n - sum(alloc.values())

    # distribute remainder
    fracs = sorted([(s, ideal[s] - alloc[s]) for s in alloc], 
                   key=lambda x: -x[1])
    i = 0
    while remainder > 0:
        alloc[fracs[i][0]] += 1
        remainder -= 1
        i = (i + 1) % len(fracs)

    rng = np.random.RandomState(seed)
    sampled = []
    for s, k in alloc.items():
        available = list(style_to_paintings[s])
        chosen = rng.choice(available, size=min(k, len(available)), replace=False)
        sampled.extend(chosen)

    return set(sampled)


# ---------------------------------------------------------
# TRAIN/VAL/TEST SPLIT
# ---------------------------------------------------------

def split_by_painting(df):
    paintings = df['painting'].unique().tolist()
    np.random.RandomState(SEED).shuffle(paintings)

    n = len(paintings)
    n_train = int(SPLIT_LOADS[0] * n)
    n_val = int(SPLIT_LOADS[1] * n)

    train_p = set(paintings[:n_train])
    val_p = set(paintings[n_train:n_train+n_val])
    test_p = set(paintings[n_train+n_val:])

    def assign(p):
        if p in train_p: return "train"
        if p in val_p: return "val"
        return "test"

    df['split'] = df['painting'].apply(assign)
    return df


# ---------------------------------------------------------
# IMAGE COPY + RESIZE
# ---------------------------------------------------------

def copy_and_resize_images(df):
    os.makedirs(IMAGES_OUT, exist_ok=True)
    print("\nCopying + resizing images to 224×224...")

    copied = 0
    missing = 0

    for style, painting in df[['art_style', 'painting']].drop_duplicates().values:
        src = Path(WIKI_ROOT) / style / f"{painting}.jpg"
        dst_dir = Path(IMAGES_OUT) / style
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{painting}.jpg"

        if not src.exists():
            missing += 1
            continue

        # copy
        shutil.copy2(src, dst)

        # resize
        try:
            img = Image.open(dst).convert("RGB")
            img = img.resize((224, 224), Image.LANCZOS)
            img.save(dst)
            copied += 1
        except:
            continue

    print(f"Copied + resized: {copied}")
    print(f"Missing: {missing}")


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\nLoading CSV...")
    df = pd.read_csv(RAW_CSV)

    # Subsample
    print("\nSubsampling paintings...")
    sampled = stratified_subsample_by_style(df, TARGET_SUBSAMPLE)
    df = df[df['painting'].isin(sampled)].reset_index(drop=True)

    # Copy + resize images
    copy_and_resize_images(df)

    # Tokenization
    print("\nTokenizing text...")

    def add_special_tokens(toks):
        return ["<start>"] + toks + ["<end>"]

    df['tokens'] = df['utterance'].astype(str).apply(lambda x: add_special_tokens(tokenize(clean_text(x))))
    df['tokens_len'] = df['tokens'].apply(len)

        # Splits
    df = split_by_painting(df)

    # Vocabulary
    print("\nBuilding vocabulary from train split...")

    MAX_VOCAB_SIZE = 8000
    MIN_WORD_FREQ = 2

    counter = Counter()
    for toks in df[df['split'] == 'train']['tokens']:
        counter.update(toks)

    vocab = Vocab.build(counter, min_freq=MIN_WORD_FREQ)
    print("Vocab size:", len(vocab))


    # Convert tokens to IDs + padding
    print("\nConverting tokens to IDs and applying padding...")

    MAX_LEN = 20   # choose 20 or 30 depending on captions
    
    def tokens_to_ids(toks, vocab, max_len):
        ids = [vocab[t] for t in toks]  # uses __getitem__
        ids = ids[:max_len]              # truncate
        ids += [vocab[Vocab.PAD]] * (max_len - len(ids))  # pad
        return ids


    df['token_ids'] = df['tokens'].apply(lambda x: tokens_to_ids(x, vocab, MAX_LEN))
    df['token_ids_len'] = df['token_ids'].apply(len)

    # Save CSVs
    df.to_csv(osp.join(OUT_DIR, "artemis_preprocessed.csv"), index=False)
    for split in ["train", "val", "test"]:
        df[df['split'] == split].to_csv(
            osp.join(OUT_DIR, f"{split}.csv"), index=False
        )

    # Save vocab
    with open(osp.join(OUT_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab.token_to_idx, f)

    # Save summary
    summary = {
        "subsample_size": TARGET_SUBSAMPLE,
        "max_len": MAX_LEN,
        "min_word_freq": MIN_WORD_FREQ,
        "vocab_size": len(vocab.token_to_idx),
    }
    with open(osp.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()

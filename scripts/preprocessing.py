#!/usr/bin/env python3
#optas github -- cite
# preprocessing.py
# Lightweight ArtEmis preprocessing: subsample CSV -> optional copy images -> text cleaning -> vocab -> encode -> save splits
"""
mkdir -p data_preprocessed

python3 scripts/preprocessing.py \
  --raw-csv artemis_dataset.csv \
  --out-dir data_preprocessed \
  --subsample-size 7500 \
  --max-len 25 \
  --min-word-freq 2 \
  --copy-images \
  --wiki-root wikiart
"""
import os
import os.path as osp
import shutil
from pathlib import Path
import pickle
from collections import Counter
import numpy as np
import pandas as pd
import argparse
import json
import re

# -------------------- CONFIG / DEFAULTS --------------------
DEFAULT_RAW = "artemis_dataset.csv"        # input CSV (change if needed)
OUT_DIR = "data_preprocessed"                           # where outputs go
OUT_SUB_CSV = osp.join(OUT_DIR, "artemis_subsampled.csv")
OUT_PREP_CSV = osp.join(OUT_DIR, "artemis_preprocessed.csv")
IMAGES_SUBROOT = osp.join(OUT_DIR, "images_subset")  # optional copied subset
WIKI_ROOT = "wikiart"                      # path to original wikiart root
SUBSAMPLE_N = 7500
SEED = 42
MAX_LEN = 25        # includes <start> and <end>
MIN_WORD_FREQ = 2
SPLIT_LOADS = (0.8, 0.1, 0.1)  # train, val, test
COPY_IMAGES = True   # set False to skip copying images
DEDUP = False        # drop exact duplicate (painting, utterance) pairs before subsample


# -------------------- TEXT UTILITIES --------------------
def clean_text(s: str) -> str:
    """Lowercase, remove unwanted chars, collapse whitespace."""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    # keep letters and spaces
    s = re.sub(r"[^a-z0-9\\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str):
    return clean_text(s).split()


# -------------------- VOCAB CLASS --------------------
class Vocab:
    PAD = "<pad>"
    START = "<start>"
    END = "<end>"
    UNK = "<unk>"

    def __init__(self, token_to_idx):
        self.token_to_idx = token_to_idx
        self.idx_to_token = {i: t for t, i in token_to_idx.items()}

    @classmethod
    def build(cls, counter: Counter, min_freq=1, max_size=None):
        specials = [cls.PAD, cls.START, cls.END, cls.UNK]
        token_to_idx = {t: i for i, t in enumerate(specials)}
        items = [(t, c) for t, c in counter.items() if c >= min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))
        if max_size is not None:
            items = items[:max_size]
        start_idx = len(token_to_idx)
        for i, (t, _) in enumerate(items):
            if t in token_to_idx:
                continue
            token_to_idx[t] = start_idx + i
        return cls(token_to_idx)

    def encode(self, tokens, max_len):
        ids = [self.token_to_idx.get(t, self.token_to_idx[self.UNK]) for t in tokens]
        ids = [self.token_to_idx[self.START]] + ids[: max_len - 2] + [self.token_to_idx[self.END]]
        if len(ids) < max_len:
            ids = ids + [self.token_to_idx[self.PAD]] * (max_len - len(ids))
        return ids

    def decode(self, ids):
        return " ".join(self.idx_to_token.get(i, self.UNK) for i in ids)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.token_to_idx, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            token_to_idx = pickle.load(f)
        return cls(token_to_idx)


# -------------------- SPLIT BY PAINTING --------------------
def split_by_painting(df, split_loads=SPLIT_LOADS, seed=SEED, too_high_repetition=-1):
    paintings = df['painting'].unique().tolist()
    rng = np.random.RandomState(seed)

    rest_set = set()
    if too_high_repetition != -1 and 'repetition' in df.columns:
        rep = df.groupby('painting')['repetition'].first()
        rest_set = set(rep[rep >= too_high_repetition].index.tolist())

    paintings = [p for p in paintings if p not in rest_set]
    rng.shuffle(paintings)
    n = len(paintings)
    n_train = int(split_loads[0] * n)
    n_val = int(split_loads[1] * n)
    train_p = set(paintings[:n_train])
    val_p = set(paintings[n_train:n_train + n_val])
    test_p = set(paintings[n_train + n_val:])

    def which(p):
        if p in rest_set:
            return 'rest'
        if p in train_p:
            return 'train'
        if p in val_p:
            return 'val'
        return 'test'

    df = df.copy()
    df['split'] = df['painting'].apply(which)
    return df


# -------------------- MAIN PIPELINE --------------------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading CSV:", args.raw_csv)
    df = pd.read_csv(args.raw_csv)
    print("Rows loaded:", len(df))

    # optional dedup
    if args.dedup:
        before = len(df)
        df = df.drop_duplicates(subset=['painting', 'utterance']).reset_index(drop=True)
        print(f"Deduplicated rows: {before} -> {len(df)}")

    # Subsample paintings (if requested)
    if args.subsample_size is not None:
        paintings = df['painting'].unique()
        if args.subsample_size > len(paintings):
            raise ValueError("subsample_size > number of paintings available")
        rng = np.random.RandomState(args.seed)
        sampled = set(rng.choice(paintings, size=args.subsample_size, replace=False))
        df = df[df['painting'].isin(sampled)].reset_index(drop=True)
        print("After subsampling rows:", len(df), "unique paintings:", df['painting'].nunique())
    else:
        print("No subsampling requested; using full CSV")

    # Save the subsampled csv (helpful)
    subsampled_csv = osp.join(args.out_dir, "artemis_subsampled.csv")
    df.to_csv(subsampled_csv, index=False)
    print("Saved subsampled CSV to", subsampled_csv)

    # Optionally copy images for convenience (safe - does not delete originals)
    if args.copy_images:
        print("Copying images to", args.images_out)
        os.makedirs(args.images_out, exist_ok=True)
        missing = 0
        for art_style, painting in df[['art_style', 'painting']].drop_duplicates().values:
            src = Path(args.wiki_root) / art_style / (painting + ".jpg")
            dst_dir = Path(args.images_out) / art_style
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / (painting + ".jpg")
            if src.exists():
                shutil.copy2(src, dst)
            else:
                missing += 1
                # don't flood stdout; only report missing count
        print(f"Done copying images. Missing files: {missing}")

    # Clean & tokenize text
    print("Cleaning and tokenizing utterances...")
    df['utterance_spelled'] = df['utterance'].astype(str).apply(clean_text)
    df['tokens'] = df['utterance_spelled'].apply(tokenize)
    df['tokens_len'] = df['tokens'].apply(len)

    # Split by painting (train/val/test), ensures no leakage
    df = split_by_painting(df, split_loads=tuple(args.split_loads), seed=args.seed,
                           too_high_repetition=args.too_high_repetition)
    print("Split counts:\n", df['split'].value_counts())

    # Optionally drop too short/long (we keep all by default)
    if args.too_short_len > 0:
        before = len(df)
        df = df[df['tokens_len'] >= args.too_short_len].reset_index(drop=True)
        print(f"Dropped short captions (<{args.too_short_len}) : {before} -> {len(df)}")

    # Build vocab from train split
    print("Building vocabulary from train split...")
    counter = Counter()
    for toks in df[df['split'] == 'train']['tokens']:
        counter.update(toks)
    vocab = Vocab.build(counter, min_freq=args.min_word_freq, max_size=args.max_vocab_size)
    print("Vocab size:", len(vocab.token_to_idx))


    # Encode tokens (pad/truncate) -> tokens_encoded column
    df['tokens_encoded'] = df['tokens'].apply(lambda t: vocab.encode(t, args.max_len))

    # Map emotion to numeric (simple map)
    if 'emotion' in df.columns:
        emotions = sorted(df['emotion'].unique())
        emo2idx = {e: i for i, e in enumerate(emotions)}
        df['emotion_label'] = df['emotion'].map(emo2idx)

    # Save processed csv and splits
    preprocessed_csv = osp.join(args.out_dir, "artemis_preprocessed.csv")
    df.to_csv(preprocessed_csv, index=False)
    print("Saved preprocessed csv:", preprocessed_csv)

    # Save train/val/test separately
    for split in ['train', 'val', 'test', 'rest']:
        out = osp.join(args.out_dir, f"{split}.csv")
        if split in df['split'].values:
            df[df['split'] == split].to_csv(out, index=False)
            print("Saved split:", split, "->", out)

    # Save vocab
    vocab_path = osp.join(args.out_dir, "vocabulary.pkl")
    vocab.save(vocab_path)
    print("Saved vocab to", vocab_path)

    # Save quick summary file
    summary = {
        "subsample_size": args.subsample_size,
        "rows_kept": len(df),
        "unique_paintings": int(df['painting'].nunique()),
        "max_len": args.max_len,
        "min_word_freq": args.min_word_freq,
        "vocab_size": len(vocab.token_to_idx),
        "split_loads": args.split_loads
    }
    with open(osp.join(args.out_dir, "preprocessing_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved preprocessing summary")

    print("Done.")


# -------------------- ARGUMENTS --------------------
def parse_args():
    p = argparse.ArgumentParser(description="ArtEmis preprocessing with subsampling and basic tokenization")
    p.add_argument("--raw-csv", type=str, default=DEFAULT_RAW, help="raw ArtEmis CSV path")
    p.add_argument("--out-dir", type=str, default=OUT_DIR)
    p.add_argument("--subsample-size", type=int, default=SUBSAMPLE_N, help="num unique paintings to sample (None -> full)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max-len", type=int, default=MAX_LEN)
    p.add_argument("--min-word-freq", type=int, default=MIN_WORD_FREQ)
    p.add_argument("--max-vocab-size", type=int, default=None, help="cap vocab size (None -> no cap)")
    p.add_argument("--split-loads", type=float, nargs=3, default=SPLIT_LOADS)
    p.add_argument("--too-short-len", type=int, default=0)
    p.add_argument("--too-high-repetition", type=int, default=-1)
    p.add_argument("--copy-images", action="store_true", help="copy images for the sampled subset into out-dir")
    p.add_argument("--images-out", type=str, default=IMAGES_SUBROOT, help="where to place copied images")
    p.add_argument("--wiki-root", type=str, default=WIKI_ROOT, help="path to wikiart root images")
    p.add_argument("--dedup", action="store_true", help="drop exact duplicate (painting, utterance) pairs before subsampling")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

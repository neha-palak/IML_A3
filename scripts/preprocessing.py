#!/usr/bin/env python3
"""
preprocessing.py

Produces a single CSV file: artemis_preprocessed.csv which contains:
 - original text/tokens/token_ids
 - emotion-prepended text/tokens/token_ids (stringified lists for CSV compatibility)
 - split column (train/val/test)
Also saves vocab.pkl and preprocessing_summary.json.
"""

import os
import os.path as osp
import argparse
import json
import pickle
import shutil
import re
from pathlib import Path
from collections import Counter
from math import floor

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ----------------------------
# Try spaCy (require en_core_web_sm)
# ----------------------------
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
    except OSError:
        raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run: python3 -m spacy download en_core_web_sm")
except Exception as e:
    raise RuntimeError("spaCy is required. Install with: python3 -m pip install spacy && python3 -m spacy download en_core_web_sm") from e

# ----------------------------
# Utilities
# ----------------------------
SPECIAL_TOKENS = ["<pad>", "<start>", "<end>", "<unk>"]

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_spacy_batch(texts):
    """Tokenize a list of texts using spaCy efficiently (nlp.pipe)."""
    toks = []
    for doc in nlp.pipe(texts, batch_size=256, disable=["parser", "ner", "textcat"]):
        toks.append([t.text.lower() for t in doc if not t.is_space])
    return toks

def stratified_subsample_by_style(df, target_n, seed=42):
    style_to_paintings = df.groupby('art_style')['painting'].unique().to_dict()
    style_counts = {s: len(lst) for s, lst in style_to_paintings.items()}
    total = sum(style_counts.values())

    ideal = {s: target_n * (count / total) for s, count in style_counts.items()}
    alloc = {s: floor(v) for s, v in ideal.items()}
    remainder = target_n - sum(alloc.values())

    # allocate remainders by largest fractional parts
    fracs = sorted([(s, ideal[s] - alloc[s]) for s in alloc], key=lambda x: -x[1])
    i = 0
    while remainder > 0 and fracs:
        alloc[fracs[i % len(fracs)][0]] += 1
        remainder -= 1
        i += 1

    rng = np.random.RandomState(seed)
    sampled = []
    for s, k in alloc.items():
        available = list(style_to_paintings[s])
        if k <= 0:
            continue
        chosen = rng.choice(available, size=min(k, len(available)), replace=False)
        sampled.extend(chosen)
    return set(sampled)

def split_by_painting(df, split_loads=(0.85,0.05,0.10), seed=42):
    paintings = df['painting'].unique().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(paintings)
    n = len(paintings)
    n_train = int(split_loads[0] * n)
    n_val = int(split_loads[1] * n)
    train_p = set(paintings[:n_train])
    val_p = set(paintings[n_train:n_train+n_val])
    test_p = set(paintings[n_train+n_val:])
    def which(p):
        if p in train_p:
            return "train"
        if p in val_p:
            return "val"
        return "test"
    df = df.copy()
    df['split'] = df['painting'].apply(which)
    return df

def build_vocab(counter, max_size=None, specials=SPECIAL_TOKENS):
    token_to_idx = {t:i for i,t in enumerate(specials)}
    # exclude specials from counter if present
    items = [(t,c) for t,c in counter.items() if t not in specials]
    items.sort(key=lambda x: (-x[1], x[0]))
    if max_size is not None:
        items = items[: max_size - len(specials)]
    for i, (tok, _) in enumerate(items, start=len(token_to_idx)):
        token_to_idx[tok] = i
    return token_to_idx

def ensure_emotion_tokens_in_vocab(token_to_idx, emotion_words):
    """Make sure every emotion word is present in vocab. If missing, append."""
    added = []
    for emo in emotion_words:
        if emo not in token_to_idx:
            token_to_idx[emo] = len(token_to_idx)
            added.append(emo)
    return added

def encode_tokens(tokens, token_to_idx, max_len):
    ids = [token_to_idx.get("<start>")]
    for t in tokens:
        ids.append(token_to_idx.get(t, token_to_idx.get("<unk>")))
        if len(ids) >= max_len - 1:  # leave room for end token
            break
    ids.append(token_to_idx.get("<end>"))
    # pad
    if len(ids) < max_len:
        ids += [token_to_idx.get("<pad>")] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

# ----------------------------
# Main function
# ----------------------------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    images_out = osp.join(args.out_dir, "images_subset")
    features_out = osp.join(args.out_dir, "features")

    # CLEANUP: remove old images/features before new run
    shutil.rmtree(images_out, ignore_errors=True)
    shutil.rmtree(features_out, ignore_errors=True)

    # recreate fresh directories
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(features_out, exist_ok=True)

    print("Loading CSV:", args.raw_csv)
    df = pd.read_csv(args.raw_csv)
    print("Rows loaded:", len(df))

    # Optional dedup
    if args.dedup:
        before = len(df)
        df = df.drop_duplicates(subset=['painting','utterance']).reset_index(drop=True)
        print(f"Deduplicated rows: {before} -> {len(df)}")

    # Subsample paintings if requested
    if args.subsample_size is not None and args.subsample_size > 0:
        print(f"Subsampling to {args.subsample_size} unique paintings (stratified by art_style)...")
        if args.subsample_by_style:
            sampled_paintings = stratified_subsample_by_style(df, args.subsample_size, seed=args.seed)
        else:
            paintings = df['painting'].unique()
            if args.subsample_size > len(paintings):
                raise ValueError("subsample_size > number of paintings available")
            rng = np.random.RandomState(args.seed)
            sampled_paintings = set(rng.choice(paintings, size=args.subsample_size, replace=False))
        df = df[df['painting'].isin(sampled_paintings)].reset_index(drop=True)
        print("After subsampling rows:", len(df), "unique paintings:", df['painting'].nunique())
    else:
        print("No subsampling requested; using full CSV")

    # Emotion filter & mapping
    EMOTION_MAP = {
        "amusement": 0, "contentment": 1, "awe": 2, "excitement": 3,
        "fear": 4, "anger": 5, "sadness": 6, "disgust": 7, "something else": 8
    }
    if 'emotion' not in df.columns:
        raise RuntimeError("Raw CSV must contain 'emotion' column")
    df = df[df['emotion'].isin(EMOTION_MAP.keys())].reset_index(drop=True)
    df['emotion_label'] = df['emotion'].map(EMOTION_MAP).astype(int)
    emotion_counts = df['emotion'].value_counts().to_dict()
    print("Emotion counts (kept):", emotion_counts)

    # Copy + resize images and save normalized .npy features (one per painting)
    if args.copy_images:
        print("Copying + resizing images from wiki root:", args.wiki_root)
        copied = 0
        missing = 0
        errors = 0
        for art_style, painting in df[['art_style','painting']].drop_duplicates().values:
            src = Path(args.wiki_root) / art_style / f"{painting}.jpg"
            dst = Path(images_out) / f"{painting}.jpg"
            feat_dst = Path(features_out) / f"{painting}.npy"
            try:
                if not src.exists():
                    missing += 1
                    continue
                # copy original
                shutil.copy2(src, dst)
                # resize and save (image_size x image_size)
                img = Image.open(dst).convert("RGB").resize((args.image_size, args.image_size), Image.LANCZOS)
                img.save(dst)
                arr = np.array(img).astype("float32") / 255.0
                np.save(feat_dst, arr)
                copied += 1
            except Exception as e:
                errors += 1
                # cleanup
                try:
                    if dst.exists(): dst.unlink()
                    if feat_dst.exists(): feat_dst.unlink()
                except Exception:
                    pass
        print("Images copied:", copied, "missing:", missing, "errors:", errors)
    else:
        print("Skipping image copy/resize (--copy-images not set)")

    # Clean text & tokenize (use spaCy pipeline)
    print("Cleaning text and tokenizing using spaCy...")
    df['utter_clean'] = df['utterance'].astype(str).apply(clean_text)
    texts = df['utter_clean'].tolist()
    tokens_list = tokenize_spacy_batch(texts)
    df['tokens'] = tokens_list
    df['tokens_len'] = df['tokens'].apply(len)
    print("Tokenization complete. Sample tokens:", df['tokens'].iloc[0] if len(df)>0 else None)

    # Create emotion-prepended utterances/tokens (but we will keep original as well)
    print("Building emotion-prepended text (kept in same CSV)...")
    df['utter_with_emotion'] = df['emotion'].astype(str) + " " + df['utter_clean'].astype(str)
    texts_emo = df['utter_with_emotion'].tolist()
    tokens_list_emo = tokenize_spacy_batch(texts_emo)
    df['tokens_with_emotion'] = tokens_list_emo
    df['tokens_with_emotion_len'] = df['tokens_with_emotion'].apply(len)

    # Drop too short / too long if requested (based on original tokens)
    if args.min_len > 0:
        before = len(df)
        df = df[df['tokens_len'] >= args.min_len].reset_index(drop=True)
        print(f"Dropped short captions (<{args.min_len} tokens): {before} -> {len(df)}")
    if args.max_len is not None and args.max_len > 0:
        before = len(df)
        df = df[df['tokens_len'] <= args.max_len].reset_index(drop=True)
        print(f"Dropped long captions (>{args.max_len} tokens): {before} -> {len(df)}")

    # Split by painting to avoid leakage
    df = split_by_painting(df, split_loads=args.split_loads, seed=args.seed)
    print("Split counts:\n", df['split'].value_counts())

    # Build vocab from training split tokens (original tokens, not emotion-prepended)
    print("Building vocab from training split (original tokens)...")
    train_tokens = df[df['split']=='train']['tokens'].tolist()
    counter = Counter()
    for toks in train_tokens:
        counter.update(toks)
    token_to_idx = build_vocab(counter, max_size=args.max_vocab_size)
    # Ensure emotion words are present in vocab (so they don't map to <unk> in emotion-prepended sequences)
    added_emotions = ensure_emotion_tokens_in_vocab(token_to_idx, list(EMOTION_MAP.keys()))
    if added_emotions:
        print("Added emotion tokens to vocab (ensured presence):", added_emotions)
    print("Vocab size:", len(token_to_idx))

    # Save vocab (pickle)
    vocab_path = osp.join(args.out_dir, "vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(token_to_idx, f)
    print("Saved vocab ->", vocab_path)

    # Encode tokens into token_ids (pad/truncate to max_len)
    print("Encoding tokens to token_ids (max_len):", args.max_len)
    df['token_ids'] = df['tokens'].apply(lambda t: encode_tokens(t, token_to_idx, args.max_len))
    df['token_ids_c1'] = df['token_ids']  # alias: original

    # Encode tokens_with_emotion into token ids (separate column)
    df['token_ids_with_emotion'] = df['tokens_with_emotion'].apply(lambda t: encode_tokens(t, token_to_idx, args.max_len))

    # BEFORE SAVING: stringify token/token_id lists so CSV is safe and easy to read
    df_to_save = df.copy()
    df_to_save['tokens_str'] = df_to_save['tokens'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    df_to_save['tokens_with_emotion_str'] = df_to_save['tokens_with_emotion'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    df_to_save['token_ids_str'] = df_to_save['token_ids'].apply(lambda x: str(x))
    df_to_save['token_ids_with_emotion_str'] = df_to_save['token_ids_with_emotion'].apply(lambda x: str(x))

    # Keep original useful columns and the stringified fields in a single CSV
    cols_keep = [
        'art_style','painting','emotion','utterance','repetition',
        'emotion_label','utter_clean','tokens_str','tokens_len',
        'tokens_with_emotion_str','tokens_with_emotion_len','split',
        'token_ids_str','token_ids_with_emotion_str'
    ]
    cols_present = [c for c in cols_keep if c in df_to_save.columns]
    out_csv = osp.join(args.out_dir, "artemis_preprocessed.csv")
    df_to_save[cols_present].to_csv(out_csv, index=False)
    print("Saved single preprocessed CSV ->", out_csv)

    # summary
    n_images_written = 0
    if args.copy_images:
        for root, _, files in os.walk(images_out):
            n_images_written += sum(1 for f in files if f.lower().endswith(".jpg"))
    n_features_written = 0
    for root, _, files in os.walk(features_out):
        n_features_written += sum(1 for f in files if f.lower().endswith(".npy"))

    summary = {
        "subsample_size": args.subsample_size,
        "rows_kept": len(df),
        "unique_paintings": int(df['painting'].nunique()),
        "vocab_size": len(token_to_idx),
        "max_len": args.max_len,
        "min_len": args.min_len,
        "split_loads": args.split_loads,
        "images_written": n_images_written,
        "features_written": n_features_written,
        "emotion_counts": emotion_counts,
        "added_emotion_tokens_to_vocab": added_emotions
    }
    with open(osp.join(args.out_dir, "preprocessing_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary ->", osp.join(args.out_dir, "preprocessing_summary.json"))
    print("Done.")

# ----------------------------
# CLI arguments
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ArtEmis preprocessing (spaCy tokenization) - single CSV output")
    p.add_argument("--raw-csv", type=str, default="artemis_dataset.csv")
    p.add_argument("--out-dir", type=str, default="new_preprocessed")
    p.add_argument("--wiki-root", type=str, default="wikiart", help="root folder of wikiart images organized by art_style")
    p.add_argument("--subsample-size", type=int, default=5500, help="number of unique paintings to sample (None or 0 => full)")
    p.add_argument("--subsample-by-style", action="store_true", help="stratify subsample by art_style (recommended)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy-images", action="store_false", dest="copy_images",
               help="disable copying images")
    p.set_defaults(copy_images=True)    
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--max-len", type=int, default=25, help="max tokens (including <start>/<end>)")
    p.add_argument("--min-len", type=int, default=3, help="min token length (after tokenization)")
    p.add_argument("--max-vocab-size", type=int, default=8000)
    p.add_argument("--split-loads", type=float, nargs=3, default=(0.8,0.1,0.1))
    p.add_argument("--dedup", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # normalize None/0 for subsample
    if args.subsample_size is None or args.subsample_size <= 0:
        args.subsample_size = None
    # map arg names used inside main
    args.max_vocab_size = getattr(args, "max_vocab_size", args.max_vocab_size if hasattr(args, "max_vocab_size") else args.max_vocab_size)
    main(args)
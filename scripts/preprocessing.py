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
import argparse
from sklearn.preprocessing import LabelEncoder

# FIXED CONFIGURATION

RAW_CSV = "artemis_dataset.csv"
OUT_DIR = "data_preprocessed"
IMAGES_OUT = osp.join(OUT_DIR, "images_subset")
FEATURES_OUT = osp.join(OUT_DIR, "features") 
WIKI_ROOT = "wikiart"

TARGET_SUBSAMPLE = 7500
SEED = 42

MAX_LEN = 20  # sequence length
MAX_VOCAB_SIZE = 8000  
SPLIT_LOADS = (0.8, 0.1, 0.1)

# LOWERCASE + BASIC CLEANING BEFORE TOKENIZATION

def clean_text_basic(s: str) -> str:
    """Lowercase & remove punctuation before word-level tokenizer."""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def tokenize_word_level(text: str):
    tokens = text.strip().split()
    return ["<start>"] + tokens + ["<end>"]

def stratified_subsample_by_style(df, target_n, seed=SEED):
    style_to_paintings = df.groupby('art_style')['painting'].unique().to_dict()
    style_counts = {s: len(lst) for s, lst in style_to_paintings.items()}
    total = sum(style_counts.values())

    ideal = {s: target_n * (count / total) for s, count in style_counts.items()}
    alloc = {s: floor(v) for s, v in ideal.items()}
    remainder = target_n - sum(alloc.values())

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

# TRAIN/VAL/TEST SPLIT

def split_by_painting(df):
    paintings = df['painting'].unique().tolist()
    np.random.RandomState(SEED).shuffle(paintings)

    n = len(paintings)
    n_train = int(SPLIT_LOADS[0] * n)
    n_val = int(SPLIT_LOADS[1] * n)

    train_p = set(paintings[:n_train])
    val_p = set(paintings[n_train:n_train + n_val])
    test_p = set(paintings[n_train + n_val:])

    def assign(p):
        if p in train_p: return "train"
        if p in val_p: return "val"
        return "test"

    df['split'] = df['painting'].apply(assign)
    return df

# IMAGE COPY + RESIZE + NORMALIZE
def copy_and_resize_images(df):
    os.makedirs(IMAGES_OUT, exist_ok=True)
    os.makedirs(FEATURES_OUT, exist_ok=True)
    print("\nCopying + resizing + normalizing images...")

    copied = 0
    missing = 0
    errors = 0

    for style, painting in df[['art_style', 'painting']].drop_duplicates().values:
        src = Path(WIKI_ROOT) / style / f"{painting}.jpg"

        # ---- Flat output ----
        dst = Path(IMAGES_OUT) / f"{painting}.jpg"
        feat_path = Path(FEATURES_OUT) / f"{painting}.npy"

        if not src.exists():
            missing += 1
            continue

        try:
            # copy original
            shutil.copy2(src, dst)

            # resize & save JPG
            img = Image.open(dst).convert("RGB")
            img = img.resize((128, 128), Image.LANCZOS)
            img.save(dst)

            # save normalized numpy array
            arr = np.array(img).astype("float32") / 255.0
            np.save(feat_path, arr)

            copied += 1

        except Exception as e:
            print("Image error:", e)
            errors += 1

            # cleanup partial files
            try:
                if dst.exists():
                    dst.unlink()
                if feat_path.exists():
                    feat_path.unlink()
            except:
                pass

            continue

    print(f"Copied: {copied}, Missing: {missing}, Errors: {errors}")


# MAIN PIPELINE

def main():
    for folder in [IMAGES_OUT, FEATURES_OUT]:
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\nLoading CSV...")
    df = pd.read_csv(RAW_CSV)

    # Subsample
    print("\nSubsampling by art_style...")
    sampled = stratified_subsample_by_style(df, TARGET_SUBSAMPLE)
    df = df[df['painting'].isin(sampled)].reset_index(drop=True)

    # Copy + normalize images (features saved to FEATURES_OUT)
    copy_and_resize_images(df)

    # Clean text before tokenization
    df['utter_clean'] = df['utterance'].astype(str).apply(clean_text_basic)

    # Word-level tokenization with <start>/<end>
    print("\nTokenizing at word level...")
    df['tokens'] = df['utter_clean'].apply(tokenize_word_level)

    # REDUCE VOCAB TO TOP 8000 TOKENS
    print("\nBuilding reduced vocab...")

    counter = Counter()
    for toks in df['tokens']:
        counter.update(toks)

    # Keep <pad>, <start>, <end>, <unk>
    SPECIALS = ["<pad>", "<start>", "<end>", "<unk>"]
    vocab_items = [(tok, cnt) for tok, cnt in counter.items() if tok not in SPECIALS]
    vocab_items.sort(key=lambda x: -x[1])
    vocab_items = vocab_items[:MAX_VOCAB_SIZE - len(SPECIALS)]

    token_to_idx = {sp: i for i, sp in enumerate(SPECIALS)}
    for i, (tok, _) in enumerate(vocab_items, start=len(SPECIALS)):
        token_to_idx[tok] = i

    print("Final vocab size:", len(token_to_idx))

    # Save vocab
    with open(osp.join(OUT_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(token_to_idx, f)

    # Encode tokens
    print("\nEncoding tokens...")
    def encode_tokens(toks):
        ids = [token_to_idx.get(tok, token_to_idx["<unk>"]) for tok in toks]
        ids = ids[:MAX_LEN]
        ids += [token_to_idx["<pad>"]] * (MAX_LEN - len(ids))
        return ids

    df['token_ids'] = df['tokens'].apply(encode_tokens)

    # Split
    df = split_by_painting(df)

    label_encoder = LabelEncoder()
    df['emotion_label'] = label_encoder.fit_transform(df['emotion'])

    # Drop requested columns
    drop_cols = ["repetition","split","token_ids_len","utterance"]
    df_out = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Save main CSV
    df_out.to_csv(osp.join(OUT_DIR, "artemis_preprocessed.csv"), index=False)

    # Save splits
    for split in ["train", "val", "test"]:
        part = df[df['split'] == split]
        part_out = part.drop(columns=[c for c in drop_cols if c in part.columns], errors="ignore")
        part_out.to_csv(osp.join(OUT_DIR, f"{split}.csv"), index=False)

    # Summary - include counts for images and features
    n_images_written = 0
    for root, _, files in os.walk(IMAGES_OUT):
        n_images_written += sum(1 for f in files if f.lower().endswith(".jpg"))
    n_features_written = 0
    for root, _, files in os.walk(FEATURES_OUT):
        n_features_written += sum(1 for f in files if f.lower().endswith(".npy"))

    summary = {
        "subsample_size": TARGET_SUBSAMPLE,
        "max_len": MAX_LEN,
        "vocab_size": len(token_to_idx),
        "normalization": "pixel values in .npy files are in [0,1]",
        "rows_kept": len(df_out),
        "unique_paintings": int(df_out['painting'].nunique()),
        "images_written_flat_folder": n_images_written,
        "features_written_flat_folder": n_features_written
    }
    with open(osp.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()

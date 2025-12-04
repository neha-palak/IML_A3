#!/usr/bin/env python3
"""
text_representation.py

Builds text representations for the new preprocessing pipeline that produces:

    new_preprocessed/artemis_preprocessed.csv
    new_preprocessed/vocab.pkl

Implements 3 embedding strategies:

1. TF-IDF + dimensionality reduction (TruncatedSVD ~ PCA for sparse data)
   - Fits on TRAIN split utter_clean text.
   - Saves:
        new_preprocessed/tfidf_vectorizer.pkl
        new_preprocessed/tfidf_svd.pkl
        (optional) per-row reduced vectors in new_preprocessed/tfidf_npy/

2. Pre-trained GloVe embeddings
   - raw_data/glove.6B.300d.txt  (edit path in CONFIG if different)
   - Saves: new_preprocessed/emb_glove_300d.npy

3. Pre-trained FastText embeddings
   - raw_data/wiki-news-300d-1M-subword.vec  (edit path in CONFIG if different)
   - Saves: new_preprocessed/emb_fasttext_300d.npy

Also writes:
    new_preprocessed/representation_summary.json
with coverage & TF-IDF stats.
"""

import os
import os.path as osp
import pickle
import json
from types import SimpleNamespace
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# ------------------------------------------------------------------
# CONFIG – edit paths / hyperparams here
# ------------------------------------------------------------------
CONFIG = {
    # paths
    "csv_path": "new_preprocessed/artemis_preprocessed.csv",
    "vocab_path": "new_preprocessed/vocab.pkl",
    "out_dir": "new_preprocessed",
    "glove_path": "raw_data/glove.6B.300d.txt",
    "fasttext_path": "raw_data/wiki-news-300d-1M-subword.vec",

    # TF-IDF
    "max_features": 20000,
    "n_components": 2084,     # dimensionality after SVD (will be clipped if > n_features)
    "save_tfidf_npy": True,  # save per-row reduced vectors for train/val/test

    # FastText loading
    "fasttext_max_load": 150000,  # max lines to scan (to limit RAM). None => full.

    # misc
    "seed": 42,
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
SPECIAL_TOKENS = ["<pad>", "<start>", "<end>", "<unk>"]

def load_vocab(vocab_path):
    with open(vocab_path, "rb") as f:
        tok2idx = pickle.load(f)
    # allow for Vocab-like object with .token_to_idx
    if not isinstance(tok2idx, dict) and hasattr(tok2idx, "token_to_idx"):
        tok2idx = tok2idx.token_to_idx
    return tok2idx

def load_preprocessed_csv(csv_path):
    """
    Loads artemis_preprocessed.csv and returns the dataframe.
    Assumes columns:
        utter_clean, tokens_str, split, ...
    """
    df = pd.read_csv(csv_path)
    if "utter_clean" not in df.columns:
        raise RuntimeError("CSV must contain column 'utter_clean'.")
    if "split" not in df.columns:
        raise RuntimeError("CSV must contain column 'split' (train/val/test).")
    return df

def tokens_from_str_column(series):
    """
    Convert tokens_str (space-separated tokens) into list[list[str]].
    """
    out = []
    for s in series.fillna(""):
        if isinstance(s, str):
            s = s.strip()
            out.append(s.split() if s else [])
        else:
            out.append([])
    return out

# ------------ GloVe / FastText loaders ------------

def load_glove_vectors(glove_path, token_set):
    found = {}
    emb_dim = None
    print(f"Reading GloVe from {glove_path} ...")
    with open(glove_path, "r", encoding="utf8", errors="ignore") as f:
        for line in tqdm(f, desc="GloVe lines"):
            parts = line.rstrip().split(" ")
            if len(parts) < 2:
                continue
            w = parts[0]
            if w in token_set:
                vec = np.asarray(parts[1:], dtype=np.float32)
                found[w] = vec
                if emb_dim is None:
                    emb_dim = vec.shape[0]
            if len(found) == len(token_set):
                break
    return found, emb_dim

def load_fasttext_vectors(ft_path, token_set, max_load=None):
    found = {}
    emb_dim = None
    print(f"Reading FastText from {ft_path} ...")
    with open(ft_path, "r", encoding="utf8", errors="ignore") as f:
        header = f.readline()
        # if header is "n d", skip; else go back
        if len(header.split()) != 2 or not header.split()[1].isdigit():
            f.seek(0)
        for i, line in enumerate(tqdm(f, desc="FastText lines")):
            parts = line.rstrip().split(" ")
            if len(parts) < 2:
                continue
            w = parts[0]
            if w in token_set:
                vec = np.asarray(parts[1:], dtype=np.float32)
                found[w] = vec
                if emb_dim is None:
                    emb_dim = vec.shape[0]
            if max_load is not None and i >= max_load:
                break
            if len(found) == len(token_set):
                break
    return found, emb_dim

def build_embedding_matrix(token_to_idx, found_vecs, emb_dim, seed=42):
    rng = np.random.RandomState(seed)
    vocab_size = len(token_to_idx)
    mat = rng.normal(scale=0.6, size=(vocab_size, emb_dim)).astype("float32")
    covered = 0
    for tok, idx in token_to_idx.items():
        vec = found_vecs.get(tok)
        if vec is not None:
            mat[idx] = vec
            covered += 1
    return mat, covered

def token_level_coverage(token_to_idx, found_vecs):
    if len(token_to_idx) == 0:
        return 0.0
    return sum(1 for t in token_to_idx if t in found_vecs) / len(token_to_idx)

def occurrence_coverage(tokens_lists, found_vecs):
    total = 0
    covered = 0
    for toks in tokens_lists:
        for t in toks:
            total += 1
            if t in found_vecs:
                covered += 1
    return covered / total if total > 0 else 0.0

# ------------ TF-IDF helpers ------------

def fit_tfidf(train_texts, max_features=20000):
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 1),
        lowercase=True
    )
    X = tfidf.fit_transform(train_texts)
    return tfidf, X

def fit_svd(X_sparse, n_components, seed=42):
    # clip to valid range
    n_components = min(n_components, X_sparse.shape[1] - 1) if X_sparse.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    Xr = svd.fit_transform(X_sparse)
    return svd, Xr

def save_tfidf_split_npy(df, split_name, tfidf, svd, out_dir):
    """
    For a given split ('train'/'val'/'test'), transform utter_clean texts
    and save each reduced TF-IDF vector as .npy.
    Filenames: tfidf_npy/{split}__{rowidx:06d}.npy
    """
    sub = df[df["split"] == split_name]
    if len(sub) == 0:
        return 0
    texts = sub["utter_clean"].fillna("").astype(str).tolist()
    X = tfidf.transform(texts)
    Xr = svd.transform(X)
    tfidf_dir = osp.join(out_dir, "tfidf_npy")
    os.makedirs(tfidf_dir, exist_ok=True)
    for i, vec in enumerate(tqdm(Xr, desc=f"Saving TF-IDF for split={split_name}")):
        path = osp.join(tfidf_dir, f"{split_name}__{i:06d}.npy")
        np.save(path, vec.astype("float32"))
    return len(sub)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)

    # ----- load vocab + CSV -----
    token_to_idx = load_vocab(cfg.vocab_path)
    vocab_size = len(token_to_idx)
    print("Vocab size:", vocab_size)

    df = load_preprocessed_csv(cfg.csv_path)
    print("Total rows in CSV:", len(df))

    # recover token lists from tokens_str (for coverage stats)
    if "tokens_str" in df.columns:
        all_tokens_lists = tokens_from_str_column(df["tokens_str"])
    else:
        # fallback: basic split of utter_clean
        all_tokens_lists = tokens_from_str_column(df["utter_clean"])

    train_df = df[df["split"] == "train"]
    train_texts = train_df["utter_clean"].fillna("").astype(str).tolist()
    train_tokens_lists = [all_tokens_lists[i] for i in train_df.index]

    summary = {
        "vocab_size": vocab_size,
        "embeddings": {},
        "tfidf": {}
    }

    # ==============================================================
    # 1) TF-IDF + SVD
    # ==============================================================
    print("\n[1] TF-IDF + SVD")
    tfidf, X_train_sparse = fit_tfidf(train_texts, max_features=cfg.max_features)
    print("TF-IDF matrix shape (train):", X_train_sparse.shape)

    # save vectorizer
    tfidf_path = osp.join(cfg.out_dir, "tfidf_vectorizer.pkl")
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)
    print("Saved TF-IDF vectorizer ->", tfidf_path)

    svd, X_train_reduced = fit_svd(X_train_sparse, cfg.n_components, seed=cfg.seed)
    svd_path = osp.join(cfg.out_dir, "tfidf_svd.pkl")
    with open(svd_path, "wb") as f:
        pickle.dump(svd, f)
    print("Saved SVD ->", svd_path)

    explained = float(svd.explained_variance_ratio_.sum())
    print(f"SVD components: {svd.n_components}, explained variance (cumulative): {explained:.4f}")

    summary["tfidf"] = {
        "n_features": int(X_train_sparse.shape[1]),
        "n_components": int(svd.n_components),
        "explained_variance": explained
    }

    if cfg.save_tfidf_npy:
        print("Saving reduced TF-IDF vectors per split ...")
        for split_name in ["train", "val", "test"]:
            cnt = save_tfidf_split_npy(df, split_name, tfidf, svd, cfg.out_dir)
            print(f"  {split_name}: saved {cnt} vectors")

    # ==============================================================
    # 2) GloVe
    # ==============================================================
    if cfg.glove_path and osp.exists(cfg.glove_path):
        print("\n[2] GloVe embeddings")
        token_set = set(token_to_idx.keys())
        glove_found, glove_dim = load_glove_vectors(cfg.glove_path, token_set)
        if glove_dim is None:
            print("WARNING: Could not infer GloVe dimension; skipping.")
        else:
            emb_glove, covered_glove = build_embedding_matrix(
                token_to_idx, glove_found, glove_dim, seed=cfg.seed
            )
            out_glove = osp.join(cfg.out_dir, f"emb_glove_{glove_dim}d.npy")
            np.save(out_glove, emb_glove)
            print("Saved GloVe matrix ->", out_glove)

            tok_cov = token_level_coverage(token_to_idx, glove_found)
            occ_cov = occurrence_coverage(train_tokens_lists, glove_found)
            print(f"GloVe token coverage: {tok_cov:.4f}")
            print(f"GloVe occurrence coverage (train): {occ_cov:.4f}")

            summary["embeddings"]["glove"] = {
                "path": out_glove,
                "dim": int(glove_dim),
                "token_coverage": float(tok_cov),
                "occurrence_coverage": float(occ_cov),
                "found_tokens": int(len(glove_found)),
            }
    else:
        print("\n[2] GloVe embeddings: path not set or file missing, skipping.")

    # ==============================================================
    # 3) FastText
    # ==============================================================
    if cfg.fasttext_path and osp.exists(cfg.fasttext_path):
        print("\n[3] FastText embeddings")
        token_set = set(token_to_idx.keys())
        ft_found, ft_dim = load_fasttext_vectors(
            cfg.fasttext_path, token_set, max_load=cfg.fasttext_max_load
        )
        if ft_dim is None:
            print("WARNING: Could not infer FastText dimension; skipping.")
        else:
            emb_ft, covered_ft = build_embedding_matrix(
                token_to_idx, ft_found, ft_dim, seed=cfg.seed
            )
            out_ft = osp.join(cfg.out_dir, f"emb_fasttext_{ft_dim}d.npy")
            np.save(out_ft, emb_ft)
            print("Saved FastText matrix ->", out_ft)

            tok_cov = token_level_coverage(token_to_idx, ft_found)
            occ_cov = occurrence_coverage(train_tokens_lists, ft_found)
            print(f"FastText token coverage: {tok_cov:.4f}")
            print(f"FastText occurrence coverage (train): {occ_cov:.4f}")

            summary["embeddings"]["fasttext"] = {
                "path": out_ft,
                "dim": int(ft_dim),
                "token_coverage": float(tok_cov),
                "occurrence_coverage": float(occ_cov),
                "found_tokens": int(len(ft_found)),
            }
    else:
        print("\n[3] FastText embeddings: path not set or file missing, skipping.")

    # ==============================================================
    # Save summary
    # ==============================================================
    summary_path = osp.join(cfg.out_dir, "representation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved representation summary ->", summary_path)
    print("Done.")


if __name__ == "__main__":
    args = SimpleNamespace(**CONFIG)
    main(args)
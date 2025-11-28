#!/usr/bin/env python3
"""
text_representation_no_cli.py

Same functionality as your original script but runs without CLI.
Edit the CONFIG dictionary below to change inputs/options, then run:
    python text_representation_no_cli.py
"""

import os
import os.path as osp
import pickle
import json
from collections import Counter
from tqdm import tqdm
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# -----------------------------
# CONFIG (edit these values)
# -----------------------------
CONFIG = {
    "vocab": "data_preprocessed/vocab.pkl",
    "train_csv": "data_preprocessed/train.csv",
    "val_csv": "data_preprocessed/val.csv",
    "test_csv": "data_preprocessed/test.csv",
    "tokens_col": "tokens",
    "text_col": "utter_clean",
    "glove": "raw_data/glove.6B.300d.txt", 
    "fasttext": "raw_data/wiki-news-300d-1M-subword.vec", 
    "fasttext_max_load": 100000,
    "out_dir": "data_preprocessed",
    "max_features": 50000,
    "n_components": 512,
    "save_tfidf_npy": False,
    "do_tfidf": True,
    "seed": 42,
}

# --------------------------
# Utilities (unchanged)
# --------------------------

def load_vocab(vocab_path):
    with open(vocab_path, "rb") as f:
        token_to_idx = pickle.load(f)
    return token_to_idx

def load_tokens_from_csv(csv_path, tokens_col="tokens", text_col="utter_clean"):
    df = pd.read_csv(csv_path)
    tokens_list = []
    texts = []
    for _, row in df.iterrows():
        # tokens (optional)
        t = row.get(tokens_col, None)
        if pd.isna(t):
            tokens_list.append([])
        elif isinstance(t, str) and t.startswith("[") and t.endswith("]"):
            try:
                tokens_list.append(eval(t))
            except Exception:
                tokens_list.append(t.split())
        elif isinstance(t, str):
            tokens_list.append(t.split())
        elif isinstance(t, (list, tuple)):
            tokens_list.append(list(t))
        else:
            tokens_list.append([])

        # text (utter_clean used for TF-IDF)
        texts.append(row.get(text_col, "") if not pd.isna(row.get(text_col, "")) else "")
    return tokens_list, texts

def load_glove_vectors(glove_path, token_set):
    found = {}
    emb_dim = None
    with open(glove_path, "r", encoding="utf8", errors="ignore") as f:
        for line in tqdm(f, desc="Reading GloVe"):
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
    with open(ft_path, "r", encoding="utf8", errors="ignore") as f:
        header = f.readline()
        if len(header.split()) == 2 and header.split()[1].isdigit():
            pass
        else:
            f.seek(0)
        for i, line in enumerate(tqdm(f, desc="Reading FastText")):
            parts = line.rstrip().split(" ")
            if len(parts) < 2:
                continue
            w = parts[0]
            if w in token_set:
                vec = np.asarray(parts[1:], dtype=np.float32)
                found[w] = vec
                if emb_dim is None:
                    emb_dim = vec.shape[0]
            if max_load and i >= max_load:
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

def fit_tfidf_and_svd(train_texts, max_features=50000, n_components=512):
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1,1))
    X = tfidf.fit_transform(train_texts)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Xr = svd.fit_transform(X)
    return tfidf, svd, X, Xr

def transform_and_save_tfidf_for_split(df_csv, tfidf, svd, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    _, texts = load_tokens_from_csv(df_csv)
    X = tfidf.transform(texts)
    Xr = svd.transform(X)
    for i, vec in enumerate(tqdm(Xr, desc=f"Saving TF-IDF reduced for {osp.basename(df_csv)}")):
        np.save(osp.join(out_dir, f"{osp.basename(df_csv)}__{i}.npy"), vec.astype("float32"))
    return len(Xr)

# MAIN (accepts args-like object)

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # load vocab
    token_to_idx = load_vocab(args.vocab)
    print("Vocab size:", len(token_to_idx))

    token_set = set(token_to_idx.keys())

    # load train tokens/text for occurrence-weighted coverage and tfidf fit
    train_tokens, train_texts = load_tokens_from_csv(args.train_csv, tokens_col=args.tokens_col, text_col=args.text_col)
    print("Train rows:", len(train_texts))

    summary = {"vocab_size": len(token_to_idx), "embeddings": {}, "tfidf": {}}

    # TF-IDF + SVD
    if args.do_tfidf:
        print("\nFitting TF-IDF + TruncatedSVD")
        tfidf, svd, X_train_sparse, X_train_reduced = fit_tfidf_and_svd(train_texts, max_features=args.max_features, n_components=args.n_components)
        # save vectorizer + svd
        with open(osp.join(args.out_dir, "tfidf_vectorizer.pkl"), "wb") as f:
            pickle.dump(tfidf, f)
        with open(osp.join(args.out_dir, "tfidf_svd.pkl"), "wb") as f:
            pickle.dump(svd, f)
        summary["tfidf"]["n_features"] = X_train_sparse.shape[1]
        summary["tfidf"]["n_components"] = args.n_components
        summary["tfidf"]["explained_variance"] = float(svd.explained_variance_ratio_.sum())
        print("TF-IDF features:", X_train_sparse.shape[1])
        print("SVD kept:", args.n_components, "explained variance:", svd.explained_variance_ratio_.sum())

        # optionally save per-row reduced tfidf for splits
        if args.save_tfidf_npy:
            out_tfidf_dir = osp.join(args.out_dir, "tfidf_npy")
            if args.train_csv:
                cnt = transform_and_save_tfidf_for_split(args.train_csv, tfidf, svd, out_tfidf_dir)
                print("Saved TF-IDF reduced for train:", cnt)
            if args.val_csv:
                cnt = transform_and_save_tfidf_for_split(args.val_csv, tfidf, svd, out_tfidf_dir)
                print("Saved TF-IDF reduced for val:", cnt)
            if args.test_csv:
                cnt = transform_and_save_tfidf_for_split(args.test_csv, tfidf, svd, out_tfidf_dir)
                print("Saved TF-IDF reduced for test:", cnt)

    # GloVe
    if args.glove:
        print("\nLoading GloVe and building matrix...")
        glove_found, glove_dim = load_glove_vectors(args.glove, token_set)
        if glove_dim is None:
            raise RuntimeError("GloVe dim could not be inferred. Check the file.")
        emb_glove, covered_glove = build_embedding_matrix(token_to_idx, glove_found, glove_dim, seed=args.seed)
        out_glove = osp.join(args.out_dir, f"emb_glove_{glove_dim}d.npy")
        np.save(out_glove, emb_glove)
        print("Saved GloVe matrix to:", out_glove)
        summary["embeddings"]["glove"] = {
            "path": out_glove,
            "dim": int(glove_dim),
            "token_coverage": token_level_coverage(token_to_idx, glove_found),
            "occurrence_coverage": occurrence_coverage(train_tokens, glove_found),
            "found_tokens": int(len(glove_found))
        }

    # FastText
    if args.fasttext:
        print("\nLoading FastText and building matrix...")
        ft_found, ft_dim = load_fasttext_vectors(args.fasttext, token_set, max_load=args.fasttext_max_load)
        if ft_dim is None:
            raise RuntimeError("FastText dim could not be inferred. Check the file.")
        emb_ft, covered_ft = build_embedding_matrix(token_to_idx, ft_found, ft_dim, seed=args.seed)
        out_ft = osp.join(args.out_dir, f"emb_fasttext_{ft_dim}d.npy")
        np.save(out_ft, emb_ft)
        print("Saved FastText matrix to:", out_ft)
        summary["embeddings"]["fasttext"] = {
            "path": out_ft,
            "dim": int(ft_dim),
            "token_coverage": token_level_coverage(token_to_idx, ft_found),
            "occurrence_coverage": occurrence_coverage(train_tokens, ft_found),
            "found_tokens": int(len(ft_found))
        }

    # save summary
    out_summary = osp.join(args.out_dir, "representation_summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved summary:", out_summary)
    print("Done.")

# --------------------------
# Run without CLI
# --------------------------

if __name__ == "__main__":
    # convert CONFIG dict into an args-like object
    args = SimpleNamespace(**CONFIG)

    # Basic checks & helpful messages
    required_files = [args.vocab, args.train_csv]
    missing = [p for p in required_files if p is None or not osp.exists(p)]
    if missing:
        print("Warning: the following required files are missing or not set:", missing)
        print("Make sure you update the CONFIG at the top of this script with correct paths.")
    main(args)

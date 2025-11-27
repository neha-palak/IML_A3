#!/usr/bin/env python3
"""
text_embeddings.py

Build embedding matrices from pretrained vectors (GloVe / FastText) aligned to your vocab,
compute coverage metrics, and save .npy + a summary json.

Usage examples:
python scripts/text_embeddings.py \
  --vocab data_preprocessed/vocab.pkl \
  --train-csv data_preprocessed/train.csv \
  --glove data/glove.6B.300d.txt \
  --out-dir data_preprocessed

# If you also downloaded FastText:
python scripts/text_embeddings.py --vocab data_preprocessed/vocab.pkl --train-csv data_preprocessed/train.csv \
  --glove data/glove.6B.300d.txt --fasttext data/fasttext/wiki-news-300d-1M-subword.vec \
  --out-dir data_preprocessed
"""
import argparse
import os
import os.path as osp
import pickle
import numpy as np
import json
from collections import Counter
from tqdm import tqdm

def load_vocab(path):
    with open(path, "rb") as f:
        token_to_idx = pickle.load(f)
    return token_to_idx

def load_train_tokens(train_csv, tokens_col="tokens"):
    import pandas as pd
    df = pd.read_csv(train_csv)
    # Attempt to handle tokens stored as Python lists (string) or actual lists
    tokens_list = []
    for t in df[tokens_col].tolist():
        if isinstance(t, str) and t.startswith("[") and t.endswith("]"):
            # e.g. "['<start>', 'a', 'man', '<end>']"
            try:
                lst = eval(t)
            except Exception:
                lst = t.split()
        elif isinstance(t, str):
            # either space-joined tokens or JSON-like
            # prefer splitting on space (our preprocessing uses a list, but be robust)
            # if tokenization contains commas, fallback to eval
            if ",'" in t or '", ' in t:
                try:
                    lst = eval(t)
                except Exception:
                    lst = t.split()
            else:
                lst = t.split()
        elif isinstance(t, (list, tuple)):
            lst = list(t)
        else:
            lst = [str(t)]
        tokens_list.append(lst)
    return tokens_list

def build_embedding_from_glove(glove_path, token_set, emb_dim=None):
    """
    Efficiently scan GloVe text file and keep only tokens in token_set.
    Returns: dict token -> vector (np.array), embedding_dim
    """
    found = {}
    with open(glove_path, "rt", encoding="utf8", errors="ignore") as f:
        for line in tqdm(f, desc=f"Reading GloVe ({osp.basename(glove_path)})"):
            parts = line.rstrip().split(" ")
            token = parts[0]
            if token in token_set:
                vec = np.asarray(parts[1:], dtype=np.float32)
                found[token] = vec
                if emb_dim is None:
                    emb_dim = vec.shape[0]
            # small optimization: break early if we found all tokens
            if len(found) == len(token_set):
                break
    return found, emb_dim

def build_embedding_from_fasttext_vec(ft_path, token_set, emb_dim=None, max_load=None):
    """
    Load FastText .vec (word2vec-style text) file. Similar to GloVe loader.
    """
    found = {}
    with open(ft_path, "rt", encoding="utf8", errors="ignore") as f:
        # first line may contain header "2000000 300"
        first = f.readline()
        if len(first.split()) == 2 and first.split()[1].isdigit():
            # header present, continue
            pass
        else:
            # header absent, the first line is a token -> rewind
            f.seek(0)
        for i, line in enumerate(tqdm(f, desc=f"Reading FastText ({osp.basename(ft_path)})")):
            parts = line.rstrip().split(" ")
            token = parts[0]
            if token in token_set:
                vec = np.asarray(parts[1:], dtype=np.float32)
                found[token] = vec
                if emb_dim is None:
                    emb_dim = vec.shape[0]
            if max_load and i > max_load:
                break
            if len(found) == len(token_set):
                break
    return found, emb_dim

def make_embedding_matrix(token_to_idx, found_embeddings, emb_dim, seed=42):
    rng = np.random.RandomState(seed)
    vocab_size = len(token_to_idx)
    mat = rng.normal(scale=0.6, size=(vocab_size, emb_dim)).astype("float32")
    covered = 0
    for tok, idx in token_to_idx.items():
        vec = found_embeddings.get(tok)
        if vec is not None:
            mat[idx] = vec
            covered += 1
    return mat, covered

def occurrence_weighted_coverage(tokens_lists, found_embeddings):
    total = 0
    covered = 0
    for toks in tokens_lists:
        for tok in toks:
            total += 1
            if tok in found_embeddings:
                covered += 1
    if total == 0:
        return 0.0
    return covered / total

def token_level_coverage(token_to_idx, found_embeddings):
    covered = sum(1 for t in token_to_idx if t in found_embeddings)
    return covered / len(token_to_idx) if len(token_to_idx) > 0 else 0.0

def save_npy(mat, out_path):
    np.save(out_path, mat)
    print("Saved:", out_path)

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    print("Loading vocab from:", args.vocab)
    token_to_idx = load_vocab(args.vocab)
    print("Vocab size:", len(token_to_idx))

    # prepare token set to look up (exclude special tokens if you prefer include them)
    token_set = set(token_to_idx.keys())

    # load training tokens for occurrence-weighted coverage
    tokens_lists = []
    if args.train_csv:
        print("Loading train tokens from:", args.train_csv)
        tokens_lists = load_train_tokens(args.train_csv, tokens_col=args.tokens_col)
        print("Train rows tokens loaded:", len(tokens_lists))

    summary = {
        "vocab_size": len(token_to_idx),
        "embeddings": {}
    }

    # ----- GloVe -----
    if args.glove:
        print("\nProcessing GloVe:", args.glove)
        glove_found, glove_dim = build_embedding_from_glove(args.glove, token_set)
        if glove_dim is None:
            raise RuntimeError("Couldn't infer GloVe embedding dim - check file.")
        mat_glove, covered = make_embedding_matrix(token_to_idx, glove_found, glove_dim, seed=args.seed)
        out_glove = osp.join(args.out_dir, f"emb_glove_{glove_dim}d.npy")
        save_npy(mat_glove, out_glove)

        token_cov = token_level_coverage(token_to_idx, glove_found)
        occ_cov = occurrence_weighted_coverage(tokens_lists, glove_found) if tokens_lists else None
        print(f"GloVe coverage token-level: {token_cov:.4f}, occurrence-weighted: {occ_cov}")
        summary["embeddings"]["glove"] = {
            "path": out_glove,
            "dim": glove_dim,
            "token_coverage": token_cov,
            "occurrence_coverage": occ_cov,
            "found_tokens": len(glove_found)
        }

    # ----- FastText (optional) -----
    if args.fasttext:
        print("\nProcessing FastText:", args.fasttext)
        ft_found, ft_dim = build_embedding_from_fasttext_vec(args.fasttext, token_set, emb_dim=None, max_load=args.fasttext_max_load)
        if ft_dim is None:
            raise RuntimeError("Couldn't infer FastText embedding dim - check file.")
        mat_ft, covered_ft = make_embedding_matrix(token_to_idx, ft_found, ft_dim, seed=args.seed)
        out_ft = osp.join(args.out_dir, f"emb_fasttext_{ft_dim}d.npy")
        save_npy(mat_ft, out_ft)

        token_cov = token_level_coverage(token_to_idx, ft_found)
        occ_cov = occurrence_weighted_coverage(tokens_lists, ft_found) if tokens_lists else None
        print(f"FastText coverage token-level: {token_cov:.4f}, occurrence-weighted: {occ_cov}")
        summary["embeddings"]["fasttext"] = {
            "path": out_ft,
            "dim": ft_dim,
            "token_coverage": token_cov,
            "occurrence_coverage": occ_cov,
            "found_tokens": len(ft_found)
        }

    # save summary
    summary_path = osp.join(args.out_dir, "representation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote summary:", summary_path)
    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=str, required=True, help="path to vocab.pkl (token->idx)")
    p.add_argument("--train-csv", type=str, default=None, help="train.csv for occurrence-weighted coverage")
    p.add_argument("--tokens-col", type=str, default="tokens", help="column name that contains token lists in train CSV")
    p.add_argument("--glove", type=str, default=None, help="path to glove .txt file (e.g. glove.6B.300d.txt)")
    p.add_argument("--fasttext", type=str, default=None, help="path to fasttext .vec (optional)")
    p.add_argument("--fasttext-max-load", type=int, default=None, help="limit lines read from fasttext (optional)")
    p.add_argument("--out-dir", type=str, required=True, help="where to save emb .npy and summary")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)

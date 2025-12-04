import os
import os.path as osp
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def load_vocab(vocab_path):
    with open(vocab_path, "rb") as f:
        tok2idx = pickle.load(f)
    if not isinstance(tok2idx, dict) and hasattr(tok2idx, "token_to_idx"):
        tok2idx = tok2idx.token_to_idx
    idx2tok = {i: t for t, i in tok2idx.items()}
    return tok2idx, idx2tok

def build_tfidf_embedding_matrix(
    tok2idx,
    tfidf_vectorizer_path,
    svd_path,
    dim=None,
):
    with open(tfidf_vectorizer_path, "rb") as f:
        tfidf: TfidfVectorizer = pickle.load(f)
    with open(svd_path, "rb") as f:
        svd: TruncatedSVD = pickle.load(f)

    vocab_size = len(tok2idx)
    d = svd.n_components if dim is None else min(dim, svd.n_components)
    emb = np.zeros((vocab_size, d), dtype="float32")

    # we iterate in idx order so that emb[idx] matches token id
    idx_to_token = {i: t for t, i in tok2idx.items()}

    for idx in tqdm(range(vocab_size), desc="Building TF-IDF embedding matrix"):
        tok = idx_to_token[idx]
        doc = tok
        X = tfidf.transform([doc]) #(1, n_features)sparse
        Xr = svd.transform(X)[:, :d] #(1, d)
        emb[idx] = Xr[0].astype("float32")

    return emb, d

def get_embedding_matrix(
    embedding_type,
    vocab_path="new_preprocessed/vocab.pkl",
    repr_dir="new_preprocessed",
):
    """
    embedding_type: "glove", "fasttext", "tfidf"
    """
    tok2idx, idx2tok = load_vocab(vocab_path)
    vocab_size = len(tok2idx)

    if embedding_type == "glove":
        path = osp.join(repr_dir, "emb_glove_300d.npy")
        mat = np.load(path).astype("float32")
        assert mat.shape[0] == vocab_size, \
            f"GloVe matrix vocab size {mat.shape[0]} != {vocab_size}"
        return mat, mat.shape[1], tok2idx, idx2tok

    elif embedding_type == "fasttext":
        path = osp.join(repr_dir, "emb_fasttext_300d.npy")
        mat = np.load(path).astype("float32")
        assert mat.shape[0] == vocab_size, \
            f"FastText matrix vocab size {mat.shape[0]} != {vocab_size}"
        return mat, mat.shape[1], tok2idx, idx2tok

    elif embedding_type == "tfidf":
        tfidf_path = osp.join(repr_dir, "tfidf_vectorizer.pkl")
        svd_path = osp.join(repr_dir, "tfidf_svd.pkl")
        mat, d = build_tfidf_embedding_matrix(tok2idx, tfidf_path, svd_path)
        return mat, d, tok2idx, idx2tok

    else:
        raise ValueError(f"Unknown embedding_type: {embedding_type}")
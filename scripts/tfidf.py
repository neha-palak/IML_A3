# scripts/tfidf_pipeline.py
import pickle, os, json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

OUT_DIR = "data_preprocessed"
TFIDF_VECT = os.path.join(OUT_DIR, "tfidf_vectorizer.pkl")
SVD_MODEL = os.path.join(OUT_DIR, "tfidf_svd.pkl")
TFIDF_NPY_DIR = os.path.join(OUT_DIR, "tfidf_npy")   # per-row tfidf reduced vectors

def fit_tfidf_and_svd(train_csv, max_features=20000, n_components=512):
    df = pd.read_csv(train_csv)
    texts = df['utter_clean'].fillna("").tolist()

    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1,1))
    print("Fitting TF-IDF ...")
    X = tfidf.fit_transform(texts)   # sparse (N, features)
    print("TF-IDF shape:", X.shape)

    print("Fitting TruncatedSVD ...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Xr = svd.fit_transform(X)  # dense (N, n_components)
    print("SVD explained variance:", svd.explained_variance_ratio_.sum())

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(TFIDF_VECT, "wb") as f:
        pickle.dump(tfidf, f)
    with open(SVD_MODEL, "wb") as f:
        pickle.dump(svd, f)
    print("Saved vectorizer & SVD.")
    return tfidf, svd, Xr, df

def transform_and_save_all(csv_path, tfidf, svd, out_dir=TFIDF_NPY_DIR):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    texts = df['utter_clean'].fillna("").tolist()
    X = tfidf.transform(texts)
    Xr = svd.transform(X)
    # Save per-row vectors: name by row index for easy mapping
    for i, vec in enumerate(tqdm(Xr, desc="Saving TFIDF npy")):
        np.save(os.path.join(out_dir, f"{i}.npy"), vec.astype("float32"))
    print("Saved", len(Xr), "reduced TF-IDF vectors to", out_dir)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", default="data_preprocessed/train.csv")
    p.add_argument("--n-components", type=int, default=512)
    p.add_argument("--max-features", type=int, default=20000)
    args = p.parse_args()

    tfidf, svd, Xr, df_train = fit_tfidf_and_svd(args.train_csv, max_features=args.max_features, n_components=args.n_components)
    # transform all splits and save them
    for split in ["train", "val", "test"]:
        csv = os.path.join("data_preprocessed", f"{split}.csv")
        if os.path.exists(csv):
            transform_and_save_all(csv, tfidf, svd)

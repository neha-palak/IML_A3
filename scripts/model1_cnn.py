#!/usr/bin/env python3
"""
model_cnn1.py

Train Model 1: Custom CNN encoder + LSTM decoder with emotion conditioning.

Supports word representation choices:
  - glove: loads data_preprocessed/emb_glove_...npy
  - fasttext: loads data_preprocessed/emb_fasttext_...npy
  - tfidf: uses per-row tfidf vectors saved as data_preprocessed/tfidf_npy/{split}__{idx:06d}.npy

Assumptions (match your preprocessing):
  - CSV splits: data_preprocessed/train.csv, val.csv, test.csv
  - Each CSV row has: painting (filename without .jpg), token_ids (python list string), emotion_label (int)
  - Image features saved as: data_preprocessed/features/{painting}.npy (H,W,C normalized to [0,1])
  - Vocab at data_preprocessed/vocab.pkl, pad token index = token_to_idx["<pad>"]
"""
import os
import argparse
import json
from pathlib import Path
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# -----------------------
# CONFIG (defaults)
# -----------------------
DATA_DIR = "data_preprocessed"
FEATURES_DIR = os.path.join(DATA_DIR, "features")
TFIDF_DIR = os.path.join(DATA_DIR, "tfidf_npy")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.pkl")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
CHECKPOINT_DIR = "checkpoints/m1"
HISTORY_PATH = os.path.join(CHECKPOINT_DIR, "history.json")

# Training hyperparams
NUM_EPOCHS = 4
BATCH_SIZE = 32
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 25  # matches your preprocessing MAX_LEN
PRINT_SAMPLES = 3

# Model defaults (can be overridden via CLI args)
DEFAULT_EMBED_DIM = 300
IMAGE_FEAT_DIM = 256
EMO_DIM = 64
LSTM_HIDDEN = 256
EMOTION_CLASSES = 9

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -----------------------
# Dataset
# -----------------------
class ArtEmisDataset(Dataset):
    def __init__(self, csv_path, features_dir, split_name,
                 use_tfidf=False, tfidf_dir=None, max_len=MAX_SEQ_LEN):
        """
        split_name: "train" or "val" or "test"  -> used to find tfidf files: {split_name}__{idx:06d}.npy
        """
        self.df = pd.read_csv(csv_path)
        self.features_dir = Path(features_dir)
        self.use_tfidf = use_tfidf
        self.tfidf_dir = Path(tfidf_dir) if tfidf_dir is not None else None
        self.split_name = split_name
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def _load_image(self, painting):
        p = self.features_dir / f"{painting}.npy"
        if not p.exists():
            # fallback black image 224x224
            arr = np.zeros((224, 224, 3), dtype=np.float32)
            return torch.tensor(arr).permute(2, 0, 1)
        arr = np.load(p)  # H,W,C (normalized)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        # ensure shape H,W,C
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        return torch.tensor(arr).permute(2, 0, 1)  # C,H,W

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        painting = row["painting"]
        img = self._load_image(painting)

        # token_ids column: may be saved as "[1,2,3]" string
        toks = row.get("token_ids")
        if isinstance(toks, str):
            try:
                token_ids = ast.literal_eval(toks)
            except Exception:
                token_ids = [int(x) for x in toks.split() if x.isdigit()]
        elif isinstance(toks, (list, tuple, np.ndarray)):
            token_ids = list(toks)
        else:
            token_ids = []

        # pad/truncate to max_len
        token_ids = token_ids[: self.max_len]
        if len(token_ids) < self.max_len:
            token_ids = token_ids + [0] * (self.max_len - len(token_ids))

        token_ids = torch.tensor(token_ids, dtype=torch.long)

        emo = int(row["emotion_label"])

        tfidf_vec = None
        if self.use_tfidf and self.tfidf_dir is not None:
            tfidf_path = self.tfidf_dir / f"{self.split_name}__{idx:06d}.npy"
            if tfidf_path.exists():
                tfidf_vec = np.load(tfidf_path).astype(np.float32)
                tfidf_vec = torch.tensor(tfidf_vec, dtype=torch.float32)
            else:
                # fallback zero
                tfidf_vec = torch.zeros((1,), dtype=torch.float32)

        return img, token_ids, torch.tensor(emo, dtype=torch.long), tfidf_vec


def collate_fn(batch):
    imgs, toks, emos, tfidfs = zip(*batch)
    imgs = torch.stack(imgs)
    toks = torch.stack(toks)
    emos = torch.stack(emos)
    if any(t is not None for t in tfidfs):
        # If some tfidf is None, replace with zeros of appropriate dim
        processed = []
        max_len = max((t.numel() if t is not None else 0) for t in tfidfs)
        if max_len == 0:
            tfidf_batch = None
        else:
            for t in tfidfs:
                if t is None:
                    processed.append(torch.zeros(max_len, dtype=torch.float32))
                else:
                    # pad to max_len if needed
                    if t.numel() < max_len:
                        pad = torch.zeros(max_len - t.numel(), dtype=torch.float32)
                        processed.append(torch.cat([t.flatten(), pad]))
                    else:
                        processed.append(t.flatten()[:max_len])
            tfidf_batch = torch.stack(processed)
    else:
        tfidf_batch = None

    return imgs, toks, emos, tfidf_batch

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNNEncoder(nn.Module):
    """
    A custom CNN from scratch.
    Output: A 256-dimensional feature vector for each image.
    """

    def __init__(self, out_dim=256):
        super().__init__()

        # Input expected: (3, 224, 224)
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14

            # Block 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )

        # Final linear projection -> 256D vector
        self.fc = nn.Linear(256 * 7 * 7, out_dim)

    def forward(self, x):
        """
        x: (B, 3, 224, 224)
        """
        feat = self.conv_layers(x)              # (B, 256, 7, 7)
        feat = feat.view(feat.size(0), -1)      # flatten
        feat = self.fc(feat)                    # (B, out_dim)
        return feat


# -----------------------
# Model: encoder + decoder (supports tfidf-mode for word representation)
# -----------------------
class CaptionModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_type="glove",  # 'glove','fasttext','tfidf','trainable'
        embedding_matrix_path=None,  # for glove/fasttext
        tfidf_dim=None,  # if using tfidf as extra or main rep
        embed_dim=DEFAULT_EMBED_DIM if "DEFAULT_EMBED_DIM" in globals() else 300,
        image_feat_dim=IMAGE_FEAT_DIM,
        emo_dim=EMO_DIM,
        lstm_hidden=LSTM_HIDDEN,
        num_emotions=EMOTION_CLASSES,
    ):
        super().__init__()
        self.embedding_type = embedding_type
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.tfidf_dim = tfidf_dim if tfidf_dim is not None else 0

        # image encoder
        self.encoder = CustomCNNEncoder(out_dim=image_feat_dim)

        # If embedding_type is 'tfidf' we will use a projection of per-row TF-IDF to embed_dim
        if embedding_type == "tfidf":
            if self.tfidf_dim == 0:
                raise ValueError("tfidf_dim must be >0 when embedding_type=='tfidf'")
            # tfidf projection -> to embed_dim
            self.tfidf_proj = nn.Linear(self.tfidf_dim, embed_dim)
            # create a trainable dummy token embedding only used for decoding (we still need token_emb for sos/pad)
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        else:
            # word embeddings lookup
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)

            if embedding_type in ("glove", "fasttext"):
                if embedding_matrix_path is None:
                    raise ValueError("embedding_matrix_path required for glove/fasttext")
                emb = np.load(embedding_matrix_path)
                if emb.shape[0] != vocab_size:
                    # try to align sizes: if vocab smaller/larger, allow partial copy
                    k = min(emb.shape[0], vocab_size)
                    temp = np.random.normal(scale=0.6, size=(vocab_size, emb.shape[1])).astype(np.float32)
                    temp[:k, :emb.shape[1]] = emb[:k, :emb.shape[1]]
                    emb = temp
                self.token_embedding.weight.data.copy_(torch.tensor(emb, dtype=torch.float32))
                # allow fine-tuning
                self.token_embedding.weight.requires_grad = True

        # emotion embedding
        self.emo_embedding = nn.Embedding(num_emotions, emo_dim)

        # LSTM input dim:
        # - token representation (embed_dim) OR tfidf_proj output (embed_dim)
        # - image_feat_dim
        # - emo_dim
        lstm_input_dim = embed_dim + image_feat_dim + emo_dim
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(lstm_hidden, vocab_size)

    def forward(self, imgs, token_ids, emo_ids, tfidf_vecs=None):
        """
        imgs: (B, C, H, W)
        token_ids: (B, T)
        emo_ids: (B,)
        tfidf_vecs: (B, tfidf_dim) or None
        """
        B, T = token_ids.shape
        # image features
        img_feats = self.encoder(imgs)  # (B, image_feat_dim)

        # token representation
        if self.embedding_type == "tfidf":
            # tfidf_vecs must be provided per sample; project to embed_dim and repeat across T
            if tfidf_vecs is None:
                # zero
                proj = torch.zeros(B, self.embed_dim, device=token_ids.device)
            else:
                proj = self.tfidf_proj(tfidf_vecs)
            token_rep = proj.unsqueeze(1).expand(-1, T, -1)  # (B,T,embed_dim)
        else:
            token_rep = self.token_embedding(token_ids)  # (B,T,embed_dim)

        # emotion embeddings (expand across time)
        emo_rep = self.emo_embedding(emo_ids).unsqueeze(1).expand(-1, T, -1)

        # image features expanded
        img_rep = img_feats.unsqueeze(1).expand(-1, T, -1)

        lstm_input = torch.cat([token_rep, img_rep, emo_rep], dim=-1)  # (B,T,input_dim)
        out, _ = self.lstm(lstm_input)
        out = self.dropout(out)
        logits = self.fc(out)  # (B,T,vocab)
        return logits

    def greedy_decode(self, img, emo_id, sos_idx, eos_idx=None, max_len=MAX_SEQ_LEN, tfidf_vec=None, device="cpu"):
        """
        Greedy decoding: feed last predicted token each step.
        Returns list of token ids (without sos).
        """
        self.eval()
        with torch.no_grad():
            img = img.to(device).unsqueeze(0)  # (1,C,H,W)
            img_feat = self.encoder(img)  # (1, image_feat_dim)
            emo = torch.tensor([emo_id], dtype=torch.long, device=device)
            generated = [sos_idx]
            hidden = None
            for t in range(max_len):
                cur_ids = torch.tensor([generated], dtype=torch.long, device=device)  # (1,t+1)
                if self.embedding_type == "tfidf":
                    tf_proj = None
                    if tfidf_vec is not None:
                        tf_proj = self.tfidf_proj(tfidf_vec.unsqueeze(0).to(device))
                else:
                    tf_proj = None
                logits = self.forward(img.repeat(1,1,1,1) if False else img, cur_ids, emo, tfidf_vecs=tfidf_vec.unsqueeze(0) if tfidf_vec is not None else None)
                # logits shape (1, seq_len, vocab); take last
                next_logits = logits[:, -1, :]
                next_id = torch.argmax(next_logits, dim=-1).item()
                generated.append(next_id)
                if eos_idx is not None and next_id == eos_idx:
                    break
            # remove initial sos
            return generated[1:]


# -----------------------
# Utilities
# -----------------------
def load_vocab(path):
    import pickle
    with open(path, "rb") as f:
        tok2idx = pickle.load(f)
    return tok2idx


# -----------------------
# Training
# -----------------------
def train_main(args):
    device = torch.device(DEVICE)
    print("Using device:", device)

    # load vocab
    vocab = load_vocab(VOCAB_PATH)
    vocab_size = len(vocab)
    pad_idx = vocab.get("<pad>", 0)
    sos_idx = vocab.get("<start>", 1)
    eos_idx = vocab.get("<end>", None)

    print(f"Vocab size: {vocab_size}, pad_idx={pad_idx}, sos_idx={sos_idx}, eos_idx={eos_idx}")

    # datasets
    use_tfidf = args.embedding == "tfidf"
    train_ds = ArtEmisDataset(TRAIN_CSV, FEATURES_DIR, "train", use_tfidf, TFIDF_DIR, max_len=MAX_SEQ_LEN)
    val_ds = ArtEmisDataset(VAL_CSV, FEATURES_DIR, "val", use_tfidf, TFIDF_DIR, max_len=MAX_SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # pick embedding matrix path if needed
    emb_path = None
    if args.embedding == "glove":
        emb_path = os.path.join(DATA_DIR, "emb_glove_300d.npy")
    elif args.embedding == "fasttext":
        emb_path = os.path.join(DATA_DIR, "emb_fasttext_300d.npy")

    # TF-IDF dim (infer from first train tfidf file)
    tfidf_dim = 0
    if use_tfidf:
        # find one example
        sample = Path(TFIDF_DIR) / "train__000000.npy"
        if sample.exists():
            tfidf_dim = int(np.load(sample).shape[0])
            print("Detected TF-IDF dim:", tfidf_dim)
        else:
            # fallback: try to detect first file
            files = list(Path(TFIDF_DIR).glob("train__*.npy"))
            if files:
                tfidf_dim = int(np.load(files[0]).shape[0])
                print("Detected TF-IDF dim:", tfidf_dim)
            else:
                raise RuntimeError("TF-IDF files not found in " + TFIDF_DIR)

    # model
    model = CaptionModel(
        vocab_size=vocab_size,
        embedding_type=args.embedding,
        embedding_matrix_path=emb_path,
        tfidf_dim=tfidf_dim if use_tfidf else None,
        embed_dim=args.embed_dim,
        image_feat_dim=IMAGE_FEAT_DIM,
        emo_dim=args.emo_dim,
        lstm_hidden=args.lstm_hidden,
        num_emotions=EMOTION_CLASSES,
    ).to(device)

    # loss + optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        # training
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc="Train", ncols=120)
        for imgs, toks, emos, tfidf in pbar:
            imgs = imgs.to(device)
            toks = toks.to(device)
            emos = emos.to(device)
            if tfidf is not None:
                tfidf = tfidf.to(device)

            optimizer.zero_grad()
            logits = model(imgs, toks[:, :-1], emos, tfidf_vecs=(tfidf if args.embedding == "tfidf" else tfidf))
            # logits shape (B, T-1, V) because input tokens exclude last token?
            # We passed toks[:, :-1] so targets are toks[:, 1:]
            B, Tm, V = logits.shape
            targets = toks[:, 1:].contiguous()
            loss = criterion(logits.view(-1, V), targets.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": loss.item()})

        avg_train = running_loss / len(train_loader.dataset)
        print(f"  Avg train loss: {avg_train:.4f}")
        history["train_loss"].append(avg_train)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Val", ncols=120)
            for imgs, toks, emos, tfidf in pbar:
                imgs = imgs.to(device)
                toks = toks.to(device)
                emos = emos.to(device)
                if tfidf is not None:
                    tfidf = tfidf.to(device)
                logits = model(imgs, toks[:, :-1], emos, tfidf_vecs=(tfidf if args.embedding == "tfidf" else tfidf))
                B, Tm, V = logits.shape
                targets = toks[:, 1:].contiguous()
                loss = criterion(logits.view(-1, V), targets.view(-1))
                val_loss += loss.item() * imgs.size(0)
                pbar.set_postfix({"loss": loss.item()})
        avg_val = val_loss / len(val_loader.dataset)
        print(f"  Avg val loss: {avg_val:.4f}")
        history["val_loss"].append(avg_val)

        # save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "config": {
                "embedding": args.embedding,
                "embed_dim": args.embed_dim,
                "lstm_hidden": args.lstm_hidden,
                "tfidf_dim": tfidf_dim
            }
        }
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"m1_epoch{epoch}.pt")
        torch.save(ckpt, ckpt_path)
        print("Saved checkpoint:", ckpt_path)

        # save best
        if avg_val < best_val:
            best_val = avg_val
            best_path = os.path.join(CHECKPOINT_DIR, "m1_best.pt")
            torch.save(ckpt, best_path)
            print("Saved BEST model ->", best_path)

        # save history json
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)

    print("\nTraining finished. Best val loss:", best_val)
    print("History saved to:", HISTORY_PATH)
    return


# -----------------------
# EVALUATION helper (quick)
# -----------------------
def evaluate_quick(checkpoint_path, embedding, max_samples=20):
    device = torch.device(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    # reconstruct model
    emb_path = None
    if embedding == "glove":
        emb_path = os.path.join(DATA_DIR, "emb_glove_300d.npy")
    elif embedding == "fasttext":
        emb_path = os.path.join(DATA_DIR, "emb_fasttext_300d.npy")
    tfidf_dim = config.get("tfidf_dim", 0)

    model = CaptionModel(
        vocab_size=len(load_vocab(VOCAB_PATH)),
        embedding_type=embedding,
        embedding_matrix_path=emb_path,
        tfidf_dim=tfidf_dim if tfidf_dim > 0 else None,
        embed_dim=config.get("embed_dim", DEFAULT_EMBED_DIM if "DEFAULT_EMBED_DIM" in globals() else 300),
        image_feat_dim=IMAGE_FEAT_DIM,
        emo_dim=EMO_DIM,
        lstm_hidden=config.get("lstm_hidden", LSTM_HIDDEN),
        num_emotions=EMOTION_CLASSES
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # load vocab map
    tok2idx = load_vocab(VOCAB_PATH)
    idx2tok = {v: k for k, v in tok2idx.items()}
    pad_idx = tok2idx.get("<pad>", 0)
    sos_idx = tok2idx.get("<start>", 1)
    eos_idx = tok2idx.get("<end>", None)

    # test loader
    test_ds = ArtEmisDataset(TEST_CSV, FEATURES_DIR, "test", use_tfidf=(embedding == "tfidf"), tfidf_dir=TFIDF_DIR)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    bleu_smooth = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    bleu_scores = []
    rouge_scores = []
    printed = 0

    for i, (imgs, toks, emos, tfidf) in enumerate(tqdm(test_loader, desc="Eval", total=min(len(test_ds), max_samples))):
        if i >= max_samples:
            break
        imgs = imgs.to(device)
        emos = emos.to(device)
        tfidf = tfidf.to(device) if tfidf is not None else None

        # greedy decode by feeding sos and using model.forward step-by-step
        # For simplicity we reuse greedy_decode (which expects single image)
        img_np = imgs[0]
        emo_id = int(emos[0].cpu().item())
        tfidf_vec = tfidf[0] if tfidf is not None else None
        out_ids = model.greedy_decode(img_np, emo_id, sos_idx, eos_idx=eos_idx, max_len=MAX_SEQ_LEN, tfidf_vec=(tfidf_vec if embedding=="tfidf" else None), device=device)
        # convert ids to tokens
        pred_tokens = [idx2tok.get(int(x), "<unk>") for x in out_ids]
        ref_ids = toks[0].tolist()
        ref_tokens = [idx2tok.get(int(x), "<unk>") for x in ref_ids if x not in (pad_idx, sos_idx, eos_idx)]
        # compute BLEU, ROUGE-L
        try:
            bleu_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=bleu_smooth)
        except Exception:
            bleu_score = 0.0
        rouge_score = rouge.score(" ".join(ref_tokens), " ".join(pred_tokens))['rougeL'].fmeasure
        bleu_scores.append(bleu_score)
        rouge_scores.append(rouge_score)

        if printed < 5:
            print("\nSample", i)
            print("Ref:", " ".join(ref_tokens))
            print("Pred:", " ".join(pred_tokens))
            printed += 1

    print("\nAvg BLEU:", float(np.mean(bleu_scores)) if bleu_scores else 0.0)
    print("Avg ROUGE-L:", float(np.mean(rouge_scores)) if rouge_scores else 0.0)


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", choices=["glove", "fasttext", "tfidf", "trainable"], default="glove",
                        help="Word representation choice.")
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--lstm_hidden", type=int, default=256)
    parser.add_argument("--emo_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    # override globals from args
    NUM_EPOCHS = args.epochs
    DEFAULT_EMBED_DIM = args.embed_dim
    LSTM_HIDDEN = args.lstm_hidden
    EMO_DIM = args.emo_dim

    # run training
    train_main(args)

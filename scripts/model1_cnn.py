#!/usr/bin/env python3
"""
Usage example:
 python model1_cnn.py \
   --train_csv data_preprocessed/train.csv \
   --val_csv data_preprocessed/val.csv \
   --vocab data_preprocessed/vocab.pkl \
   --features_dir data_preprocessed/features \
   --images_dir data_preprocessed/images_subset \
   --out_dir checkpoints/m1_pt \
   --out_dir_final checkpoints/summary \
   --epochs 3 --batch_size 16 --embedding_type glove --embedding_dim 300 --freeze_emb
"""
import os
import os.path as osp
import argparse
import json
import pickle
from pathlib import Path
from glob import glob
from typing import Optional, Tuple

from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ---------------------------
# Utilities
# ---------------------------
def load_vocab(path: str):
    with open(path, "rb") as f:
        token_to_idx = pickle.load(f)
    idx_to_token = {i: t for t, i in token_to_idx.items()}
    return token_to_idx, idx_to_token

def parse_token_ids(cell):
    if pd.isna(cell):
        return []
    if isinstance(cell, str):
        cell = cell.strip()
        if cell.startswith("[") and cell.endswith("]"):
            try:
                return eval(cell)
            except Exception:
                parts = cell.strip("[]").split(",")
                return [int(p) for p in parts if p.strip().isdigit()]
        else:
            parts = cell.split()
            return [int(p) for p in parts if p.isdigit()]
    elif isinstance(cell, (list, tuple, np.ndarray)):
        return list(cell)
    else:
        return []

def find_embedding_file(out_dir: str, which: str, emb_dim: int) -> Optional[str]:
    """
    which: 'glove' or 'fasttext'
    emb_dim: e.g., 300
    looks for patterns like emb_glove_300d.npy or emb_fasttext_300d.npy
    """
    pat = osp.join(out_dir, f"emb_{which}_*{emb_dim}*.npy")
    matches = glob(pat)
    return matches[0] if matches else None

def load_embedding_matrix_from_npy(path: str, expected_vocab_size: int=None) -> np.ndarray:
    mat = np.load(path)
    if expected_vocab_size is not None and mat.shape[0] != expected_vocab_size:
        print(f"Warning: embedding matrix rows {mat.shape[0]} != vocab size {expected_vocab_size}.")
    return mat

# ---------------------------
# Dataset
# ---------------------------
class ArtEmisDataset(Dataset):
    def __init__(self, csv_path: str, token_to_idx: dict, images_dir: Optional[str]=None, features_dir: Optional[str]=None,
                 max_len: int = 20, transform=None):
        self.df = pd.read_csv(csv_path)
        self.token_to_idx = token_to_idx
        self.images_dir = Path(images_dir) if images_dir else None
        self.features_dir = Path(features_dir) if features_dir else None
        self.max_len = max_len
        self.transform = transform or T.Compose([T.Resize((224,224)), T.ToTensor()])
        if 'token_ids' not in self.df.columns:
            raise RuntimeError(f"{csv_path} missing 'token_ids' column")
        # parse token ids
        self.df['token_ids_parsed'] = self.df['token_ids'].apply(parse_token_ids)

    def __len__(self):
        return len(self.df)

    def _load_image_tensor(self, painting: str):
        # Prefer .npy normalized features (H x W x C scaled to [0,1])
        if self.features_dir:
            feat_path = self.features_dir / f"{painting}.npy"
            if feat_path.exists():
                arr = np.load(feat_path).astype("float32")
                if arr.ndim == 3:
                    return torch.from_numpy(arr).permute(2,0,1)  # C,H,W
        # fallback load jpg
        if self.images_dir:
            img_path = self.images_dir / f"{painting}.jpg"
            img = Image.open(img_path).convert("RGB")
            return self.transform(img)
        # fallback zeros
        return torch.zeros(3,224,224, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        painting = str(row['painting'])
        img = self._load_image_tensor(painting)
        ids = row['token_ids_parsed'][:self.max_len]
        if len(ids) < self.max_len:
            pad = self.token_to_idx.get("<pad>", 0)
            ids = ids + [pad] * (self.max_len - len(ids))
        ids = torch.tensor(ids, dtype=torch.long)
        return img, ids

def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    caps = torch.stack([b[1] for b in batch], dim=0)
    return imgs, caps

# ---------------------------
# Model definitions
# ---------------------------
class CNNEncoder(nn.Module):
    def __init__(self, feature_dim=256, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),   # 224->112
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # 112->56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # 56->28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(), # 28->14
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(),# 14->7
            nn.AdaptiveAvgPool2d((1,1))                                       # 1x1
        )
        self.fc = nn.Linear(256, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        f = self.conv(x)           # B x 256 x 1 x 1
        f = f.view(f.size(0), -1)  # B x 256
        f = self.dropout(f)
        return self.fc(f)          # B x feature_dim

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int=256, hidden_dim: int=256, feature_dim: int=256,
                 pad_idx: int=0, bidirectional: bool=False, dropout: float=0.3,
                 embedding_weights: Optional[np.ndarray]=None, freeze_embedding: bool=False):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
            self.embedding.weight.requires_grad = (not freeze_embedding)
        
        self.dropout_in = nn.Dropout(dropout)  # dropout on input embeddings
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim + feature_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.dropout_out = nn.Dropout(dropout)  # dropout on LSTM outputs
        
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc_out = nn.Linear(out_dim, vocab_size)

    def forward(self, captions, features):
        """
        captions: B x T
        features: B x F
        """
        emb = self.embedding(captions)                  # B x T x E
        emb = self.dropout_in(emb)                      # Dropout on embeddings
        
        # repeat image feature vector for all time steps
        F = features.unsqueeze(1).repeat(1, emb.size(1), 1)  # B x T x F
        
        lstm_in = torch.cat([emb, F], dim=-1)          # B x T x (E+F)
        outputs, _ = self.lstm(lstm_in)                # B x T x H
        outputs = self.dropout_out(outputs)            # Dropout on LSTM outputs
        
        logits = self.fc_out(outputs)                  # B x T x V
        return logits

# ---------------------------
# Training & evaluation helpers
# ---------------------------
def train_epoch(device, encoder, decoder, dataloader, optimizer, criterion, pad_idx):
    encoder.train(); decoder.train()
    total_loss = 0.0
    for imgs, caps in tqdm(dataloader, desc="train"):
        imgs = imgs.to(device)
        caps = caps.to(device)
        optimizer.zero_grad()
        feats = encoder(imgs)                       # B x F
        inputs = caps[:, :-1]                       # B x (T-1)
        targets = caps[:, 1:].contiguous().view(-1) # B*(T-1)
        logits = decoder(inputs, feats)             # B x (T-1) x V
        logits_flat = logits.contiguous().view(-1, logits.size(-1))  # B*(T-1) x V
        loss = criterion(logits_flat, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_epoch(device, encoder, decoder, dataloader, criterion, pad_idx):
    encoder.eval(); decoder.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, caps in tqdm(dataloader, desc="val"):
            imgs = imgs.to(device)
            caps = caps.to(device)
            feats = encoder(imgs)
            inputs = caps[:, :-1]
            targets = caps[:, 1:].contiguous().view(-1)
            logits = decoder(inputs, feats)
            logits_flat = logits.contiguous().view(-1, logits.size(-1))
            loss = criterion(logits_flat, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def greedy_decode_single(image_tensor: torch.Tensor, encoder, decoder, token_to_idx: dict, idx_to_token: dict, max_len: int, device: str):
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        img = image_tensor.unsqueeze(0).to(device)
        feat = encoder(img)  # 1 x F
        start_id = token_to_idx.get("<start>")
        end_id = token_to_idx.get("<end>")
        if start_id is None or end_id is None:
            raise RuntimeError("vocab missing <start> or <end>")
        generated = [start_id]
        for _ in range(max_len - 1):
            seq = torch.tensor([generated], dtype=torch.long).to(device)  # 1 x t
            logits = decoder(seq, feat)  # 1 x t x V
            last = logits[0, -1, :]
            next_id = int(torch.argmax(last).cpu().item())
            generated.append(next_id)
            if next_id == end_id:
                break
        toks = [idx_to_token.get(i, "<unk>") for i in generated]
        if toks and toks[0] == "<start>": toks = toks[1:]
        if toks and toks[-1] == "<end>": toks = toks[:-1]
        return toks

def save_checkpoint(state: dict, path: str):
    torch.save(state, path)

# ---------------------------
# Main CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data_preprocessed/train.csv")
    parser.add_argument("--val_csv", default="data_preprocessed/val.csv")
    parser.add_argument("--vocab", default="data_preprocessed/vocab.pkl")
    parser.add_argument("--features_dir", default="data_preprocessed/features")
    parser.add_argument("--images_dir", default="data_preprocessed/images_subset")
    parser.add_argument("--out_dir", default="checkpoints/m1_pt")
    parser.add_argument("--out_dir_final", default="checkpoints/summary")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16) # 32 was too slow 
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--embedding_type", choices=["random","glove","fasttext"], default="fasttext")
    parser.add_argument("--freeze_emb", action="store_true", help="If set, freeze pretrained embeddings")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4) 
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.out_dir_final, exist_ok=True)
    # load vocab
    token_to_idx, idx_to_token = load_vocab(args.vocab)
    vocab_size = len(token_to_idx)
    pad_idx = token_to_idx.get("<pad>", 0)
    print("Vocab size:", vocab_size)

    # load optional pretrained embedding matrix
    embedding_weights = None
    if args.embedding_type in ("glove","fasttext"):
        emb_file = find_embedding_file(osp.dirname(args.vocab), args.embedding_type, args.embed_dim)
        if emb_file is None:
            # also search out_dir
            emb_file = find_embedding_file("data_preprocessed", args.embedding_type, args.embed_dim)
        if emb_file:
            print(f"Loading pretrained embeddings from: {emb_file}")
            mat = load_embedding_matrix_from_npy(emb_file, expected_vocab_size=vocab_size)
            # if rows mismatch, try to handle (if mat has same dim but smaller rows, we'll expand random)
            if mat.shape[0] != vocab_size:
                print("Embedding matrix rows don't match vocab. Creating new matrix and copying intersecting rows (by index).")
                rng = np.random.RandomState(42)
                embedding_weights = rng.normal(scale=0.6, size=(vocab_size, mat.shape[1])).astype("float32")
                # if same vocabulary ordering was used when saving mat, copying by row is correct; otherwise this is best-effort
                m = min(mat.shape[0], vocab_size)
                embedding_weights[:m] = mat[:m]
            else:
                embedding_weights = mat.astype("float32")
        else:
            print(f"Warning: no pretrained embedding file found for {args.embedding_type} with dim {args.embed_dim}. Using random init.")
            embedding_weights = None

    # create datasets
    transform = T.Compose([T.Resize((224,224)), T.ToTensor()])
    train_ds = ArtEmisDataset(args.train_csv, token_to_idx, images_dir=args.images_dir, features_dir=args.features_dir, max_len=args.max_len, transform=transform)
    val_ds = ArtEmisDataset(args.val_csv, token_to_idx, images_dir=args.images_dir, features_dir=args.features_dir, max_len=args.max_len, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    device = torch.device(args.device)
    encoder = CNNEncoder(feature_dim=args.feature_dim).to(device)
    decoder = LSTMDecoder(vocab_size=vocab_size, embed_dim=(embedding_weights.shape[1] if embedding_weights is not None else args.embed_dim),
                          hidden_dim=args.hidden_dim, feature_dim=args.feature_dim,
                          pad_idx=pad_idx, embedding_weights=embedding_weights, freeze_embedding=args.freeze_emb).to(device)

    # If embedding_weights is None but embed_dim argument differs, PyTorch will create embeddings with embed_dim
    # set accordingly; ensure decoder.embedding dimension equals args.embed_dim
    if decoder.embedding.weight.shape[1] != args.embed_dim:
        print(f"Note: decoder embedding dim ({decoder.embedding.weight.shape[1]}) != requested embed_dim ({args.embed_dim}). Using actual embedding dim.")

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss = train_epoch(device, encoder, decoder, train_loader, optimizer, criterion, pad_idx)
        val_loss = eval_epoch(device, encoder, decoder, val_loader, criterion, pad_idx)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        ckpt = {
            "epoch": epoch,
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "vocab": token_to_idx
        }
        ckpt_path = osp.join(args.out_dir, f"m1_epoch{epoch}.pt")
        save_checkpoint(ckpt, ckpt_path)
        print("Saved checkpoint:", ckpt_path)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_train_loss = train_loss
            best_val_loss = val_loss
            best_path = osp.join(args.out_dir, f"m1_best_{args.embedding_type}.pt")
            save_checkpoint(ckpt, best_path)
            print("Saved best checkpoint:", best_path)

        # update history with best info
        history["best"] = {
            "epoch": best_epoch,
            "train_loss": best_train_loss,
            "val_loss": best_val_loss,
            "checkpoint": best_path
    }


        # save history
        # --------------------------------------------
    # SAVE HISTORY PER EMBEDDING TYPE (DO NOT OVERWRITE)
    # --------------------------------------------
    history_path = osp.join(args.out_dir_final, "cnn_history.json")

    # 1. Load existing history file
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            all_histories = json.load(f)
    else:
        all_histories = {}

    # 2. Store this run’s history under its embedding type
    all_histories[args.embedding_type] = history

    # 3. Save everything back
    with open(history_path, "w") as f:
        json.dump(all_histories, f, indent=2)


    print("Training finished. Best val loss:", best_val)
    print("Final best saved at:", osp.join(args.out_dir, f"m1_best_{args.embedding_type}.pt"))

if __name__ == "__main__":
    main()
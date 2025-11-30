#!/usr/bin/env python3
"""
train_model1_cnn_lstm_emotion.py

Model 1: Custom CNN (from scratch) + LSTM captioner with emotion conditioning.

- CNN encodes image -> compact vector (img_feat_dim)
- Emotion embedding is learned (num_emotions x emo_dim)
- Decoder LSTM input at each time-step = concat(word_emb, img_feat, emo_emb)
- Loss: CrossEntropyLoss(ignore_index=pad_idx)
- Checkpoints:
    checkpoints/model1/m1_epoch{epoch}.pt
    checkpoints/model1/m1_best.pt
- History: checkpoints/summary/m1_history.json

Adjust CONFIG as needed.
"""

import os
import time
import json
import tempfile
import datetime
from pathlib import Path
from typing import Optional

import math
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------
# CONFIG - edit as needed
# -------------------------
CONFIG = {
    # data
    "train_csv": "data_preprocessed/train.csv",
    "val_csv": "data_preprocessed/val.csv",
    "features_dir": "data_preprocessed/features",  # expects painting.npy with HxWxC normalized [0,1]
    "vocab_path": "data_preprocessed/vocab.pkl",   # token->idx dict
    "pretrained_token_emb_weights": None,          # optional: .npy file (V, E_token)

    # model architecture
    "img_feat_dim": 256,       # final image feature dimension (output of CNN)
    "emo_emb_dim": 64,         # emotion embedding dimension
    "word_emb_dim": 300,       # token embedding dim (can be changed)
    "lstm_hidden": 512,        # LSTM hidden size
    "lstm_layers": 1,
    "dropout": 0.3,
    "bidirectional": False,

    "vocab_size": 8000,        # fallback; will be overwritten if vocab_path loads
    "max_seq_len": 25,         # length of token sequences (including <start>, <end>)
    "num_emotions": 9,

    # training
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 32,
    "num_epochs": 3,
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,
    "grad_clip": 2.0,

    # checkpoints
    "checkpoint_root": "checkpoints",
    "model1_subdir": "m1_pt",
    "summary_subdir": "summary",
}

# -------------------------
# utils
# -------------------------
def atomic_write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        with open(path, "w", encoding="utf8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)

def parse_token_field(tok_field, max_len):
    """Parse token field (string like '[1,2,3]' or whitespace separated ints or list)."""
    if isinstance(tok_field, str):
        tok_field = tok_field.strip()
        if tok_field.startswith("[") and tok_field.endswith("]"):
            try:
                lst = eval(tok_field)
                return [int(x) for x in lst][:max_len]
            except Exception:
                parts = tok_field.strip("[]").split(",")
                return [int(p) for p in parts if p.strip().isdigit()][:max_len]
        else:
            parts = tok_field.split()
            return [int(p) for p in parts if p.isdigit()][:max_len]
    elif isinstance(tok_field, (list, tuple, np.ndarray)):
        return [int(x) for x in tok_field][:max_len]
    elif tok_field != tok_field:  # NaN
        return []
    else:
        try:
            return [int(tok_field)]
        except Exception:
            return []

# -------------------------
# Dataset
# -------------------------
class CaptionDatasetCNN(Dataset):
    def __init__(self, csv_path, features_dir, token_col_candidates=None, max_len=25):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.features_dir = Path(features_dir)
        self.max_len = max_len

        if 'painting' not in self.df.columns:
            raise RuntimeError("CSV must contain 'painting' column")
        if 'emotion_label' not in self.df.columns:
            raise RuntimeError("CSV must contain 'emotion_label' column")

        candidates = token_col_candidates or ["token_ids_c1", "token_ids", "tokens"]
        chosen = None
        for c in candidates:
            if c in self.df.columns:
                chosen = c
                break
        if chosen is None:
            raise RuntimeError(f"No token column found. Checked {candidates}")
        self.token_col = chosen

    def __len__(self):
        return len(self.df)

    def _load_image(self, painting):
        p = self.features_dir / f"{painting}.npy"
        if not p.exists():
            # fallback zero image
            return torch.zeros(3, CONFIG["img_size"] if "img_size" in CONFIG else 224, CONFIG.get("img_size", 224))
        arr = np.load(p)
        if arr.ndim == 3:
            return torch.tensor(arr).permute(2, 0, 1).float()  # (C,H,W)
        else:
            return torch.tensor(arr).float()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        painting = row['painting']
        img = self._load_image(painting)
        tok_field = row.get(self.token_col, row.get('token_ids', None))
        token_ids = parse_token_field(tok_field, self.max_len)
        # pad/truncate
        if len(token_ids) < self.max_len:
            token_ids = token_ids + [0] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        emo = int(row['emotion_label'])
        return img, token_ids, torch.tensor(emo, dtype=torch.long)

# -------------------------
# CNN encoder (custom small CNN)
# -------------------------
class SmallCNNEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        # Input expected (B,3,224,224)
        self.net = nn.Sequential(
            # block1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 112
            # block2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 56
            # block3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 28
            # block4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # global pooling -> (B,512,1,1)
        )
        self.fc = nn.Linear(512, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        # x: (B,3,H,W)
        h = self.net(x)               # (B,512,1,1)
        h = h.view(h.size(0), -1)     # (B,512)
        out = self.fc(h)              # (B,out_dim)
        return out

# -------------------------
# Decoder LSTM with emotion & image feature concatenated to token embedding
# -------------------------
class LSTMDecoderWithEmotion(nn.Module):
    def __init__(self, vocab_size, word_emb_dim, img_feat_dim, emo_emb_dim,
                 lstm_hidden, lstm_layers=1, dropout=0.3, pretrained_token_emb=None, pad_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.word_emb_dim = word_emb_dim
        self.img_feat_dim = img_feat_dim
        self.emo_emb_dim = emo_emb_dim
        self.input_dim = word_emb_dim + img_feat_dim + emo_emb_dim
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.pad_idx = pad_idx

        # token embedding
        if pretrained_token_emb is not None:
            W = torch.tensor(pretrained_token_emb, dtype=torch.float32)
            if W.shape[0] == vocab_size and W.shape[1] == word_emb_dim:
                self.token_emb = nn.Embedding.from_pretrained(W, freeze=False, padding_idx=pad_idx)
            else:
                # fallback to learnable embedding and try to copy overlapping dims
                self.token_emb = nn.Embedding(vocab_size, word_emb_dim, padding_idx=pad_idx)
                try:
                    k = min(W.shape[0], vocab_size)
                    j = min(W.shape[1], word_emb_dim)
                    self.token_emb.weight.data[:k, :j] = W[:k, :j]
                except Exception:
                    pass
        else:
            self.token_emb = nn.Embedding(vocab_size, word_emb_dim, padding_idx=pad_idx)

        # emotion embedding
        self.emo_emb = nn.Embedding(CONFIG["num_emotions"], emo_emb_dim)

        # LSTM
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0,
                            bidirectional=False)
        # projection to vocab
        self.fc_out = nn.Linear(lstm_hidden, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids, img_feats, emo_ids, teacher_forcing=True):
        """
        token_ids: (B, T) token ids (with start token at pos0)
        img_feats: (B, img_feat_dim) repeated/concatenated along time
        emo_ids: (B,)
        Returns logits: (B, T, V)
        """
        B, T = token_ids.shape
        # embed tokens
        tok_emb = self.token_emb(token_ids)   # (B, T, E_word)
        emo_vec = self.emo_emb(emo_ids)       # (B, emo_emb_dim)
        # expand image & emotion across time
        img_expand = img_feats.unsqueeze(1).expand(-1, T, -1)   # (B,T,img_feat_dim)
        emo_expand = emo_vec.unsqueeze(1).expand(-1, T, -1)     # (B,T,emo_emb_dim)

        # concat inputs
        lstm_inputs = torch.cat([tok_emb, img_expand, emo_expand], dim=-1)  # (B,T,input_dim)
        lstm_inputs = self.dropout(lstm_inputs)

        # run LSTM
        outputs, _ = self.lstm(lstm_inputs)  # outputs: (B,T,hidden)
        outputs = self.dropout(outputs)
        logits = self.fc_out(outputs)        # (B,T,V)
        return logits

    def greedy_decode(self, img_feat, emo_id, sos_idx, eos_idx=None, max_len=25, device='cpu'):
        self.eval()
        with torch.no_grad():
            # start token
            cur = torch.LongTensor([[sos_idx]]).to(device)   # (1,1)
            generated = []
            B = 1
            # prepare expanded img, emo for concatenation
            img_feat = img_feat.to(device).unsqueeze(0)  # (1, dim)
            emo_vec = self.emo_emb(torch.LongTensor([emo_id]).to(device))  # (1, emo_dim)
            for step in range(max_len):
                tok_emb = self.token_emb(cur)  # (1, t, E_word)
                # take only last token embedding
                last_tok_emb = tok_emb[:, -1:, :]  # (1,1,E)
                img_expand = img_feat.unsqueeze(1)  # (1,1,img_dim)
                emo_expand = emo_vec.unsqueeze(1)   # (1,1,emo_dim)
                inp = torch.cat([last_tok_emb, img_expand, emo_expand], dim=-1)  # (1,1,input_dim)
                out, hidden = self.lstm(inp)  # (1,1,hidden)
                logits = self.fc_out(out[:, -1, :])  # (1, V)
                next_id = torch.argmax(logits, dim=-1).item()
                generated.append(next_id)
                if eos_idx is not None and next_id == eos_idx:
                    break
                cur = torch.cat([cur, torch.LongTensor([[next_id]]).to(device)], dim=1)
        return generated

# -------------------------
# Helpers: load vocab
# -------------------------
def load_vocab(vocab_path):
    if not Path(vocab_path).exists():
        return None
    with open(vocab_path, "rb") as f:
        tok2idx = pickle.load(f)
    if not isinstance(tok2idx, dict) and hasattr(tok2idx, "token_to_idx"):
        tok2idx = tok2idx.token_to_idx
    return tok2idx

# -------------------------
# Train / validation functions
# -------------------------
def train_one_epoch(model_cnn, model_dec, loader, optimizer, device, pad_idx, grad_clip=None):
    model_cnn.train()
    model_dec.train()
    total_loss = 0.0
    n = 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    for imgs, token_ids, emos in loader:
        imgs = imgs.to(device)
        token_ids = token_ids.to(device)
        emos = emos.to(device)

        optimizer.zero_grad()
        img_feats = model_cnn(imgs)  # (B, img_feat_dim)
        logits = model_dec(token_ids, img_feats, emos, teacher_forcing=True)  # (B,T,V)
        preds = logits[:, :-1, :].contiguous()
        targets = token_ids[:, 1:].contiguous()

        B, Tm, V = preds.shape
        loss = criterion(preds.view(-1, V), targets.view(-1))
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(list(model_cnn.parameters()) + list(model_dec.parameters()), grad_clip)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    avg_loss = total_loss / max(1, n)
    return avg_loss

def validate_epoch(model_cnn, model_dec, loader, device, pad_idx):
    model_cnn.eval()
    model_dec.eval()
    total_loss = 0.0
    n = 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    with torch.no_grad():
        for imgs, token_ids, emos in loader:
            imgs = imgs.to(device)
            token_ids = token_ids.to(device)
            emos = emos.to(device)
            img_feats = model_cnn(imgs)
            logits = model_dec(token_ids, img_feats, emos, teacher_forcing=True)
            preds = logits[:, :-1, :].contiguous()
            targets = token_ids[:, 1:].contiguous()
            B, Tm, V = preds.shape
            loss = criterion(preds.view(-1, V), targets.view(-1))
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
    avg_loss = total_loss / max(1, n)
    return avg_loss

# -------------------------
# Main: wiring + training loop
# -------------------------
def main(cfg):
    device = torch.device(cfg["device"])
    print("Device:", device)

    # checkpoint dirs
    root = Path(cfg["checkpoint_root"])
    model1_dir = root / cfg["model1_subdir"]
    summary_dir = root / cfg["summary_subdir"]
    model1_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    history_path = summary_dir / "m1new_history.json"

    # load or init history
    if history_path.exists():
        try:
            history = json.load(open(history_path, "r"))
        except Exception:
            history = {"epochs": []}
    else:
        history = {"epochs": []}

    best_val = float("inf")
    for e in history.get("epochs", []):
        if "val_loss" in e and e["val_loss"] is not None:
            best_val = min(best_val, e["val_loss"])

    # load vocab
    tok2idx = load_vocab(cfg["vocab_path"])
    if tok2idx is not None:
        vocab_size = len(tok2idx)
        idx2tok = {i: t for t, i in tok2idx.items()}
        cfg["vocab_size"] = vocab_size
        print("Loaded vocab size:", vocab_size)
    else:
        vocab_size = cfg["vocab_size"]
        idx2tok = None
        print("Using fallback vocab_size:", vocab_size)

    pad_idx = 0
    sos_idx = 1 if 1 in range(vocab_size) else 1
    eos_idx = None
    if idx2tok:
        # attempt to find end token id
        for i, tok in idx2tok.items():
            if tok in ("<end>", "</s>", "</end>"):
                eos_idx = i
                break

    # load pretrained token embeddings if provided
    pretrained_emb = None
    if cfg.get("pretrained_token_emb_weights") and Path(cfg["pretrained_token_emb_weights"]).exists():
        pretrained_emb = np.load(cfg["pretrained_token_emb_weights"])
        print("Loaded pretrained token embedding weights:", pretrained_emb.shape)

    # datasets + loaders
    train_ds = CaptionDatasetCNN(cfg["train_csv"], cfg["features_dir"], max_len=cfg["max_seq_len"])
    val_loader = None
    if Path(cfg["val_csv"]).exists():
        val_ds = CaptionDatasetCNN(cfg["val_csv"], cfg["features_dir"], max_len=cfg["max_seq_len"])
        val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    # models
    cnn = SmallCNNEncoder(out_dim=cfg["img_feat_dim"]).to(device)
    decoder = LSTMDecoderWithEmotion(
        vocab_size=cfg["vocab_size"],
        word_emb_dim=cfg["word_emb_dim"],
        img_feat_dim=cfg["img_feat_dim"],
        emo_emb_dim=cfg["emo_emb_dim"],
        lstm_hidden=cfg["lstm_hidden"],
        lstm_layers=cfg["lstm_layers"],
        dropout=cfg["dropout"],
        pretrained_token_emb=pretrained_emb,
        pad_idx=pad_idx
    ).to(device)

    # optimizer (joint)
    optimizer = torch.optim.AdamW(list(cnn.parameters()) + list(decoder.parameters()), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    # training loop
    for epoch in range(1, cfg["num_epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(cnn, decoder, train_loader, optimizer, device, pad_idx, grad_clip=cfg["grad_clip"])
        t1 = time.time()
        print(f"Epoch {epoch}/{cfg['num_epochs']}  train_loss={train_loss:.4f}  time={t1-t0:.1f}s")

        val_loss = None
        if val_loader is not None:
            val_loss = validate_epoch(cnn, decoder, val_loader, device, pad_idx)
            print(f"            val_loss={val_loss:.4f}")

        # save epoch checkpoint
        ckpt = {
            "epoch": epoch,
            "cnn_state": cnn.state_dict(),
            "decoder_state": decoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": cfg
        }
        epoch_path = model1_dir / f"m1new_epoch{epoch}.pt"
        torch.save(ckpt, epoch_path)
        print(f"Saved checkpoint -> {epoch_path}")

        # update history
        hist_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "timestamp": datetime.datetime.now().isoformat()
        }
        history.setdefault("epochs", []).append(hist_entry)
        history["last_updated"] = datetime.datetime.now().isoformat()
        atomic_write_json(history, str(history_path))
        print("Updated history ->", history_path)

        # save best
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            best_path = model1_dir / "m1new_best.pt"
            torch.save(ckpt, best_path)
            print("Saved NEW BEST ->", best_path)

    print("Training finished. History saved to:", history_path)

if __name__ == "__main__":
    # minor default housekeeping: ensure features dir exists
    if not Path(CONFIG["features_dir"]).exists():
        print(f"Warning: features dir {CONFIG['features_dir']} not found. Update CONFIG if needed.")
    main(CONFIG)
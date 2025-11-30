#!/usr/bin/env python3
"""
train_model2_vlt_emotion.py

Vision-Language Transformer training script (emotion-conditioned).
- Minimal validation (only val loss)
- Saves epoch checkpoints and best checkpoint
- Writes history to checkpoints/summary/vlt_history.json
"""

import os
import time
import math
import json
import datetime
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pickle

# -------------------------
# CONFIG - edit as needed
# -------------------------
CONFIG = {
    # data
    "train_csv": "data_preprocessed/train.csv",
    "val_csv": "data_preprocessed/val.csv",
    "images_features_root": "data_preprocessed/features",  # .npy per painting, RGB HxWxC normalized to [0,1]
    "vocab_path": "data_preprocessed/vocab.pkl",           # token->idx dict (or object with token_to_idx)
    "pretrained_token_emb_weights": None,  # optional .npy path (V, D_token)

    # model / training
    "image_size": 224,
    "patch_size": 32,
    "vit_embed_dim": 256,
    "vit_depth": 2,
    "vit_num_heads": 4,
    "decoder_embed_dim": 256,
    "decoder_depth": 2,
    "decoder_num_heads": 4,
    "vocab_size": 8000,
    "max_seq_len": 20,   # includes <start> and <end>
    "dropout": 0.1,
    "num_emotions": 9,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 16,
    "num_epochs": 3,
    "learning_rate": 1e-4,

    # checkpointing / history
    "checkpoint_root": "checkpoints",
    "model2_subdir": "m2_pt",
    "summary_subdir": "summary",
}

# -------------------------
# Helpers
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

def sinusoidal_positional_encoding(n_pos: int, d_model: int):
    pe = torch.zeros(n_pos, d_model)
    position = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# -------------------------
# Patch embed + Encoder (Kaggle-style)
# -------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=32, in_chans=3, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                     # (B, E, H', W')
        B, E, Hn, Wn = x.shape
        return x.flatten(2).transpose(1, 2)  # (B, N, E)

class VisionTransformerEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=32, embed_dim=256, depth=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        P = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        pos = sinusoidal_positional_encoding(P + 1, embed_dim)
        self.pos_embed = nn.Parameter(pos, requires_grad=False)
        layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4, dropout=dropout)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)         # (B, N+1, D)
        pos = self.pos_embed.unsqueeze(0).to(x.device)
        if pos.size(1) != x.size(1):
            pos = pos[:, :x.size(1), :]
        x = x + pos
        x = x.transpose(0, 1)                  # (S, B, D)
        x = self.encoder(x)
        x = self.norm(x.transpose(0, 1))       # (B, S, D)
        return x

# -------------------------
# Decoder building blocks
# -------------------------
class DecoderLayerCustom(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))
        tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = self.norm2(tgt + self.dropout(tgt2))
        ff = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout(ff))
        return tgt

class TransformerDecoderCustom(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer] + [type(layer)(layer.self_attn.embed_dim, layer.self_attn.num_heads, layer.linear1.out_features, layer.dropout.p) for _ in range(num_layers - 1)])

    def forward(self, tgt, memory, tgt_mask=None):
        out = tgt
        for layer in self.layers:
            out = layer(out, memory, tgt_mask=tgt_mask)
        return out

# -------------------------
# Full Model (emotion prepended)
# -------------------------
class VisionLanguageTransformer(nn.Module):
    def __init__(self, cfg, token_embedding_weights=None):
        super().__init__()
        self.cfg = cfg
        D_enc = cfg["vit_embed_dim"]
        D_dec = cfg["decoder_embed_dim"]

        self.encoder = VisionTransformerEncoder(img_size=cfg["image_size"], patch_size=cfg["patch_size"],
                                                embed_dim=D_enc, depth=cfg["vit_depth"], num_heads=cfg["vit_num_heads"], dropout=cfg["dropout"])
        # token embedding (decoder side)
        self.token_embed = nn.Embedding(cfg["vocab_size"], D_dec)
        if token_embedding_weights is not None:
            w = torch.tensor(token_embedding_weights, dtype=torch.float32)
            if w.shape == (cfg["vocab_size"], w.shape[1]):
                # if dims mismatch, we still copy up to min dims
                try:
                    if w.shape[1] == D_dec:
                        self.token_embed.weight.data.copy_(w)
                except Exception:
                    pass

        # pos enc for tokens (we allocate max_seq_len + 1 because we'll prepend emotion slot)
        pos_tokens = sinusoidal_positional_encoding(cfg["max_seq_len"] + 1, D_dec)
        self.pos_embed_tokens = nn.Parameter(pos_tokens, requires_grad=False)

        # projector from encoder to decoder dim if necessary
        if D_enc != D_dec:
            self.enc_to_dec = nn.Linear(D_enc, D_dec)
        else:
            self.enc_to_dec = nn.Identity()

        # emotion embedding
        self.emotion_emb = nn.Embedding(cfg["num_emotions"], D_dec)
        nn.init.xavier_uniform_(self.emotion_emb.weight)

        layer = DecoderLayerCustom(D_dec, cfg["decoder_num_heads"], dim_feedforward=D_dec * 4, dropout=cfg["dropout"])
        self.decoder = TransformerDecoderCustom(layer, cfg["decoder_depth"])
        self.output_proj = nn.Linear(D_dec, cfg["vocab_size"])
        self.pad_idx = 0

    def forward(self, images, token_ids, emo_ids):
        # images: (B,3,H,W) normalized
        # token_ids: (B, T)
        # emo_ids: (B,)
        device = images.device
        B, T = token_ids.shape

        enc = self.encoder(images)                 # (B, S, D_enc)
        enc = self.enc_to_dec(enc)                 # (B, S, D_dec)
        memory = enc.transpose(0, 1)               # (S, B, D_dec)

        tok_emb = self.token_embed(token_ids)      # (B, T, D_dec)
        emo_vec = self.emotion_emb(emo_ids).unsqueeze(1)  # (B,1,D_dec)

        dec_in = torch.cat([emo_vec, tok_emb], dim=1)    # (B, T+1, D_dec)
        # add positional encodings for T+1
        pos = self.pos_embed_tokens[: (T + 1), :].unsqueeze(0).to(device)
        dec_in = dec_in + pos
        dec_in = dec_in.transpose(0, 1)            # (T+1, B, D_dec)

        tgt_mask = torch.triu(torch.ones((T + 1, T + 1), device=device), diagonal=1).bool()
        dec_out = self.decoder(dec_in, memory, tgt_mask=tgt_mask)  # (T+1, B, D)
        dec_out = dec_out.transpose(0, 1)          # (B, T+1, D)
        logits = self.output_proj(dec_out)         # (B, T+1, V)
        return logits[:, 1:, :]                    # (B, T, V) aligned to tokens (skip emo pos)

    def greedy_decode(self, image, emo_id, sos_idx, eos_idx=None, max_len=None, device='cpu'):
        self.eval()
        if max_len is None:
            max_len = self.cfg["max_seq_len"]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            enc = self.encoder(image)
            enc = self.enc_to_dec(enc)
            memory = enc.transpose(0, 1)
            emo_vec = self.emotion_emb(torch.LongTensor([emo_id]).to(device))
            cur = torch.LongTensor([[sos_idx]]).to(device)
            generated = []
            for step in range(max_len):
                # pad cur to max_len
                cur_padded = F.pad(cur, (0, max_len - cur.size(1)), value=0)
                logits = self.forward(image, cur_padded, torch.LongTensor([emo_id]).to(device))  # (1,max_len,V)
                step_idx = cur.size(1) - 1
                logit_step = logits[:, step_idx, :]
                next_id = torch.argmax(logit_step, dim=-1).item()
                generated.append(next_id)
                if eos_idx is not None and next_id == eos_idx:
                    break
                cur = torch.cat([cur, torch.LongTensor([[next_id]]).to(device)], dim=1)
        return generated

# -------------------------
# Dataset
# -------------------------
class CaptionDataset(Dataset):
    def __init__(self, csv_path, images_root, token_col_candidates=None, max_len=20):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.images_root = Path(images_root)
        self.max_len = max_len

        # ensure required columns exist
        if 'painting' not in self.df.columns:
            raise RuntimeError("CSV must contain 'painting' column")
        if 'emotion_label' not in self.df.columns:
            raise RuntimeError("CSV must contain 'emotion_label' column")

        # choose token column
        candidates = token_col_candidates or ["token_ids_c1", "token_ids", "tokens"]
        chosen = None
        for c in candidates:
            if c in self.df.columns:
                chosen = c
                break
        if chosen is None:
            raise RuntimeError(f"No token column found in CSV. Checked: {candidates}")
        self.token_col = chosen

    def __len__(self):
        return len(self.df)

    def _load_image(self, painting_name):
        p = self.images_root / f"{painting_name}.npy"
        if not p.exists():
            # fallback zeros
            return torch.zeros(3, CONFIG["image_size"], CONFIG["image_size"]).float()
        arr = np.load(p)
        if arr.ndim == 3:
            return torch.tensor(arr).permute(2, 0, 1).float()
        else:
            return torch.tensor(arr).float()

    def _parse_token_field(self, tok_field):
        if isinstance(tok_field, str):
            tok_field = tok_field.strip()
            if tok_field.startswith("[") and tok_field.endswith("]"):
                try:
                    lst = eval(tok_field)
                    return [int(x) for x in lst]
                except Exception:
                    parts = tok_field.strip("[]").split(",")
                    return [int(p) for p in parts if p.strip().isdigit()]
            else:
                parts = tok_field.split()
                return [int(p) for p in parts if p.isdigit()]
        elif isinstance(tok_field, (list, tuple, np.ndarray)):
            return [int(x) for x in tok_field]
        elif tok_field != tok_field:  # NaN
            return [0] * self.max_len
        else:
            return [int(tok_field)]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        painting = row['painting']
        img = self._load_image(painting)
        tok_field = row.get(self.token_col, row.get('token_ids', None))
        token_ids = self._parse_token_field(tok_field)
        token_ids = token_ids[:self.max_len]
        token_ids += [0] * (self.max_len - len(token_ids))
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        emo = int(row['emotion_label'])
        return img, token_ids, torch.tensor(emo, dtype=torch.long)

# -------------------------
# Training / Validation helpers
# -------------------------
def train_one_epoch(model, dataloader, optimizer, device, criterion):
    model.train()
    total_loss = 0.0
    n = 0
    for imgs, token_ids, emos in dataloader:
        imgs = imgs.to(device)
        token_ids = token_ids.to(device)
        emos = emos.to(device)
        optimizer.zero_grad()
        logits = model(imgs, token_ids, emos)  # (B, T, V)
        logits_in = logits[:, :-1]              # (B, T-1, V)
        targets = token_ids[:, 1:].contiguous() # (B, T-1)
        B, Tm, V = logits_in.shape
        loss = criterion(logits_in.reshape(B * Tm, V), targets.reshape(B * Tm))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / max(1, n)

def validate_epoch(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for imgs, token_ids, emos in dataloader:
            imgs = imgs.to(device)
            token_ids = token_ids.to(device)
            emos = emos.to(device)
            logits = model(imgs, token_ids, emos)
            logits_in = logits[:, :-1]
            targets = token_ids[:, 1:].contiguous()
            B, Tm, V = logits_in.shape
            loss = criterion(logits_in.reshape(B * Tm, V), targets.reshape(B * Tm))
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
    return total_loss / max(1, n)

# -------------------------
# Main
# -------------------------
def main(cfg):
    device = torch.device(cfg["device"])
    print("Using device:", device)

    # prepare checkpoint dirs
    root = Path(cfg["checkpoint_root"])
    model2_dir = root / cfg["model2_subdir"]
    summary_dir = root / cfg["summary_subdir"]
    model2_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    history_path = summary_dir / "vlt_history.json"

    # load or init history
    if history_path.exists():
        try:
            history = json.load(open(history_path, "r"))
        except Exception:
            history = {"epochs": []}
    else:
        history = {"epochs": []}

    # set best_val from history if available
    best_val = float("inf")
    for e in history.get("epochs", []):
        if "val_loss" in e and e["val_loss"] is not None:
            best_val = min(best_val, e["val_loss"])

    # try load vocab to get vocab_size if possible
    if Path(cfg["vocab_path"]).exists():
        try:
            tok2idx = pickle.load(open(cfg["vocab_path"], "rb"))
            if not isinstance(tok2idx, dict) and hasattr(tok2idx, "token_to_idx"):
                tok2idx = tok2idx.token_to_idx
            cfg["vocab_size"] = len(tok2idx)
            idx2tok = {i: t for t, i in tok2idx.items()}
            print("Loaded vocab size:", cfg["vocab_size"])
        except Exception:
            print("Could not load vocab.pkl; using config vocab_size")
            tok2idx = None
            idx2tok = None
    else:
        tok2idx = None
        idx2tok = None

    # try load pretrained token embedding weights (optional)
    token_emb_weights = None
    if cfg.get("pretrained_token_emb_weights") and Path(cfg["pretrained_token_emb_weights"]).exists():
        token_emb_weights = np.load(cfg["pretrained_token_emb_weights"])

    # datasets + loaders
    train_ds = CaptionDataset(cfg["train_csv"], Path(cfg["images_features_root"]), max_len=cfg["max_seq_len"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    val_loader = None
    if cfg.get("val_csv") and Path(cfg["val_csv"]).exists():
        val_ds = CaptionDataset(cfg["val_csv"], Path(cfg["images_features_root"]), max_len=cfg["max_seq_len"])
        val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    # build model
    model = VisionLanguageTransformer(cfg, token_embedding_weights=token_emb_weights)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # training loop
    for epoch in range(1, cfg["num_epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        t1 = time.time()
        print(f"Epoch {epoch}/{cfg['num_epochs']}: train_loss={train_loss:.4f}  time={t1-t0:.1f}s")

        val_loss = None
        if val_loader is not None:
            val_loss = validate_epoch(model, val_loader, device, criterion)
            print(f"               val_loss={val_loss:.4f}")

        # save epoch checkpoint
        epoch_ckpt = model2_dir / f"m2_epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": cfg
        }, epoch_ckpt)
        print(f"Saved checkpoint: {epoch_ckpt}")

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
        print(f"Updated history -> {history_path}")

        # save best model by val_loss if available
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            best_path = model2_dir / "m2_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "config": cfg
            }, best_path)
            print(f"New BEST saved -> {best_path}")

    # finalize history
    history["finished_at"] = datetime.datetime.now().isoformat()
    atomic_write_json(history, str(history_path))
    print("Training finished. History saved to:", history_path)

if __name__ == "__main__":
    main(CONFIG)
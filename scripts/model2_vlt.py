#!/usr/bin/env python3
"""
train_model2_vlt_emotion.py

Vision-Language Transformer training script (emotion-conditioned).

Supports multiple embedding strategies:
  - random (trainable nn.Embedding)
  - glove / fasttext (load .npy matrix aligned to vocab)
  - tfidf (expects per-row reduced TF-IDF .npy files; will prepend a projected TF-IDF vector token)

Checkpointing:
  - Epochs -> checkpoints/model2/m2_epoch{E}.pt
  - Best per-embedding -> checkpoints/model2/m2_best_{embedding}.pt
History -> checkpoints/model2/vlt_history.json
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

import argparse
import pickle

# -------------------------
# DEFAULT CONFIG - can be overridden by CLI args
# -------------------------
DEFAULT_CONFIG = {
    # data
    "train_csv": "data_preprocessed/train.csv",
    "val_csv": "data_preprocessed/val.csv",
    "images_features_root": "data_preprocessed/features",  # .npy per painting, RGB HxWxC normalized to [0,1]
    "vocab_path": "data_preprocessed/vocab.pkl",           # token->idx dict (or object with token_to_idx)

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
    "num_epochs": 4,
    "learning_rate": 1e-4,

    # checkpointing / history
    "checkpoint_root": "checkpoints",
    "model2_subdir": "model2",   # will create checkpoints/model2
}

# -------------------------
# Small utilities
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
# Patch embed + Encoder
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
# Full Model (emotion prepended + embedding options)
# -------------------------
class VisionLanguageTransformer(nn.Module):
    def __init__(self, cfg, pretrained_token_emb_weights: Optional[np.ndarray]=None, embedding_type: str="random", tfidf_dim: Optional[int]=None, freeze_emb: bool=False):
        super().__init__()
        self.cfg = cfg
        D_enc = cfg["vit_embed_dim"]
        D_dec = cfg["decoder_embed_dim"]

        # encoder
        self.encoder = VisionTransformerEncoder(img_size=cfg["image_size"], patch_size=cfg["patch_size"],
                                                embed_dim=D_enc, depth=cfg["vit_depth"], num_heads=cfg["vit_num_heads"], dropout=cfg["dropout"])

        # Token embedding (decoder side)
        self.embedding_type = embedding_type
        self.token_embed = nn.Embedding(cfg["vocab_size"], D_dec)

        self.pretrained_proj = None
        if embedding_type in ("glove", "fasttext") and pretrained_token_emb_weights is not None:
            w = torch.tensor(pretrained_token_emb_weights, dtype=torch.float32)
            pre_dim = w.shape[1]
            if pre_dim == D_dec and w.shape[0] == cfg["vocab_size"]:
                self.token_embed.weight.data.copy_(w)
            else:
                # create a frozen lookup and a learned projection
                self.pretrained_lookup = nn.Embedding(w.shape[0], pre_dim)
                self.pretrained_lookup.weight.data.copy_(w)
                self.pretrained_lookup.weight.requires_grad = False
                self.pretrained_proj = nn.Linear(pre_dim, D_dec)
                # initialize token_embed small
                nn.init.xavier_uniform_(self.token_embed.weight)
        else:
            # random init
            nn.init.xavier_uniform_(self.token_embed.weight)

        # Optionally freeze the primary token_embed (if using direct copy)
        self.freeze_emb = freeze_emb
        if self.freeze_emb:
            self.token_embed.weight.requires_grad = False

        # Positional tokens: +1 for emotion-prepend slot
        pos_tokens = sinusoidal_positional_encoding(cfg["max_seq_len"] + 1, D_dec)
        self.pos_embed_tokens = nn.Parameter(pos_tokens, requires_grad=False)

        # projector from encoder to decoder dim if necessary
        if D_enc != D_dec:
            self.enc_to_dec = nn.Linear(D_enc, D_dec)
        else:
            self.enc_to_dec = nn.Identity()

        # emotion embedding (prepended)
        self.emotion_emb = nn.Embedding(cfg["num_emotions"], D_dec)
        nn.init.xavier_uniform_(self.emotion_emb.weight)

        # if TF-IDF mode: a linear to project TF-IDF vector into D_dec and we'll prepend it (instead of emotion token)
        self.tfidf_proj = None
        if embedding_type == "tfidf":
            if tfidf_dim is None:
                raise RuntimeError("TF-IDF embedding selected but tfidf_dim is None. Provide tfidf_dim.")
            self.tfidf_proj = nn.Linear(tfidf_dim, D_dec)

        layer = DecoderLayerCustom(D_dec, cfg["decoder_num_heads"], dim_feedforward=D_dec * 4, dropout=cfg["dropout"])
        self.decoder = TransformerDecoderCustom(layer, cfg["decoder_depth"])
        self.output_proj = nn.Linear(D_dec, cfg["vocab_size"])
        self.pad_idx = 0

    def forward(self, images, token_ids, emo_ids=None, tfidf_vecs=None):
        # images: (B,3,H,W) normalized
        # token_ids: (B, T)
        # emo_ids: (B,)  (int labels)  -- used when emotion-prepend mode
        # tfidf_vecs: (B, k)  -- used if embedding_type == 'tfidf'
        device = images.device
        B, T = token_ids.shape

        enc = self.encoder(images)                 # (B, S, D_enc)
        enc = self.enc_to_dec(enc)                 # (B, S, D_dec)
        memory = enc.transpose(0, 1)               # (S, B, D_dec)

        # token embedding (consider pretrained lookup + proj path)
        if hasattr(self, "pretrained_lookup") and self.pretrained_lookup is not None:
            pre = self.pretrained_lookup(token_ids)     # (B,T,pre_dim)
            tok_emb = self.pretrained_proj(pre)        # (B,T,D_dec)
        else:
            tok_emb = self.token_embed(token_ids)      # (B,T,D_dec)

        # handle emotion or tfidf prepending
        if self.embedding_type == "tfidf":
            if tfidf_vecs is None:
                raise RuntimeError("tfidf_vecs must be provided when embedding_type=='tfidf'")
            tfidf_token = self.tfidf_proj(tfidf_vecs).unsqueeze(1)  # (B,1,D)
            dec_in = torch.cat([tfidf_token, tok_emb], dim=1)      # (B, T+1, D)
        else:
            # emotion prepend (default)
            if emo_ids is None:
                raise RuntimeError("emo_ids must be provided for emotion-prepend mode")
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
        return logits[:, 1:, :]                    # (B, T, V) aligned to tokens (skip prepend slot)

    def greedy_decode(self, image, emo_id, sos_idx, eos_idx=None, max_len=None, device='cpu', tfidf_vec=None):
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
            cur = torch.LongTensor([[sos_idx]]).to(device)
            generated = []
            for step in range(max_len):
                # pad cur to max_len
                cur_padded = F.pad(cur, (0, max_len - cur.size(1)), value=0)
                # forward
                if self.embedding_type == "tfidf":
                    if tfidf_vec is None:
                        tfidf_vec = torch.zeros(1, self.tfidf_proj.in_features).to(device)
                    logits = self.forward(image, cur_padded, emo_ids=torch.tensor([0]).to(device), tfidf_vecs=tfidf_vec)
                else:
                    logits = self.forward(image, cur_padded, emo_ids=torch.LongTensor([emo_id]).to(device))
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
    def __init__(self, csv_path, images_root, max_len=20, token_col_candidates=None, embedding_type="random", tfidf_dir: Optional[str]=None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.images_root = Path(images_root)
        self.max_len = max_len
        self.embedding_type = embedding_type
        self.tfidf_dir = Path(tfidf_dir) if tfidf_dir else None

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
            raise RuntimeError(f"No token column found in CSV. Checked: {candidates}")
        self.token_col = chosen

    def __len__(self):
        return len(self.df)

    def _load_image(self, painting_name):
        p_npy = self.images_root / f"{painting_name}.npy"
        if not p_npy.exists():
            return torch.zeros(3, DEFAULT_CONFIG["image_size"], DEFAULT_CONFIG["image_size"]).float()
        arr = np.load(p_npy)
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

        # TF-IDF vector if requested (expects files named "{painting}.npy" or indexed .npy)
        if self.embedding_type == "tfidf":
            if self.tfidf_dir is None:
                raise RuntimeError("TF-IDF dir not provided for TF-IDF embedding mode.")
            # try painting-named file first
            by_paint = self.tfidf_dir / f"{painting}.npy"
            if by_paint.exists():
                tfidf_vec = np.load(by_paint)
            else:
                # fallback: try index-based file (best-effort)
                try:
                    candidate = list(self.tfidf_dir.glob("*.npy"))[idx]
                    tfidf_vec = np.load(candidate)
                except Exception:
                    tfidf_vec = np.zeros((self.tfidf_dir and 1 or 1,), dtype="float32")
            tfidf_vec = torch.tensor(tfidf_vec, dtype=torch.float32)
            return img, token_ids, torch.tensor(emo, dtype=torch.long), tfidf_vec

        # Non-TFIDF mode: return only three items
        return img, token_ids, torch.tensor(emo, dtype=torch.long)

# -------------------------
# Training / Validation helpers
# -------------------------
def train_one_epoch(model, dataloader, optimizer, device, criterion, embedding_type="random"):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in dataloader:
        if embedding_type == "tfidf":
            imgs, token_ids, emos, tfidf_vecs = batch
            tfidf_vecs = tfidf_vecs.to(device)
        else:
            imgs, token_ids, emos = batch
            tfidf_vecs = None
        imgs = imgs.to(device)
        token_ids = token_ids.to(device)
        emos = emos.to(device)
        optimizer.zero_grad()
        if embedding_type == "tfidf":
            logits = model(imgs, token_ids, emos, tfidf_vecs)  # (B, T, V)
        else:
            logits = model(imgs, token_ids, emos)
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

def validate_epoch(model, dataloader, device, criterion, embedding_type="random"):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            if embedding_type == "tfidf":
                imgs, token_ids, emos, tfidf_vecs = batch
                tfidf_vecs = tfidf_vecs.to(device)
            else:
                imgs, token_ids, emos = batch
                tfidf_vecs = None
            imgs = imgs.to(device)
            token_ids = token_ids.to(device)
            emos = emos.to(device)
            if embedding_type == "tfidf":
                logits = model(imgs, token_ids, emos, tfidf_vecs)
            else:
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
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", default=DEFAULT_CONFIG["train_csv"])
    p.add_argument("--val-csv", default=DEFAULT_CONFIG["val_csv"])
    p.add_argument("--images-root", default=DEFAULT_CONFIG["images_features_root"])
    p.add_argument("--vocab", default=DEFAULT_CONFIG["vocab_path"])
    p.add_argument("--device", default=DEFAULT_CONFIG["device"])
    p.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--num-epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    p.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    p.add_argument("--checkpoint-root", default=DEFAULT_CONFIG["checkpoint_root"])
    p.add_argument("--embedding-type", choices=["random", "glove", "fasttext", "tfidf"], default="tfidf")
    p.add_argument("--pretrained-emb", default=None, help="path to .npy pretrained embedding matrix aligned to vocab")
    p.add_argument("--freeze-emb", action="store_true", help="freeze token embedding weights (if loaded directly)")
    p.add_argument("--tfidf-dir", default=None, help="dir with per-row or per-painting tfidf .npy files (required for tfidf mode)")
    p.add_argument("--max-seq-len", type=int, default=DEFAULT_CONFIG["max_seq_len"])
    p.add_argument("--vocab-size", type=int, default=None, help="override vocab size")
    p.add_argument("--num-emotions", type=int, default=DEFAULT_CONFIG["num_emotions"])
    return p.parse_args()

def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG.copy()
    cfg["train_csv"] = args.train_csv
    cfg["val_csv"] = args.val_csv
    cfg["images_features_root"] = args.images_root
    cfg["vocab_path"] = args.vocab
    cfg["device"] = args.device
    cfg["batch_size"] = args.batch_size
    cfg["num_epochs"] = args.num_epochs
    cfg["learning_rate"] = args.learning_rate
    cfg["checkpoint_root"] = args.checkpoint_root
    cfg["model2_subdir"] = "model2"
    cfg["max_seq_len"] = args.max_seq_len
    cfg["num_emotions"] = args.num_emotions

    device = torch.device(cfg["device"])
    print("Using device:", device)

    # Prepare checkpoint dirs
    root = Path(cfg["checkpoint_root"])
    model2_dir = root / cfg["model2_subdir"]
    model2_dir.mkdir(parents=True, exist_ok=True)
    history_path = model2_dir / "vlt_history.json"

    # Load or init history (single JSON storing all embedding runs)
    if history_path.exists():
        try:
            history = json.load(open(history_path, "r"))
            if not isinstance(history, dict):
                history = {"by_embedding": {}}
        except Exception:
            history = {"by_embedding": {}}
    else:
        history = {"by_embedding": {}}

    # Migrate legacy flat 'epochs' entries if present (move them under by_embedding)
    # If historical entries lack embedding info, attribute them to the current embedding type.
    legacy_epochs = history.get("epochs", None)
    if legacy_epochs:
        # ensure by_embedding exists
        history.setdefault("by_embedding", {})
        for e in legacy_epochs:
            emb = e.get("embedding_type", args.embedding_type)
            history["by_embedding"].setdefault(emb, {"epochs": [], "best_val": None, "best_ckpt": None})
            history["by_embedding"][emb]["epochs"].append(e)
        # remove top-level legacy epochs to avoid duplication
        history.pop("epochs", None)

    # Ensure structure exists for current embedding type
    emb = args.embedding_type
    history.setdefault("by_embedding", {})
    history["by_embedding"].setdefault(emb, {"epochs": [], "best_val": None, "best_ckpt": None})

    # Derive best_val for current embedding from history if available
    best_val = float("inf")
    emb_epochs = history["by_embedding"][emb].get("epochs", [])
    for e in emb_epochs:
        if e.get("val_loss") is not None:
            try:
                best_val = min(best_val, float(e["val_loss"]))
            except Exception:
                pass
    if best_val == float("inf"):
        best_val = float("inf")
    else:
        history["by_embedding"][emb]["best_val"] = best_val

    # Load vocab to set vocab size and idx2tok
    tok2idx = None
    idx2tok = None
    if Path(cfg["vocab_path"]).exists():
        try:
            tok2idx = pickle.load(open(cfg["vocab_path"], "rb"))
            if not isinstance(tok2idx, dict) and hasattr(tok2idx, "token_to_idx"):
                tok2idx = tok2idx.token_to_idx
            if args.vocab_size:
                cfg["vocab_size"] = args.vocab_size
            else:
                cfg["vocab_size"] = len(tok2idx)
            idx2tok = {i: t for t, i in tok2idx.items()}
            print("Loaded vocab size:", cfg["vocab_size"])
        except Exception:
            print("Could not load vocab.pkl; using config vocab_size")

    # Load pretrained embedding matrix if requested or auto-detect
    pretrained_matrix = None
    if args.pretrained_emb:
        p = Path(args.pretrained_emb)
        if p.exists():
            print("Loading pretrained embedding matrix from:", p)
            pretrained_matrix = np.load(p)
    else:
        # try sensible defaults in data_preprocessed
        if args.embedding_type == "glove":
            cand = Path("data_preprocessed/emb_glove_300d.npy")
            if cand.exists():
                pretrained_matrix = np.load(cand)
                print("Auto-loaded GloVe embedding matrix:", cand)
        if args.embedding_type == "fasttext" and pretrained_matrix is None:
            cand = Path("data_preprocessed/emb_fasttext_300d.npy")
            if cand.exists():
                pretrained_matrix = np.load(cand)
                print("Auto-loaded FastText embedding matrix:", cand)

    # For TF-IDF we need tfidf_dir and a dimension guess (load first file to infer dim)
 
            # --- TF-IDF detection with auto-search for data_preprocessed/tfidf_npy ---
    tfidf_dim = None
    tfidf_dir = None

    if args.embedding_type == "tfidf":
        candidates = []

        # If user provided a directory, try that first
        if args.tfidf_dir:
            candidates.append(Path(args.tfidf_dir))

        # Auto-detect common TF-IDF folders (including yours)
        candidates.extend([
            Path("data_preprocessed/tfidf_npy"),      # <-- YOUR FOLDER
            Path("data_preprocessed/tfidf"),
            Path("data_preprocessed/tfidf_vectors"),
            Path("data_preprocessed/tfidf_vecs"),
            Path("data_preprocessed/tfidf.npy"),      # single-file fallback
        ])

        found = None
        for c in candidates:
            if c.exists():
                if c.is_dir() and any(c.glob("*.npy")):
                    found = c
                    break
                if c.is_file() and c.suffix == ".npy":
                    found = c
                    break

        if found is None:
            raise RuntimeError(
                "TF-IDF embedding selected but no TF-IDF directory/file found.\n"
                "Tip: Your folder appears to be: data_preprocessed/tfidf_npy\n"
                "Run: --tfidf-dir data_preprocessed/tfidf_npy"
            )

        tfidf_dir = found

        # Infer dimension
        if tfidf_dir.is_file():
            arr = np.load(tfidf_dir)
            if arr.ndim != 2:
                raise RuntimeError(f"TF-IDF file shape invalid: {arr.shape}")
            tfidf_dim = int(arr.shape[1])
            print("Detected single-file TF-IDF matrix:", tfidf_dir, "dim =", tfidf_dim)
        else:
            sample = next(tfidf_dir.glob("*.npy"))
            tfidf_dim = int(np.load(sample).shape[-1])
            print("Detected TF-IDF directory:", tfidf_dir, "dim =", tfidf_dim)
    

    # Datasets
    # Pass the detected tfidf_dir (may be Path or None) into the Dataset
    train_ds = CaptionDataset(cfg["train_csv"], cfg["images_features_root"], max_len=cfg["max_seq_len"],
                            token_col_candidates=None, embedding_type=args.embedding_type, tfidf_dir=tfidf_dir)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    val_loader = None
    if cfg["val_csv"] and Path(cfg["val_csv"]).exists():
        val_ds = CaptionDataset(cfg["val_csv"], cfg["images_features_root"], max_len=cfg["max_seq_len"],
                                token_col_candidates=None, embedding_type=args.embedding_type, tfidf_dir=tfidf_dir)
        val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    # Build model
    model = VisionLanguageTransformer(cfg, pretrained_token_emb_weights=pretrained_matrix,
                                      embedding_type=args.embedding_type, tfidf_dim=tfidf_dim, freeze_emb=args.freeze_emb)
    model.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training loop
    history["by_embedding"].setdefault(emb, {"epochs": [], "best_val": None, "best_ckpt": None})
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion, embedding_type=args.embedding_type)
        t1 = time.time()
        print(f"Epoch {epoch}/{args.num_epochs}: train_loss={train_loss:.4f}  time={t1-t0:.1f}s")

        val_loss = None
        if val_loader is not None:
            val_loss = validate_epoch(model, val_loader, device, criterion, embedding_type=args.embedding_type)
            print(f"               val_loss={val_loss:.4f}")

        # Save epoch checkpoint
        epoch_ckpt = model2_dir / f"m2_epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": cfg,
            "embedding_type": args.embedding_type
        }, epoch_ckpt)
        print(f"Saved checkpoint: {epoch_ckpt}")

        # Update history entry for this embedding
        hist_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "timestamp": datetime.datetime.now().isoformat(),
            "embedding_type": args.embedding_type
        }
        history["by_embedding"][emb].setdefault("epochs", []).append(hist_entry)
        history["by_embedding"][emb]["last_updated"] = datetime.datetime.now().isoformat()
        # optionally update global last_updated
        history["last_updated"] = datetime.datetime.now().isoformat()
        atomic_write_json(history, str(history_path))
        print(f"Updated history -> {history_path}")

        # Save best model by val_loss per-embedding if available
        if val_loss is not None:
            current_best = history["by_embedding"][emb].get("best_val", None)
            if (current_best is None) or (val_loss < float(current_best)):
                history["by_embedding"][emb]["best_val"] = float(val_loss)
                best_path = model2_dir / f"m2_best_{emb}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": cfg,
                    "embedding_type": args.embedding_type
                }, best_path)
                history["by_embedding"][emb]["best_ckpt"] = str(best_path)
                history["by_embedding"][emb]["best_saved_at"] = datetime.datetime.now().isoformat()
                atomic_write_json(history, str(history_path))
                print(f"New BEST for embedding '{emb}' saved -> {best_path}")

    # finalize history
    history["by_embedding"][emb]["finished_at"] = datetime.datetime.now().isoformat()
    history["finished_at"] = datetime.datetime.now().isoformat()
    atomic_write_json(history, str(history_path))
    print("Training finished. History saved to:", history_path)

if __name__ == "__main__":
    main()
# scripts/model2_vlt.py
"""
Simplified Vision-Language Transformer script
- No multiprocessing
- No pin_memory
- Dataloaders are universal for all OS
"""

import os, time, math, json, datetime
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import pickle

# ---------- CONFIG ----------
DEFAULT_CONFIG = {
    "image_size": 224,
    "patch_size": 32,
    "vit_embed_dim": 256,
    "vit_depth": 2,
    "vit_num_heads": 4,
    "decoder_embed_dim": 256,
    "decoder_depth": 2,
    "decoder_num_heads": 4,
    "vocab_size": 8000,
    "max_seq_len": 20,
    "dropout": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ---------- Helpers ----------
def sinusoidal_positional_encoding(n_pos: int, d_model: int):
    pe = torch.zeros(n_pos, d_model)
    position = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# ---------- PatchEmbed ----------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, E, Hn, Wn = x.shape
        return x.flatten(2).transpose(1, 2)

# ---------- Vision Transformer Encoder ----------
class VisionTransformerEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=256, depth=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        P = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(sinusoidal_positional_encoding(P + 1, embed_dim), requires_grad=False)

        layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim*4, dropout=dropout)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        N = x.shape[1]

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        pos = self.pos_embed.unsqueeze(0).to(x.device)
        if pos.size(1) != x.size(1):
            pos = pos[:, :x.size(1), :]

        x = x + pos
        x = x.transpose(0, 1)
        x = self.encoder(x)
        x = self.norm(x.transpose(0, 1))
        return x

# ---------- Decoder Layer ----------
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

# ---------- Decoder ----------
class TransformerDecoderCustom(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [layer] + [type(layer)(layer.self_attn.embed_dim, layer.self_attn.num_heads, layer.linear1.out_features, layer.dropout.p)
                       for _ in range(num_layers - 1)]
        )

    def forward(self, tgt, memory, tgt_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask)
        return tgt

# ---------- Full Model ----------
class VisionLanguageTransformer(nn.Module):
    def __init__(self, config=DEFAULT_CONFIG, token_embedding_weights=None):
        super().__init__()
        self.encoder = VisionTransformerEncoder(
            config["image_size"], config["patch_size"], config["vit_embed_dim"],
            config["vit_depth"], config["vit_num_heads"], config["dropout"]
        )

        self.token_embed = nn.Embedding(config["vocab_size"], config["decoder_embed_dim"])
        if token_embedding_weights is not None:
            w = torch.tensor(token_embedding_weights, dtype=torch.float32)
            if w.shape == (config["vocab_size"], config["decoder_embed_dim"]):
                self.token_embed.weight.data.copy_(w)

        self.pos_embed_tokens = nn.Parameter(
            sinusoidal_positional_encoding(config["max_seq_len"], config["decoder_embed_dim"]),
            requires_grad=False
        )

        if config["vit_embed_dim"] != config["decoder_embed_dim"]:
            self.enc_to_dec = nn.Linear(config["vit_embed_dim"], config["decoder_embed_dim"])
        else:
            self.enc_to_dec = nn.Identity()

        layer = DecoderLayerCustom(
            config["decoder_embed_dim"],
            config["decoder_num_heads"],
            config["decoder_embed_dim"] * 4,
            config["dropout"]
        )
        self.decoder = TransformerDecoderCustom(layer, config["decoder_depth"])

        self.output_proj = nn.Linear(config["decoder_embed_dim"], config["vocab_size"])

    def forward(self, images, token_ids):
        B, T = token_ids.shape
        enc = self.encoder(images)
        enc = self.enc_to_dec(enc)

        tok_emb = self.token_embed(token_ids)
        tgt = tok_emb + self.pos_embed_tokens[:T].unsqueeze(0).to(images.device)
        tgt = tgt.transpose(0, 1)
        memory = enc.transpose(0, 1)

        tgt_mask = torch.triu(torch.ones(T, T, device=images.device), diagonal=1).bool()
        out = self.decoder(tgt, memory, tgt_mask)
        out = out.transpose(0, 1)

        return self.output_proj(out)

# ---------- Dataset ----------
class CaptionDataset(Dataset):
    def __init__(self, csv_path, images_root, max_len=DEFAULT_CONFIG["max_seq_len"]):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.images_root = Path(images_root)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_root / f"{row['painting']}.npy"

        if img_path.exists():
            arr = np.load(img_path)
            img = torch.tensor(arr).permute(2, 0, 1).float()
        else:
            img = torch.zeros(3, DEFAULT_CONFIG["image_size"], DEFAULT_CONFIG["image_size"]).float()

        tok = row["token_ids"]
        token_ids = eval(tok) if isinstance(tok, str) else tok
        token_ids = list(token_ids)[:self.max_len]
        token_ids += [0] * (self.max_len - len(token_ids))
        return img, torch.tensor(token_ids, dtype=torch.long)

# ---------- Training ----------
def train_one_epoch(model, dataloader, optimizer, device, epoch, max_steps=None):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0

    for i, (imgs, token_ids) in enumerate(dataloader):
        imgs, token_ids = imgs.to(device), token_ids.to(device)

        optimizer.zero_grad()
        logits = model(imgs, token_ids)

        logits_in = logits[:, :-1].contiguous()
        targets = token_ids[:, 1:].contiguous()

        B, Tm, V = logits_in.shape
        loss = criterion(
            logits_in.reshape(B * Tm, V),
            targets.reshape(B * Tm)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if max_steps is not None and (i+1) >= max_steps:
            break

    return total_loss / max(1, i + 1)

# ---------- Main ----------
if __name__ == "__main__":
    NUM_EPOCHS = 3
    BATCH_SIZE = 16

    device = DEFAULT_CONFIG["device"]
    print("Using device:", device)

    vocab_path = "data_preprocessed/vocab.pkl"
    token_to_idx = pickle.load(open(vocab_path, "rb")) if Path(vocab_path).exists() else None

    train_csv = "data_preprocessed/train.csv"
    val_csv = "data_preprocessed/val.csv"
    images_root = Path("data_preprocessed/features")

    train_ds = CaptionDataset(train_csv, images_root)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    val_loader = None
    if Path(val_csv).exists():
        val_ds = CaptionDataset(val_csv, images_root)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = VisionLanguageTransformer(DEFAULT_CONFIG)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ckpt_dir = Path("checkpoints")
    (ckpt_dir / "m2_pt").mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "summary").mkdir(exist_ok=True)

    summary_path = ckpt_dir / "summary" / "vlt_summary.json"
    summary = {
        "train_loss": [],
        "val_loss": [],
        "best": {"epoch": None, "train_loss": None, "val_loss": None, "checkpoint": None},
        "config": DEFAULT_CONFIG,
        "started_at": str(datetime.datetime.now())
    }
    json.dump(summary, open(summary_path, "w"), indent=2)

    best_val = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        summary["train_loss"].append(None)
        summary["val_loss"].append(None)
        json.dump(summary, open(summary_path, "w"), indent=2)

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

        val_loss = None
        if val_loader is not None:
            model.eval()
            total = 0
            n = 0
            with torch.no_grad():
                for imgs, token_ids in val_loader:
                    imgs, token_ids = imgs.to(device), token_ids.to(device)
                    logits = model(imgs, token_ids)
                    logits_in = logits[:, :-1]
                    targets = token_ids[:, 1:]
                    B, Tm, V = logits_in.shape
                    loss = nn.CrossEntropyLoss(ignore_index=0)(logits_in.reshape(B*Tm, V), targets.reshape(B*Tm))
                    total += loss.item()
                    n += 1
            val_loss = total / max(1, n)
            print(f"          val_loss={val_loss:.4f}")

        # Save checkpoint
        ckpt_path = ckpt_dir / "m2_pt" / f"m2_epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "token_to_idx": token_to_idx,
            "config": DEFAULT_CONFIG
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # Update summary
        summary["train_loss"][-1] = train_loss
        summary["val_loss"][-1] = val_loss

        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            best_path = ckpt_dir / "m2_pt" / "vlt_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "token_to_idx": token_to_idx,
                "config": DEFAULT_CONFIG
            }, best_path)
            summary["best"] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "checkpoint": str(best_path)
            }

        json.dump(summary, open(summary_path, "w"), indent=2)

    summary["finished_at"] = str(datetime.datetime.now())
    json.dump(summary, open(summary_path, "w"), indent=2)

    print(f"Training complete. Summary saved to {summary_path}")

#!/usr/bin/env python3
"""
model2_vlt_embed.py

Vision-Language Transformer (Model 2) with:
- Custom ViT-like image encoder
- Transformer decoder for text
- Emotion conditioning via separate embedding
- Pluggable text embedding types: glove | fasttext | tfidf

Data assumptions (from preprocessing.py):
- CSV: new_preprocessed/artemis_preprocessed.csv
  columns: painting, token_ids_str, emotion_label, split, ...
- Image features: new_preprocessed/features/<painting>.npy  (H,W,3 in [0,1])
"""

import os
import os.path as osp
import json
import datetime
from pathlib import Path
from ast import literal_eval

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from scripts.embedding_utils import get_embedding_matrix
except ImportError:
    from embedding_utils import get_embedding_matrix


class CaptionDatasetVLT(Dataset):
    """
    Uses:
      - painting (for image features)
      - token_ids_str (pre-tokenized ids from preprocessing)
      - emotion_label (0..8)
      - split (train/val)
    """

    def __init__(self, csv_df, features_root, max_len=25):
        """
        csv_df: a pandas DataFrame filtered by split ('train' or 'val')
        features_root: folder containing <painting>.npy
        """
        self.df = csv_df.reset_index(drop=True)
        self.features_root = Path(features_root)
        self.max_len = max_len

        for col in ["painting", "token_ids_str", "emotion_label"]:
            if col not in self.df.columns:
                raise RuntimeError(f"Column '{col}' missing in dataframe")

    def __len__(self):
        return len(self.df)

    def _parse_ids(self, s):
        """Parse token_ids_str into a LongTensor of length max_len."""
        if isinstance(s, str):
            s = s.strip()
            try:
                if s.startswith("[") and s.endswith("]"):
                    ids = literal_eval(s)
                else:
                    ids = [int(x) for x in s.split() if x.strip().isdigit()]
            except Exception:
                ids = []
        else:
            ids = list(s) if s is not None else []

        ids = ids[:self.max_len]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def _load_image(self, painting):
        path = self.features_root / f"{painting}.npy"
        if not path.exists():
            # fallback if feature missing
            return torch.zeros(3, 128, 128).float()
        arr = np.load(path)
        if arr.ndim == 3:
            # H,W,3 in [0,1]
            img = torch.tensor(arr).permute(2, 0, 1).float()
        else:
            # if somehow already (3,H,W)
            img = torch.tensor(arr).float()
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        painting = row["painting"]
        img = self._load_image(painting)
        token_ids = self._parse_ids(row["token_ids_str"])
        emo = int(row["emotion_label"])
        emo = torch.tensor(emo, dtype=torch.long)
        return img, token_ids, emo



def sinusoidal_positional_encoding(n_pos: int, d_model: int):
    pe = torch.zeros(n_pos, d_model)
    position = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class PatchEmbed(nn.Module):
    """
    Simple patch embedding: conv with stride=patch_size.
    """

    def __init__(self, img_size=128, patch_size=32, in_chans=3, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B, E, H', W')
        x = self.proj(x)              
        B, E, Hn, Wn = x.shape
        # (B, N, E)
        x = x.flatten(2).transpose(1, 2)  
        return x


class VisionTransformerEncoder(nn.Module):
    

    def __init__(self, img_size=128, patch_size=32,
                 embed_dim=256, depth=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        P = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        pos = sinusoidal_positional_encoding(P + 1, embed_dim)
        self.pos_embed = nn.Parameter(pos, requires_grad=False)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=False  
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        
        B = x.size(0)
        # (B, N, D)
        x = self.patch_embed(x)  
        cls = self.cls_token.expand(B, -1, -1)
        # (B, N+1, D)
        x = torch.cat([cls, x], dim=1)  

        pos = self.pos_embed.unsqueeze(0).to(x.device)
        if pos.size(1) != x.size(1):
            pos = pos[:, :x.size(1), :]
        x = x + pos

        # (S, B, D)
        x = x.transpose(0, 1)    
        x = self.encoder(x)
        # (B, S, D)
        x = self.norm(x.transpose(0, 1))  
        return x


class DecoderLayerCustom(nn.Module):


    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_ff = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + tgt2)

        # Cross-attention
        tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = self.norm2(tgt + tgt2)

        # Feed-forward
        ff = self.linear2(self.dropout_ff(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + ff)
        return tgt


class TransformerDecoderCustom(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [layer] +
            [DecoderLayerCustom(layer.self_attn.embed_dim,
                                layer.self_attn.num_heads,
                                layer.linear1.out_features,
                                layer.dropout_ff.p) for _ in range(num_layers - 1)]
        )

    def forward(self, tgt, memory, tgt_mask=None):
        out = tgt
        for layer in self.layers:
            out = layer(out, memory, tgt_mask=tgt_mask)
        return out


class VisionLanguageTransformerEmotion(nn.Module):
   

    def __init__(self,
                 cfg,
                 vocab_size,
                 token_embed_dim,
                 token_emb_matrix=None,
                 freeze_token_emb=True):
        super().__init__()
        self.cfg = cfg
        D_enc = cfg["vit_embed_dim"]
        D_dec = token_embed_dim
        self.pad_idx = cfg.get("pad_idx", 0)

        # Vision encoder
        self.encoder = VisionTransformerEncoder(
            img_size=cfg["image_size"],
            patch_size=cfg["patch_size"],
            embed_dim=D_enc,
            depth=cfg["vit_depth"],
            num_heads=cfg["vit_num_heads"],
            dropout=cfg["dropout"]
        )

        # Project encoder output to decoder dimension if needed
        if D_enc != D_dec:
            self.enc_to_dec = nn.Linear(D_enc, D_dec)
        else:
            self.enc_to_dec = nn.Identity()

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, D_dec, padding_idx=self.pad_idx)
        if token_emb_matrix is not None:
            w = torch.tensor(token_emb_matrix, dtype=torch.float32)
            if w.shape == (vocab_size, D_dec):
                self.token_embed.weight.data.copy_(w)
            else:
                # if shapes mismatch, copy overlapping part
                min_v = min(vocab_size, w.shape[0])
                min_d = min(D_dec, w.shape[1])
                self.token_embed.weight.data[:min_v, :min_d] = w[:min_v, :min_d]
        self.token_embed.weight.requires_grad = not freeze_token_emb

        # Emotion embedding
        self.num_emotions = cfg["num_emotions"]
        self.emotion_emb = nn.Embedding(self.num_emotions, D_dec)
        nn.init.xavier_uniform_(self.emotion_emb.weight)

        # Positional encoding for decoder (extra slot for emotion token)
        pe_tokens = sinusoidal_positional_encoding(cfg["max_seq_len"] + 1, D_dec)
        self.pos_embed_tokens = nn.Parameter(pe_tokens, requires_grad=False)

        # Decoder
        layer = DecoderLayerCustom(
            d_model=D_dec,
            nhead=cfg["decoder_num_heads"],
            dim_feedforward=D_dec * 4,
            dropout=cfg["dropout"]
        )
        self.decoder = TransformerDecoderCustom(layer, cfg["decoder_depth"])

        # Final projection
        self.output_proj = nn.Linear(D_dec, vocab_size)

    def forward(self, images, token_ids, emo_ids):
    
        device = images.device
        B, T = token_ids.shape

        # encode image
        mem = self.encoder(images)         
        mem = self.enc_to_dec(mem)        
        mem = mem.transpose(0, 1)          

        # token + emotion embeddings
        tok_emb = self.token_embed(token_ids)     
        emo_vec = self.emotion_emb(emo_ids)       
        emo_vec = emo_vec.unsqueeze(1)            

        dec_in = torch.cat([emo_vec, tok_emb], dim=1)    

        pos = self.pos_embed_tokens[:T+1, :].unsqueeze(0).to(device)  
        dec_in = dec_in + pos

        dec_in = dec_in.transpose(0, 1)   

        tgt_mask = torch.triu(torch.ones(T+1, T+1, device=device), diagonal=1).bool()

        dec_out = self.decoder(dec_in, mem, tgt_mask=tgt_mask)  
        dec_out = dec_out.transpose(0, 1)                       

        # ignore the emotion position for prediction
        logits = self.output_proj(dec_out[:, 1:, :])  
        return logits



def train_one_epoch(model, loader, optimizer, device, pad_idx=0, epoch_idx=1, total_epochs=1):
    model.train()
    ce = nn.CrossEntropyLoss(ignore_index=pad_idx)
    total_loss = 0.0
    n_samples = 0

    loop = tqdm(loader, desc=f"Train epoch {epoch_idx}/{total_epochs}", leave=False)

    for imgs, token_ids, emos in loop:
        imgs = imgs.to(device)
        token_ids = token_ids.to(device)
        emos = emos.to(device)

        optimizer.zero_grad()
        logits = model(imgs, token_ids, emos)  

        # next token prediction
        logits_in = logits[:, :-1, :]  
        targets = token_ids[:, 1:]      

        B, Tm, V = logits_in.shape
        loss = ce(logits_in.reshape(B * Tm, V), targets.reshape(B * Tm))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * B
        n_samples += B

        avg_loss = total_loss / max(1, n_samples)
        loop.set_postfix(loss=f"{avg_loss:.4f}")

    return total_loss / max(1, n_samples)


def eval_epoch(model, loader, device, pad_idx=0, epoch_idx=1, total_epochs=1):
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=pad_idx)
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        loop = tqdm(loader, desc=f"Val   epoch {epoch_idx}/{total_epochs}", leave=False)
        for imgs, token_ids, emos in loop:
            imgs = imgs.to(device)
            token_ids = token_ids.to(device)
            emos = emos.to(device)

            logits = model(imgs, token_ids, emos)
            logits_in = logits[:, :-1, :]
            targets = token_ids[:, 1:]

            B, Tm, V = logits_in.shape
            loss = ce(logits_in.reshape(B * Tm, V), targets.reshape(B * Tm))
            total_loss += loss.item() * B
            n_samples += B

            avg_loss = total_loss / max(1, n_samples)
            loop.set_postfix(loss=f"{avg_loss:.4f}")

    return total_loss / max(1, n_samples)



def main():
    import argparse

    parser = argparse.ArgumentParser(description="Vision-Language Transformer (Model 2) with embedding types")
    parser.add_argument("--csv", type=str, default="new_preprocessed/artemis_preprocessed.csv")
    parser.add_argument("--features-root", type=str, default="new_preprocessed/features")
    parser.add_argument("--vocab", type=str, default="new_preprocessed/vocab.pkl")
    parser.add_argument("--repr-dir", type=str, default="new_preprocessed")
    parser.add_argument("--embedding-type", type=str,
                        default="tfidf",
                        choices=["glove", "fasttext", "tfidf"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-root", type=str, default="new_checkpoints")
    parser.add_argument("--max-seq-len", type=int, default=25)
    parser.add_argument("--num-emotions", type=int, default=9)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--vit-embed-dim", type=int, default=256)
    parser.add_argument("--vit-depth", type=int, default=2)
    parser.add_argument("--vit-heads", type=int, default=4)
    parser.add_argument("--dec-depth", type=int, default=2)
    parser.add_argument("--dec-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()
    device = torch.device(args.device)
    print("Device:", device)
    print("Embedding type:", args.embedding_type)

    # load embedding matrix + vocab
    emb_matrix, emb_dim, tok2idx, idx2tok = get_embedding_matrix(
        args.embedding_type,
        vocab_path=args.vocab,
        repr_dir=args.repr_dir
    )
    vocab_size = len(tok2idx)
    pad_idx = tok2idx.get("<pad>", 0)

    if emb_dim is None:
        emb_dim = 256  

    cfg = {
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "vit_embed_dim": args.vit_embed_dim,
        "vit_depth": args.vit_depth,
        "vit_num_heads": args.vit_heads,
        "decoder_embed_dim": emb_dim,          
        "decoder_depth": args.dec_depth,
        "decoder_num_heads": args.dec_heads,
        "max_seq_len": args.max_seq_len,
        "dropout": args.dropout,
        "num_emotions": args.num_emotions,
        "pad_idx": pad_idx,
    }

    # read full CSV and split
    full_df = pd.read_csv(args.csv)
    train_df = full_df[full_df["split"] == "train"].copy()
    val_df = full_df[full_df["split"] == "val"].copy()

    print("Train rows:", len(train_df), "Val rows:", len(val_df))

    train_ds = CaptionDatasetVLT(train_df, args.features_root, max_len=args.max_seq_len)
    val_ds = CaptionDatasetVLT(val_df, args.features_root, max_len=args.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = VisionLanguageTransformerEmotion(
        cfg=cfg,
        vocab_size=vocab_size,
        token_embed_dim=emb_dim,
        token_emb_matrix=emb_matrix,
        freeze_token_emb=(args.embedding_type != "random")
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_dir = Path(args.checkpoint_root) / args.embedding_type
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    hist_path = ckpt_dir / "vlt_history.json"

    history = {
        "embedding_type": args.embedding_type,
        "config": vars(args),
        "epochs": []
    }
    best_val = float("inf")

    for epoch in range(1, args.num_epochs + 1):
        t_start = datetime.datetime.now().isoformat()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            pad_idx=pad_idx, epoch_idx=epoch, total_epochs=args.num_epochs
        )
        val_loss = eval_epoch(
            model, val_loader, device,
            pad_idx=pad_idx, epoch_idx=epoch, total_epochs=args.num_epochs
        )

        print(f"Epoch {epoch}/{args.num_epochs}: train={train_loss:.4f}, val={val_loss:.4f}")

        # save epoch checkpoint
        ep_ckpt = ckpt_dir / f"m2_{args.embedding_type}_epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "tok2idx": tok2idx,
            "cfg": cfg,
        }, ep_ckpt)

        # update history
        history["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "timestamp": t_start,
            "checkpoint": str(ep_ckpt)
        })
        history["last_updated"] = datetime.datetime.now().isoformat()

        # save best
        if val_loss < best_val:
            best_val = val_loss
            best_path = ckpt_dir / f"m2_{args.embedding_type}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "tok2idx": tok2idx,
                "cfg": cfg,
            }, best_path)
            history["best"] = {
                "epoch": epoch,
                "val_loss": val_loss,
                "checkpoint": str(best_path),
            }

        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

    history["finished_at"] = datetime.datetime.now().isoformat()
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print("Training finished. History saved at:", hist_path)


if __name__ == "__main__":
    main()
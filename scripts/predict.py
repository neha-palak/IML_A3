#!/usr/bin/env python3
# scripts/predict.py
"""
Predict captions using trained VLT and/or CNN+LSTM models.
Supports:
 - sampling random rows from val.csv (--sample-from-val)
 - painting names that map to data_preprocessed/features/<painting>.npy
 - explicit .npy paths passed in --inputs
 - emotion strings or numeric ids
 - saves JSON predictions to out-dir
"""

import os
import json
import time
import argparse
import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

# -------------------------
# Config / defaults
# -------------------------
DEFAULT_FEATURES_ROOT = "data_preprocessed/features"
DEFAULT_VOCAB = "data_preprocessed/vocab.pkl"
DEFAULT_VAL_CSV = "data_preprocessed/val.csv"
DEFAULT_OUT_DIR = "eval_outputs"

# -------------------------
# Basic utils
# -------------------------
def load_vocab(path):
    with open(path, "rb") as f:
        token_to_idx = pickle.load(f)
    # accept object with token_to_idx
    if not isinstance(token_to_idx, dict) and hasattr(token_to_idx, "token_to_idx"):
        token_to_idx = token_to_idx.token_to_idx
    idx_to_token = {i: t for t, i in token_to_idx.items()}
    return token_to_idx, idx_to_token

def map_emotion_str_to_id(e_str):
    em = {
        "amusement": 0, "contentment": 1, "awe": 2, "excitement": 3,
        "fear": 4, "anger": 5, "sadness": 6, "disgust": 7, "something else": 8
    }
    return em.get(e_str.lower(), 8)

def load_feature_by_name_or_path(name_or_path: str, features_root: str):
    p = Path(name_or_path)
    if p.suffix == ".npy" and p.exists():
        arr = np.load(p)
    else:
        # try features_root/<name>.npy
        cand = Path(features_root) / (name_or_path + ".npy")
        if cand.exists():
            arr = np.load(cand)
        else:
            raise FileNotFoundError(f"Feature file not found for '{name_or_path}'. Tried: {cand}")
    # ensure shape (H,W,C) -> convert to (C,H,W)
    if arr.ndim == 3:
        if arr.shape[2] <= 4:
            # H,W,C -> C,H,W
            arr = arr.transpose(2, 0, 1)
        else:
            # maybe already C,H,W
            arr = arr
    elif arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    else:
        raise RuntimeError(f"Unsupported feature ndim {arr.ndim} for {name_or_path}")
    return torch.tensor(arr).float()

# -------------------------
# Fallback model implementations (compact)
# -------------------------
# Vision-Language Transformer (compact variant similar to training script)
def sinusoidal_positional_encoding(n_pos: int, d_model: int):
    import math
    pe = torch.zeros(n_pos, d_model)
    position = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=32, in_chans=3, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)               # (B, E, H', W')
        B, E, Hn, Wn = x.shape
        return x.flatten(2).transpose(1, 2)  # (B, N, E)

class VisionTransformerEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=32, embed_dim=256, depth=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        P = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        pos = sinusoidal_positional_encoding(P + 1, embed_dim)
        self.pos_embed = nn.Parameter(pos, requires_grad=False)
        layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim*4, dropout=dropout)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = self.pos_embed.unsqueeze(0).to(x.device)
        if pos.size(1) != x.size(1):
            pos = pos[:, :x.size(1), :]
        x = x + pos
        x = x.transpose(0,1)
        x = self.encoder(x)
        x = self.norm(x.transpose(0,1))
        return x

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
        self.layers = nn.ModuleList([layer] + [type(layer)(layer.self_attn.embed_dim, layer.self_attn.num_heads, layer.linear1.out_features, layer.dropout.p) for _ in range(num_layers-1)])
    def forward(self, tgt, memory, tgt_mask=None):
        out = tgt
        for layer in self.layers:
            out = layer(out, memory, tgt_mask=tgt_mask)
        return out

class VisionLanguageTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D_enc = cfg.get("vit_embed_dim", cfg.get("vit_embed", 256))
        D_dec = cfg.get("decoder_embed_dim", cfg.get("decoder_embed", 256))
        self.encoder = VisionTransformerEncoder(img_size=cfg.get("image_size",224),
                                                patch_size=cfg.get("patch_size",32),
                                                embed_dim=D_enc,
                                                depth=cfg.get("vit_depth",2),
                                                num_heads=cfg.get("vit_num_heads",4),
                                                dropout=cfg.get("dropout",0.1))
        self.token_embed = nn.Embedding(cfg["vocab_size"], D_dec)
        pos_tokens = sinusoidal_positional_encoding(cfg.get("max_seq_len",20)+1, D_dec)
        self.pos_embed_tokens = nn.Parameter(pos_tokens, requires_grad=False)
        self.enc_to_dec = nn.Linear(D_enc, D_dec) if D_enc != D_dec else nn.Identity()
        self.emotion_emb = nn.Embedding(cfg.get("num_emotions",9), D_dec)
        nn.init.xavier_uniform_(self.emotion_emb.weight)
        layer = DecoderLayerCustom(D_dec, cfg.get("decoder_num_heads",4), dim_feedforward=D_dec*4, dropout=cfg.get("dropout",0.1))
        self.decoder = TransformerDecoderCustom(layer, cfg.get("decoder_depth",2))
        self.output_proj = nn.Linear(D_dec, cfg["vocab_size"])
        self.pad_idx = 0
    def forward(self, images, token_ids, emo_ids):
        B, T = token_ids.shape
        enc = self.encoder(images)
        enc = self.enc_to_dec(enc)
        memory = enc.transpose(0,1)
        tok_emb = self.token_embed(token_ids)
        emo_vec = self.emotion_emb(emo_ids).unsqueeze(1)
        dec_in = torch.cat([emo_vec, tok_emb], dim=1)
        pos = self.pos_embed_tokens[:(T+1),:].unsqueeze(0).to(images.device)
        dec_in = dec_in + pos
        dec_in = dec_in.transpose(0,1)
        tgt_mask = torch.triu(torch.ones((T+1, T+1), device=images.device), diagonal=1).bool()
        dec_out = self.decoder(dec_in, memory, tgt_mask=tgt_mask)
        dec_out = dec_out.transpose(0,1)
        logits = self.output_proj(dec_out)
        return logits[:,1:,:]   # skip emotion position

    def greedy_decode(self, image, emo_id, sos_idx, eos_idx=None, max_len=None, device='cpu'):
        self.eval()
        if max_len is None: max_len = self.pos_embed_tokens.size(0)-1
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            enc = self.encoder(image)
            enc = self.enc_to_dec(enc)
            memory = enc.transpose(0,1)
            cur = torch.LongTensor([[sos_idx]]).to(device)
            for step in range(max_len):
                cur_padded = F.pad(cur, (0, max_len - cur.size(1)), value=0)
                logits = self.forward(image, cur_padded, torch.LongTensor([emo_id]).to(device))
                step_idx = cur.size(1)-1
                logit_step = logits[:, step_idx, :]
                next_id = torch.argmax(logit_step, dim=-1).item()
                if eos_idx is not None and next_id == eos_idx:
                    return cur.squeeze().tolist()[1:]  # exclude sos
                cur = torch.cat([cur, torch.LongTensor([[next_id]]).to(device)], dim=1)
            return cur.squeeze().tolist()[1:]

# Simple CNN + LSTM fallback
class SimpleCNNEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(128, out_dim)
        )
    def forward(self,x): return self.net(x)

class CNNLSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        img_feat_dim = cfg.get("img_feat_dim", 256)
        self.encoder = SimpleCNNEncoder(out_dim=img_feat_dim)
        emb_dim = cfg.get("emb_dim", 256)
        self.token_embed = nn.Embedding(cfg["vocab_size"], emb_dim)
        self.emotion_emb = nn.Embedding(cfg.get("num_emotions",9), emb_dim)
        self.lstm = nn.LSTM(emb_dim, cfg.get("lstm_hidden",256), batch_first=True)
        self.out = nn.Linear(cfg.get("lstm_hidden",256), cfg["vocab_size"])
    def forward(self, images, token_ids, emo_ids):
        B,T = token_ids.shape
        img_feat = self.encoder(images).unsqueeze(1)  # (B,1,F)
        tok_emb = self.token_embed(token_ids)         # (B,T,emb)
        emo_vec = self.emotion_emb(emo_ids).unsqueeze(1)
        # prepend emotion and image feature by concatenation-per-time-step trick:
        # We'll prepend a single step representing [EMO] (embedding) and let LSTM consume
        inp = torch.cat([emo_vec, tok_emb], dim=1)  # (B, T+1, emb)
        out, _ = self.lstm(inp)
        logits = self.out(out)  # (B, T+1, V)
        return logits[:,1:,:]

    def greedy_decode(self, image, emo_id, sos_idx, eos_idx=None, max_len=20, device='cpu'):
        self.eval()
        if image.ndim == 3: image = image.unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            img_feat = self.encoder(image)  # (1, F)
            cur = [sos_idx]
            generated = []
            hidden = None
            for step in range(max_len):
                cur_t = torch.LongTensor([cur]).to(device)  # (1, L)
                tok_emb = self.token_embed(cur_t)           # (1,L,emb)
                emo_vec = self.emotion_emb(torch.LongTensor([emo_id]).to(device)).unsqueeze(1)
                inp = torch.cat([emo_vec, tok_emb], dim=1)
                out, hidden = self.lstm(inp, hidden)
                logit = self.out(out[:, -1, :])
                next_id = torch.argmax(logit, dim=-1).item()
                if eos_idx is not None and next_id == eos_idx:
                    break
                generated.append(next_id)
                cur.append(next_id)
            return generated

# -------------------------
# Checkpoint loading helper
# -------------------------
def load_checkpoint_maybe_build(path: str, device: str, model_type: str, vocab_size: int):
    """
    Try to load checkpoint and instantiate an appropriate model using stored config (if exists).
    model_type: "vlt" or "cnn"
    """
    ck = torch.load(path, map_location=device)
    # many checkpoints saved as dict; handle
    if isinstance(ck, dict) and "model_state_dict" in ck and "config" in ck:
        cfg = ck["config"]
        # ensure vocab_size present
        cfg["vocab_size"] = vocab_size
        if model_type == "vlt":
            model = VisionLanguageTransformer(cfg)
        else:
            model = CNNLSTM(cfg)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        model.to(device)
        model.eval()
        return model, ck
    else:
        # fallback: try to directly load state_dict into local model constructed from minimal cfg
        print("Checkpoint missing structured dict with config; trying fallback instantiate.")
        if model_type == "vlt":
            cfg = {"vocab_size": vocab_size, "max_seq_len": 20, "decoder_embed_dim":256, "vit_embed_dim":256, "decoder_num_heads":4, "decoder_depth":2}
            model = VisionLanguageTransformer(cfg)
        else:
            cfg = {"vocab_size": vocab_size, "lstm_hidden":256, "emb_dim":256, "num_emotions":9}
            model = CNNLSTM(cfg)
        try:
            if isinstance(ck, dict):
                model.load_state_dict(ck, strict=False)
            else:
                model.load_state_dict(ck, strict=False)
            model.to(device)
            model.eval()
            return model, {"loaded_raw": True}
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint into fallback model: {e}")

# -------------------------
# Main predict flow
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", help="painting names (no .npy) OR full paths to .npy files", default=[])
    p.add_argument("--emotions", nargs="+", help="emotion strings or numeric ids", default=[])
    p.add_argument("--sample-from-val", action="store_true")
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--val-csv", type=str, default=DEFAULT_VAL_CSV)
    p.add_argument("--features-root", type=str, default=DEFAULT_FEATURES_ROOT)
    p.add_argument("--vlt-ckpt", type=str, default=None)
    p.add_argument("--cnn-ckpt", type=str, default=None)
    p.add_argument("--vocab", type=str, default=DEFAULT_VOCAB)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--max-len", type=int, default=20)
    return p.parse_args()

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    device = torch.device(device)
    # load vocab
    tok2idx, idx2tok = load_vocab(args.vocab)
    sos_idx = tok2idx.get("<start>", 1)
    eos_idx = tok2idx.get("<end>", 2)

    # decide inputs/emotions
    inputs = list(args.inputs)
    emotions_input = list(args.emotions)

    # sample from val if requested
    if args.sample_from_val:
        import pandas as pd
        df = pd.read_csv(args.val_csv)
        if len(df) == 0:
            raise RuntimeError("val.csv empty")
        sampled = df.sample(n=args.num_samples)
        inputs = []
        emotions_input = []
        for _, row in sampled.iterrows():
            inputs.append(row["painting"])
            if "emotion_label" in row and not pd.isna(row["emotion_label"]):
                emotions_input.append(int(row["emotion_label"]))
            elif "emotion" in row and not pd.isna(row["emotion"]):
                emotions_input.append(map_emotion_str_to_id(str(row["emotion"])))
            else:
                emotions_input.append(8)

    if len(inputs) == 0:
        raise RuntimeError("No inputs provided (use --inputs or --sample-from-val)")

    # if emotions provided as strings, map to ids
    emotions = []
    if len(emotions_input) == 0:
        raise RuntimeError("No emotions provided; pass as strings/ids or use sample-from-val to pick from val.csv")
    for e in emotions_input:
        if isinstance(e, (int, np.integer)):
            emotions.append(int(e))
        else:
            try:
                emotions.append(int(e))
            except Exception:
                emotions.append(map_emotion_str_to_id(str(e)))

    if len(inputs) != len(emotions):
        raise ValueError("inputs and emotions must have same length")

    # load models if provided
    vlt_model = None
    cnn_model = None
    if args.vlt_ckpt:
        print("Loading VLT checkpoint:", args.vlt_ckpt)
        vlt_model, ck_vlt = load_checkpoint_maybe_build(args.vlt_ckpt, device, "vlt", vocab_size=len(tok2idx))
    if args.cnn_ckpt:
        print("Loading CNN checkpoint:", args.cnn_ckpt)
        cnn_model, ck_cnn = load_checkpoint_maybe_build(args.cnn_ckpt, device, "cnn", vocab_size=len(tok2idx))

    # prepare outputs
    out = []
    os.makedirs(args.out_dir, exist_ok=True)

    for inp, emo in zip(inputs, emotions):
        try:
            feat = load_feature_by_name_or_path(inp, args.features_root)  # returns C,H,W tensor
        except Exception as e:
            print(f"[WARN] Could not load feature for {inp}: {e}")
            # append empty result
            out.append({"input": inp, "emotion": emo, "error": str(e)})
            continue

        # ensure float tensor on device
        feat_t = feat.to(device).float()
        # models expect (B,3,H,W)
        if feat_t.ndim == 3:
            feat_b = feat_t.unsqueeze(0)
        else:
            feat_b = feat_t

        example = {"input": inp, "emotion": int(emo), "generations": {}}

        # VLT
        if vlt_model is not None:
            try:
                gen_ids = vlt_model.greedy_decode(feat_b, int(emo), sos_idx, eos_idx=eos_idx, max_len=args.max_len, device=device)
                # gen_ids is a list of token ids (may include numbers); decode to words
                words = [idx2tok.get(int(i), "<unk>") for i in gen_ids if int(i) in idx2tok]
                example["generations"]["vlt"] = " ".join(words)
            except Exception as e:
                example["generations"]["vlt_error"] = str(e)

        # CNN+LSTM
        if cnn_model is not None:
            try:
                gen_ids = cnn_model.greedy_decode(feat_b, int(emo), sos_idx, eos_idx=eos_idx, max_len=args.max_len, device=device)
                words = [idx2tok.get(int(i), "<unk>") for i in gen_ids if int(i) in idx2tok]
                example["generations"]["cnn_lstm"] = " ".join(words)
            except Exception as e:
                example["generations"]["cnn_error"] = str(e)

        out.append(example)
        print(f"Processed {inp} | emo={emo} -> {example['generations']}")

    # save results
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out_dir) / f"predictions_{ts}.json"
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Saved predictions to", out_path)

if __name__ == "__main__":
    main()
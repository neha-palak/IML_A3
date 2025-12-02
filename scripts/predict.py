#!/usr/bin/env python3
"""
predict.py

Generate captions from one or more Vision-Language Transformer checkpoints (one per embedding type)
and a single CNN-LSTM checkpoint. Output JSON like:

[
  {"input":"painting-name","emotion":4,"generations":{"vlt_glove":"...","vlt_fasttext":"...","cnn":"..."}},
  ...
]

Usage examples:

# provide painting names and emotions directly:
python scripts/predict.py --inputs "pierre-puvis-de-chavannes_the-beheading-of-st-john-the-baptist-1869,jacob-jordaens_the-childhood-of-zeus" \
    --emotions "4,4" \
    --features-root data_preprocessed/features \
    --vocab data_preprocessed/vocab.pkl \
    --out preds.json

# or rely on found checkpoints and pass a list-file (json lines or simple CSV):
python scripts/predict.py --inputs-file examples_to_run.jsonl --features-root data_preprocessed/features \
    --vocab data_preprocessed/vocab.pkl --out preds.json --beam-width 3

"""
import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import pickle
import glob
import sys
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Utilities (vocab + token helpers)
# ----------------------------
def load_vocab(vocab_path: str):
    with open(vocab_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        token_to_idx = obj
    elif hasattr(obj, "token_to_idx"):
        token_to_idx = obj.token_to_idx
    else:
        # try to coerce list->dict
        if isinstance(obj, (list, tuple)):
            token_to_idx = {t: i for i, t in enumerate(obj)}
        else:
            raise RuntimeError("Unsupported vocab.pkl format")
    idx2token = {i: t for t, i in token_to_idx.items()}
    # find pad/sos/eos defaults
    def _find(keys, default):
        for k in keys:
            if k in token_to_idx:
                return token_to_idx[k]
        return default
    pad_idx = _find(["<pad>", "<PAD>", "[PAD]"], 0)
    sos_idx = _find(["<start>", "<s>", "<START>"], 1)
    eos_idx = _find(["<end>", "</s>", "<END>"], 2)
    return token_to_idx, idx2token, pad_idx, sos_idx, eos_idx

# ----------------------------
# Minimal Vision-Language Transformer (compatible with training script)
# ----------------------------
def sinusoidal_positional_encoding(n_pos: int, d_model: int):
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
        self.num_patches = (img_size // patch_size) ** 2
    def forward(self, x):
        x = self.proj(x)
        B, D, H, W = x.shape
        return x.flatten(2).transpose(1, 2)

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
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = self.pos_embed.unsqueeze(0).to(x.device)
        if pos.size(1) != x.size(1):
            pos = pos[:, :x.size(1), :]
        x = x + pos
        x = x.transpose(0,1)
        x = self.encoder(x)
        return self.norm(x.transpose(0,1))

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

class VisionLanguageTransformer(nn.Module):
    def __init__(self, cfg: dict, token_embedding_weights=None, num_emotions=9):
        super().__init__()
        D_enc = cfg.get("vit_embed_dim", cfg.get("vit_dim", 256))
        D_dec = cfg.get("decoder_embed_dim", cfg.get("decoder_dim", 256))
        self.encoder = VisionTransformerEncoder(img_size=cfg.get("image_size",224), patch_size=cfg.get("patch_size",32),
                                                embed_dim=D_enc, depth=cfg.get("vit_depth",2), num_heads=cfg.get("vit_num_heads",4), dropout=cfg.get("dropout",0.1))
        self.token_embed = nn.Embedding(cfg["vocab_size"], D_dec)
        if token_embedding_weights is not None:
            w = torch.tensor(token_embedding_weights, dtype=torch.float32)
            if w.shape == (cfg["vocab_size"], w.shape[1]) and w.shape[1] == D_dec:
                self.token_embed.weight.data.copy_(w)
        pos_tokens = sinusoidal_positional_encoding(cfg.get("max_seq_len",20) + 1, D_dec)
        self.pos_embed_tokens = nn.Parameter(pos_tokens, requires_grad=False)
        if D_enc != D_dec:
            self.enc_to_dec = nn.Linear(D_enc, D_dec)
        else:
            self.enc_to_dec = nn.Identity()
        self.emotion_emb = nn.Embedding(num_emotions, D_dec)
        nn.init.xavier_uniform_(self.emotion_emb.weight)
        layer = DecoderLayerCustom(D_dec, cfg.get("decoder_num_heads",4), dim_feedforward=D_dec * 4, dropout=cfg.get("dropout",0.1))
        self.decoder = TransformerDecoderCustom(layer, cfg.get("decoder_depth",2))
        self.output_proj = nn.Linear(D_dec, cfg["vocab_size"])
    def forward(self, images, token_ids, emo_ids):
        B, T = token_ids.shape
        device = images.device
        enc = self.encoder(images)
        enc = self.enc_to_dec(enc)
        memory = enc.transpose(0,1)
        tok_emb = self.token_embed(token_ids)
        emo_vec = self.emotion_emb(emo_ids).unsqueeze(1)
        dec_in = torch.cat([emo_vec, tok_emb], dim=1)
        pos = self.pos_embed_tokens[: dec_in.size(1), :].unsqueeze(0).to(device)
        dec_in = dec_in + pos
        dec_in = dec_in.transpose(0,1)
        tgt_mask = torch.triu(torch.ones((dec_in.size(0), dec_in.size(0)), device=device), diagonal=1).bool()
        dec_out = self.decoder(dec_in, memory, tgt_mask=tgt_mask)
        dec_out = dec_out.transpose(0,1)
        logits = self.output_proj(dec_out)
        return logits[:, 1:, :]

    def greedy_decode(self, image, emo_id, sos_idx, eos_idx=None, max_len=None, device='cpu'):
        self.eval()
        if max_len is None:
            max_len = 20
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(device)
        emo = torch.tensor([emo_id], dtype=torch.long).to(device)
        with torch.no_grad():
            enc = self.encoder(image)
            enc = self.enc_to_dec(enc)
            memory = enc.transpose(0,1)
            cur = torch.LongTensor([[sos_idx]]).to(device)
            generated = []
            for step in range(max_len):
                tok_emb = self.token_embed(cur)
                emo_emb = self.emotion_emb(emo).unsqueeze(1)
                dec_in = torch.cat([emo_emb, tok_emb], dim=1)
                pos = self.pos_embed_tokens[: dec_in.size(1), :].unsqueeze(0).to(device)
                dec_in = (dec_in + pos).transpose(0,1)
                tgt_mask = torch.triu(torch.ones((dec_in.size(0), dec_in.size(0)), device=device), diagonal=1).bool()
                dec_out = self.decoder(dec_in, memory, tgt_mask=tgt_mask).transpose(0,1)
                logits_next = self.output_proj(dec_out[:, -1, :])
                next_id = int(torch.argmax(logits_next, dim=-1).item())
                generated.append(next_id)
                if eos_idx is not None and next_id == eos_idx:
                    break
                cur = torch.cat([cur, torch.LongTensor([[next_id]]).to(device)], dim=1)
        return generated

# ----------------------------
# Minimal CNN + LSTM fallback
# ----------------------------
class SmallCNNEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU()
        )
    def forward(self,x):
        return self.net(x)

class CNN_LSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = SmallCNNEncoder(out_dim=cfg.get("img_feat_dim",256))
        self.embed = nn.Embedding(cfg["vocab_size"], cfg.get("dec_embed_dim",256))
        self.lstm = nn.LSTM(cfg.get("dec_embed_dim",256) + cfg.get("img_feat_dim",256), cfg.get("lstm_hidden",256), batch_first=True)
        self.out = nn.Linear(cfg.get("lstm_hidden",256), cfg["vocab_size"])
    def forward(self, images, token_ids, emo_ids=None):
        # token_ids (B,T), returns logits (B,T,V)
        B,T = token_ids.shape
        img_feat = self.encoder(images)  # (B, feat)
        emb = self.embed(token_ids)      # (B,T,emb)
        # repeat img_feat across time and concat
        img_rep = img_feat.unsqueeze(1).expand(-1, T, -1)
        inp = torch.cat([emb, img_rep], dim=-1)
        out,_ = self.lstm(inp)
        logits = self.out(out)
        return logits
    def greedy_decode(self, image, emo_id, sos_idx, eos_idx=None, max_len=20, device='cpu'):
        self.eval()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            feat = self.encoder(image)  # (1,feat)
            cur = torch.LongTensor([[sos_idx]]).to(device)
            hidden = None
            generated = []
            for step in range(max_len):
                emb = self.embed(cur)  # (1,t,emb)
                # take last token embedding
                last_emb = emb[:,-1:,:]
                inp = torch.cat([last_emb, feat.unsqueeze(1)], dim=-1)
                out, hidden = self.lstm(inp, hidden)
                logits_next = self.out(out[:, -1, :])
                next_id = int(torch.argmax(logits_next, dim=-1).item())
                generated.append(next_id)
                if eos_idx is not None and next_id == eos_idx:
                    break
                cur = torch.cat([cur, torch.LongTensor([[next_id]]).to(device)], dim=1)
        return generated

# ----------------------------
# Checkpoint loader helpers
# ----------------------------
def discover_vlt_ckpts(provided_list: List[str]) -> List[str]:
    if provided_list:
        return provided_list
    # search for files that look like vlt best checkpoints
    candidates = glob.glob("checkpoints/**/m2_best*.pt", recursive=True) + glob.glob("checkpoints/**/m2_*best*.pt", recursive=True)
    candidates = sorted(list(set(candidates)))
    return candidates

def discover_cnn_ckpt(provided: str):
    if provided:
        return provided
    candidates = glob.glob("checkpoints/**/cnn_best*.pt", recursive=True) + glob.glob("checkpoints/**/m1_best*.pt", recursive=True)
    candidates = sorted(list(set(candidates)))
    return candidates[0] if candidates else None

def load_checkpoint_into_vlt(ckpt_path, device, vocab_size):
    try:
        ck = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        print("Could not load checkpoint:", ckpt_path, e)
        return None, None
    # extract config if present
    cfg = ck.get("config", {})
    if not cfg:
        # fallback defaults and set vocab_size
        cfg = {
            "image_size": 224,
            "patch_size": 32,
            "vit_embed_dim": 256,
            "vit_depth": 2,
            "vit_num_heads": 4,
            "decoder_embed_dim": 256,
            "decoder_depth": 2,
            "decoder_num_heads": 4,
            "vocab_size": vocab_size,
            "max_seq_len": 20,
            "dropout": 0.1
        }
    else:
        cfg = dict(cfg)
        cfg["vocab_size"] = vocab_size
    model = VisionLanguageTransformer(cfg, token_embedding_weights=None, num_emotions=cfg.get("num_emotions",9))
    model.to(device)
    state = ck.get("model_state_dict", ck)
    try:
        model.load_state_dict(state, strict=False)
        print("Loaded VLT checkpoint into fallback model:", ckpt_path)
    except Exception as e:
        print("Warning: loading with strict=False produced errors for", ckpt_path, e)
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            pass
    model.eval()
    return model, cfg

def load_checkpoint_into_cnn(ckpt_path, device, vocab_size):
    try:
        ck = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        print("Could not load CNN checkpoint:", ckpt_path, e)
        return None, None
    cfg = ck.get("config", {})
    if not cfg:
        cfg = {"vocab_size": vocab_size, "img_feat_dim":256, "dec_embed_dim":256, "lstm_hidden":256}
    else:
        cfg = dict(cfg)
        cfg["vocab_size"] = vocab_size
    model = CNN_LSTM(cfg)
    model.to(device)
    state = ck.get("model_state_dict", ck)
    try:
        model.load_state_dict(state, strict=False)
        print("Loaded CNN checkpoint:", ckpt_path)
    except Exception as e:
        print("Warning: loading CNN with strict=False produced errors", e)
    model.eval()
    return model, cfg

# ----------------------------
# IO helpers to read inputs
# ----------------------------
def read_inputs(args) -> List[Tuple[str,int]]:
    pairs = []
    if args.inputs_file:
        p = Path(args.inputs_file)
        if not p.exists():
            raise FileNotFoundError(args.inputs_file)
        # accept jsonl of {"input":..., "emotion":int} or CSV with two cols
        try:
            for line in p.open():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and "input" in obj and "emotion" in obj:
                    pairs.append((obj["input"], int(obj["emotion"])))
                else:
                    # fallback to tuple-like arrays
                    if isinstance(obj, (list,tuple)) and len(obj) >= 2:
                        pairs.append((obj[0], int(obj[1])))
        except Exception:
            # try CSV simple parsing
            import csv
            with open(p, newline='') as f:
                rdr = csv.reader(f)
                for row in rdr:
                    if len(row) >= 2:
                        pairs.append((row[0].strip(), int(row[1])))
    else:
        if not args.inputs:
            raise ValueError("No inputs provided")
        inputs = [s.strip() for s in args.inputs.split(",")]
        emotions = [int(s.strip()) for s in args.emotions.split(",")]
        if len(inputs) != len(emotions):
            raise ValueError("inputs and emotions must match length")
        pairs = list(zip(inputs, emotions))
    return pairs

# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=str, help="comma-separated painting names (no .npy).")
    p.add_argument("--emotions", type=str, help="comma-separated emotion ids.")
    p.add_argument("--inputs-file", type=str, help="file with inputs (jsonl or csv)")
    p.add_argument("--features-root", type=str, required=True, help="where painting .npy files are stored")
    p.add_argument("--vocab", type=str, required=True)
    p.add_argument("--vlt-ckpts", type=str, nargs="*", help="explicit list of VLT checkpoint files (optional)")
    p.add_argument("--cnn-ckpt", type=str, help="optional CNN-LSTM checkpoint file (optional)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--beam-width", type=int, default=1, help="use beam search when >1")
    p.add_argument("--out", type=str, default="predictions.json")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    token_to_idx, idx2token, pad_idx, sos_idx, eos_idx = load_vocab(args.vocab)
    vocab_size = len(token_to_idx)

    pairs = read_inputs(args)

    # discover vlt ckpts if none provided
    vlt_ckpts = discover_vlt_ckpts(args.vlt_ckpts or [])
    if not vlt_ckpts:
        print("No VLT checkpoints found (use --vlt-ckpts to pass explicit paths).")
    else:
        print("Found VLT checkpoints:", vlt_ckpts)

    cnn_ckpt = discover_cnn_ckpt(args.cnn_ckpt)
    if cnn_ckpt:
        print("Using CNN checkpoint:", cnn_ckpt)
    else:
        print("No CNN checkpoint found (ok: CNN predictions will be skipped)")

    # load models
    vlt_models = []
    for ck in vlt_ckpts:
        try:
            model, cfg = load_checkpoint_into_vlt(ck, device, vocab_size)
            if model is not None:
                # name for model from filename
                name = Path(ck).stem
                # derive a short embed name if exists
                if "glove" in name.lower():
                    short = "vlt_glove"
                elif "fasttext" in name.lower() or "ft" in name.lower():
                    short = "vlt_fasttext"
                elif "random" in name.lower():
                    short = "vlt_random"
                else:
                    short = "vlt_" + name
                vlt_models.append((short, model))
        except Exception as e:
            print("Error loading VLT ckpt", ck, e)

    cnn_model = None
    if cnn_ckpt:
        try:
            cnn_model, cnn_cfg = load_checkpoint_into_cnn(cnn_ckpt, device, vocab_size)
        except Exception as e:
            print("Could not load CNN model:", e)

    results = []
    features_root = Path(args.features_root)
    for painting, emo in pairs:
        rec = {"input": painting, "emotion": int(emo), "generations": {}}
        feat_file = features_root / f"{painting}.npy"
        if not feat_file.exists():
            print("Missing feature file for", painting, "expected at", feat_file)
            rec["error"] = "missing_feature"
            results.append(rec)
            continue
        arr = np.load(feat_file)
        # arrange to torch tensor (C,H,W)
        if arr.ndim == 3:
            img = torch.tensor(arr).permute(2,0,1).unsqueeze(0).float().to(device)
        elif arr.ndim == 2:  # single channel? convert
            img = torch.tensor(arr).unsqueeze(0).unsqueeze(0).float().to(device)
        else:
            # assume HxWxC ordering
            img = torch.tensor(arr).permute(2,0,1).unsqueeze(0).float().to(device)

        # VLT models
        for name, model in vlt_models:
            try:
                if args.beam_width and args.beam_width > 1:
                    # fallback to greedy if beam not implemented
                    if hasattr(model, "beam_search_decode"):
                        seq = model.beam_search_decode(img.squeeze(0), int(emo), sos_idx, eos_idx, max_len=20, beam_width=args.beam_width, device=device)
                    else:
                        seq = model.greedy_decode(img.squeeze(0), int(emo), sos_idx, eos_idx, max_len=20, device=device)
                else:
                    seq = model.greedy_decode(img.squeeze(0), int(emo), sos_idx, eos_idx, max_len=20, device=device)
                words = [idx2token.get(i, "<unk>") for i in seq]
                # postprocess: stop at eos if present
                if eos_idx in seq:
                    words = words[:words.index(idx2token.get(eos_idx, "<end>"))]
                rec["generations"][name] = " ".join(words).strip()
            except Exception as e:
                print("Error generating with", name, e)
                rec["generations"][name] = None

        # CNN model (single)
        if cnn_model is not None:
            try:
                seq = cnn_model.greedy_decode(img.squeeze(0), int(emo), sos_idx, eos_idx, max_len=20, device=device)
                words = [idx2token.get(i, "<unk>") for i in seq]
                if eos_idx in seq:
                    words = words[:seq.index(eos_idx)]
                rec["generations"]["cnn"] = " ".join(words).strip()
            except Exception as e:
                print("Error generating CNN for", painting, e)
                rec["generations"]["cnn"] = None

        results.append(rec)

    # write output
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Saved", len(results), "predictions to", outp)

if __name__ == "__main__":
    main()
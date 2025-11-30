#!/usr/bin/env python3
"""
evaluate.py - robust evaluator for Vision-Language Transformer (VLT)

Features:
- Loads checkpoint (CPU-safe), attempts to instantiate user's model class (if available),
  falls back to a compatible minimal VLT implementation.
- Repairs common state_dict/key differences (patch_embed vs patch, pos-embed length mismatch).
- Greedy decoding (with optional small beam fallback).
- Computes sacreBLEU (corpus) and ROUGE-L (avg best-ref).
- METEOR computed only if NLTK+WordNet already available (no automatic downloads).
- Saves predictions CSV and eval_summary.json in out-dir.

Example:
python3 scripts/evaluate.py --ckpt checkpoints/m2_pt/m2_best.pt \
    --test-csv data_preprocessed/test.csv \
    --images-root data_preprocessed/features \
    --vocab data_preprocessed/vocab.pkl \
    --out-dir eval_outputs --max-len 25 --device cuda
"""

import argparse
import json
import os
import time
from pathlib import Path
import csv
import pickle
from collections import OrderedDict
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# metrics
import sacrebleu
from rouge_score import rouge_scorer

# optional METEOR (only if nltk and wordnet are present; do NOT attempt download)
USE_METEOR = False
try:
    import nltk
    from nltk.translate.meteor_score import meteor_score  # noqa: F401
    # check wordnet presence
    try:
        nltk.data.find("corpora/wordnet")
        USE_METEOR = True
    except LookupError:
        USE_METEOR = False
except Exception:
    USE_METEOR = False

# Try to import user's model class from common module names
MODEL_MODULE_NAMES = ["model2_vlt", "scripts.model2_vlt", "model2_vlt_emotion", "train_model2_vlt"]

def try_import_model_class():
    for name in MODEL_MODULE_NAMES:
        try:
            mod = __import__(name, fromlist=["VisionLanguageTransformer"])
            if hasattr(mod, "VisionLanguageTransformer"):
                return mod.VisionLanguageTransformer
        except Exception:
            continue
    return None

# -------------------------
# Fallback minimal VLT
# (compatible with earlier training script shapes)
# -------------------------
import math

def sinusoidal_positional_encoding(n_pos: int, d_model: int):
    pe = torch.zeros(n_pos, d_model)
    position = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class FallbackPatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
    def forward(self, x):
        x = self.proj(x)
        B, E, Hn, Wn = x.shape
        return x.flatten(2).transpose(1, 2)

class FallbackVisionEncoder(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, depth, num_heads, dropout):
        super().__init__()
        self.patch = FallbackPatchEmbed(img_size, patch_size, 3, embed_dim)
        P = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(sinusoidal_positional_encoding(P+1, embed_dim), requires_grad=False)
        layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim*4, dropout=dropout)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        B = x.shape[0]
        x = self.patch(x)
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

class FallbackDecoderLayer(nn.Module):
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
    def forward(self, tgt, memory, tgt_mask=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))
        tgt2, _ = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout(tgt2))
        ff = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout(ff))
        return tgt

class FallbackTransformerDecoder(nn.Module):
    def __init__(self, d_model, num_layers, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        layer = FallbackDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.layers = nn.ModuleList([layer] + [FallbackDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers-1)])
    def forward(self, tgt, memory, tgt_mask=None):
        out = tgt
        for layer in self.layers:
            out = layer(out, memory, tgt_mask=tgt_mask)
        return out

class FallbackVLT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = FallbackVisionEncoder(cfg["image_size"], cfg["patch_size"], cfg["vit_embed_dim"],
                                             cfg["vit_depth"], cfg["vit_num_heads"], cfg.get("dropout",0.1))
        self.token_embed = nn.Embedding(cfg["vocab_size"], cfg["decoder_embed_dim"])
        pos_tokens = sinusoidal_positional_encoding(cfg["max_seq_len"], cfg["decoder_embed_dim"])
        self.pos_embed_tokens = nn.Parameter(pos_tokens, requires_grad=False)
        if cfg["vit_embed_dim"] != cfg["decoder_embed_dim"]:
            self.enc_to_dec = nn.Linear(cfg["vit_embed_dim"], cfg["decoder_embed_dim"])
        else:
            self.enc_to_dec = nn.Identity()
        self.emotion_emb = nn.Embedding(cfg.get("num_emotions", 9), cfg["decoder_embed_dim"])
        self.decoder = FallbackTransformerDecoder(cfg["decoder_embed_dim"], cfg["decoder_depth"], cfg["decoder_num_heads"], cfg["decoder_embed_dim"]*4, cfg.get("dropout",0.1))
        self.output_proj = nn.Linear(cfg["decoder_embed_dim"], cfg["vocab_size"])

    def forward(self, images, token_ids, emo_ids=None):
        B, T = token_ids.shape
        enc = self.encoder(images)
        enc = self.enc_to_dec(enc)
        memory = enc.transpose(0,1)
        tok_emb = self.token_embed(token_ids)
        if emo_ids is not None:
            emo_vec = self.emotion_emb(emo_ids).unsqueeze(1)
            # place emotion token at position 0 and shift token embeddings by 1 (training convention)
            dec_in = torch.cat([emo_vec, tok_emb[:, :-1, :]], dim=1)
        else:
            dec_in = tok_emb
        pos = self.pos_embed_tokens[:dec_in.size(1)].unsqueeze(0).to(images.device)
        dec_in = dec_in + pos
        dec_in = dec_in.transpose(0,1)
        tgt_mask = torch.triu(torch.ones(dec_in.size(0), dec_in.size(0), device=images.device), diagonal=1).bool()
        out = self.decoder(dec_in, memory, tgt_mask)
        out = out.transpose(0,1)
        logits = self.output_proj(out)
        return logits

    def greedy_decode(self, image, emo_id, sos_idx, eos_idx=None, max_len=None, device='cpu'):
        self.eval()
        if max_len is None:
            max_len = self.cfg.get("max_seq_len", 20)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            enc = self.encoder(image)
            enc = self.enc_to_dec(enc)
            memory = enc.transpose(0,1)
            cur = torch.LongTensor([[sos_idx]]).to(device)
            generated = []
            for step in range(max_len):
                cur_padded = F.pad(cur, (0, max_len - cur.size(1)), value=0)
                logits = self.forward(image, cur_padded, torch.LongTensor([emo_id]).to(device))
                step_idx = cur.size(1) - 1
                logit_step = logits[:, step_idx, :]
                next_id = torch.argmax(logit_step, dim=-1).item()
                generated.append(next_id)
                if eos_idx is not None and next_id == eos_idx:
                    break
                cur = torch.cat([cur, torch.LongTensor([[next_id]]).to(device)], dim=1)
        return generated

# -------------------------
# Utilities to repair and load checkpoint state dict
# -------------------------
def repair_and_load_state_dict(model: nn.Module, ck_state_dict: Dict):
    """
    Repair common key mismatches and pos-embed length mismatches before loading.
    Returns (ok:bool, info:dict).
    """
    info = {"renamed": [], "trimmed_pos": False, "padded_pos": False, "missing": [], "unexpected": []}
    sd = OrderedDict((k, v.clone()) for k, v in ck_state_dict.items())

    model_keys = list(model.state_dict().keys())
    ck_keys = list(sd.keys())

    # rename patterns: patch_embed <-> patch
    if any("patch_embed" in k for k in ck_keys) and any("encoder.patch." in mk for mk in model_keys):
        new = OrderedDict()
        for k, v in sd.items():
            if "patch_embed" in k:
                k2 = k.replace("patch_embed", "patch")
                new[k2] = v
                info["renamed"].append((k, k2))
            else:
                new[k] = v
        sd = new
    if any("encoder.patch." in k for k in ck_keys) and any("patch_embed" in mk for mk in model_keys):
        new = OrderedDict()
        for k, v in sd.items():
            if "encoder.patch." in k:
                k2 = k.replace("encoder.patch.", "encoder.patch_embed.")
                new[k2] = v
                info["renamed"].append((k, k2))
            else:
                new[k] = v
        sd = new

    # positional embedding mismatch handling
    # find pos keys in ck and model
    ck_pos = None; model_pos = None
    for k in list(sd.keys()):
        if "pos_embed" in k and "token" in k or ("pos_embed" in k and "tokens" in k):
            ck_pos = k
            break
    for mk in model_keys:
        if "pos_embed" in mk and "token" in mk or ("pos_embed" in mk and "tokens" in mk):
            model_pos = mk
            break
    # fallback: any pos_embed
    if ck_pos is None:
        for k in sd.keys():
            if "pos_embed" in k:
                ck_pos = k
                break
    if model_pos is None:
        for mk in model_keys:
            if "pos_embed" in mk:
                model_pos = mk
                break

    if ck_pos and model_pos and ck_pos in sd and model_pos in model.state_dict():
        ck_val = sd[ck_pos]
        model_val = model.state_dict()[model_pos]
        if ck_val.shape != model_val.shape and ck_val.ndim == 2 and model_val.ndim == 2 and ck_val.shape[1] == model_val.shape[1]:
            min_r = min(ck_val.shape[0], model_val.shape[0])
            new_pos = model_val.clone()
            new_pos[:min_r] = ck_val[:min_r]
            sd[model_pos] = new_pos
            if ck_pos != model_pos and ck_pos in sd:
                del sd[ck_pos]
            if ck_val.shape[0] > model_val.shape[0]:
                info["trimmed_pos"] = True
            else:
                info["padded_pos"] = True

    # Prepare filtered state dict with only keys matching and same shape
    model_state = model.state_dict()
    filtered = OrderedDict()
    for k, v in sd.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered[k] = v

    missing = [k for k in model_state.keys() if k not in filtered]
    unexpected = [k for k in sd.keys() if k not in model_state]

    # load with strict=False using filtered keys
    try:
        model.load_state_dict(filtered, strict=False)
        info["missing"] = missing
        info["unexpected"] = unexpected
        return True, info
    except Exception as e:
        # final attempt: load full sd with strict=False
        try:
            model.load_state_dict(sd, strict=False)
            info["missing"] = missing
            info["unexpected"] = unexpected
            return True, info
        except Exception as e2:
            info["error"] = str(e2)
            return False, info

# -------------------------
# Decoding helpers
# -------------------------
def greedy_decode_model(model, img_tensor, emo_id, sos_idx, eos_idx, max_len, device):
    try:
        return model.greedy_decode(img_tensor, emo_id, sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_len, device=device)
    except Exception:
        # fallback sequential greedy
        model.eval()
        with torch.no_grad():
            cur = [sos_idx]
            for step in range(max_len):
                cur_tensor = torch.LongTensor([cur]).to(device)
                emo_tensor = torch.LongTensor([emo_id]).to(device)
                logits = model(img_tensor.to(device), cur_tensor, emo_tensor)
                step_idx = cur_tensor.size(1)-1
                logit_step = logits[:, step_idx, :].squeeze(0)
                next_id = int(torch.argmax(logit_step).cpu().numpy().item())
                cur.append(next_id)
                if eos_idx is not None and next_id == eos_idx:
                    break
            return cur[1:]

def beam_search_decode(model, img_tensor, emo_id, sos_idx, eos_idx, max_len, device, beam_width=3):
    # simple beam - works with forward API
    model.eval()
    with torch.no_grad():
        hyps = [(0.0, [sos_idx])]
        for _ in range(max_len):
            candidates = []
            for score, seq in hyps:
                cur_tensor = torch.LongTensor([seq]).to(device)
                emo_tensor = torch.LongTensor([emo_id]).to(device)
                logits = model(img_tensor.to(device), cur_tensor, emo_tensor)
                step_idx = cur_tensor.size(1)-1
                logit_step = logits[:, step_idx, :].squeeze(0)
                logp = F.log_softmax(logit_step, dim=-1)
                topk = torch.topk(logp, k=min(beam_width, logp.size(0))).indices.cpu().numpy().tolist()
                for nxt in topk:
                    new_score = score + float(logp[nxt].cpu().numpy())
                    new_seq = seq + [int(nxt)]
                    candidates.append((new_score, new_seq))
            candidates.sort(key=lambda x: -x[0])
            hyps = candidates[:beam_width]
            if all((eos_idx is not None and hyp[1][-1] == eos_idx) for hyp in hyps):
                break
        return hyps[0][1][1:]

# -------------------------
# Metrics helpers
# -------------------------
def compute_corpus_bleu(references_list: List[List[str]], hypotheses: List[str]):
    refs_for_sacre = list(zip(*references_list)) if references_list and isinstance(references_list[0], list) else [references_list]
    bleu = sacrebleu.corpus_bleu(hypotheses, refs_for_sacre)
    return bleu

def compute_rouge_l_avg(references_list: List[List[str]], hypotheses: List[str]):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for refs, hyp in zip(references_list, hypotheses):
        best_f = 0.0
        for r in refs:
            sc = scorer.score(r, hyp)
            best_f = max(best_f, sc['rougeL'].fmeasure)
        scores.append(best_f)
    return float(sum(scores) / len(scores)) if scores else 0.0

def compute_meteor_avg(references_list: List[List[str]], hypotheses: List[str]):
    if not USE_METEOR:
        return None
    scores = []
    for refs, hyp in zip(references_list, hypotheses):
        try:
            s = meteor_score(refs, hyp)
        except Exception:
            s = 0.0
        scores.append(s)
    return float(sum(scores) / len(scores)) if scores else 0.0

# -------------------------
# Utilities: load checkpoint & vocab
# -------------------------
def load_checkpoint(path: Path):
    ck = torch.load(str(path), map_location="cpu")
    return ck

def load_vocab(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        tok2idx = pickle.load(f)
    if not isinstance(tok2idx, dict) and hasattr(tok2idx, "token_to_idx"):
        tok2idx = tok2idx.token_to_idx
    return tok2idx

def decode_ids_to_text(ids: List[int], idx2tok: Dict[int,str]):
    toks = []
    for i in ids:
        tok = idx2tok.get(i, str(i))
        if tok in ("<end>", "</s>", "<eos>"):
            break
        if tok in ("<pad>",):
            continue
        toks.append(tok)
    return " ".join(toks)

def build_refs_from_test_csv(test_csv: Path, candidate_cols=("utterance_spelled","utterance","utter_clean")):
    df = pd.read_csv(test_csv)
    refs = {}
    for _, row in df.iterrows():
        pid = row['painting']
        found = None
        for c in candidate_cols:
            if c in df.columns and not pd.isna(row.get(c)):
                found = row.get(c)
                break
        if found is None:
            found = str(row.get("utterance", "")).strip()
        refs.setdefault(pid, []).append(str(found))
    return refs

# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--vocab", required=False)
    p.add_argument("--refs-pkl", required=False, help="optional grouped refs pickle")
    p.add_argument("--out-dir", default="eval_outputs")
    p.add_argument("--max-len", type=int, default=20)
    p.add_argument("--device", default=None)
    p.add_argument("--beam", action="store_true", help="use small beam search fallback (slower)")
    p.add_argument("--beam-width", type=int, default=3)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if user asked for cuda but torch isn't compiled with it, this will still choose cpu in practice when mapping tensors
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print("Loading checkpoint:", ckpt_path)
    ck = load_checkpoint(ckpt_path)

    # try to instantiate user model
    ModelClass = try_import_model_class()
    cfg = ck.get("config", None)
    model = None
    if ModelClass is not None and cfg is not None:
        try:
            print("Attempting to instantiate user's VisionLanguageTransformer with checkpoint config...")
            model = ModelClass(cfg)
            # do not call model.to(device) before load_state_dict
            # try to load state dict; let repair helper fix mismatches
            ok, info = repair_and_load_state_dict(model, ck["model_state_dict"])
            print("Repair/load info:", info)
            if not ok:
                print("Warning: failed to fully load into user model, will fallback to FallbackVLT.")
                model = None
            else:
                model.to(device)
                model.eval()
        except Exception as e:
            print("Could not instantiate user model class due to:", e)
            model = None

    if model is None:
        if cfg is None:
            raise RuntimeError("Checkpoint missing 'config' and user model could not be instantiated. Provide checkpoint with 'config' or place model2_vlt.py in PATH.")
        print("Using fallback VLT with config from checkpoint.")
        model = FallbackVLT(cfg)
        ok, info = repair_and_load_state_dict(model, ck["model_state_dict"])
        print("Repair/load info:", info)
        if not ok:
            raise RuntimeError("Failed to load checkpoint into fallback model after attempted repairs. Info: " + json.dumps(info))
        model.to(device)
        model.eval()

    # load vocab mapping if available
    tok2idx = None
    if args.vocab:
        tp = Path(args.vocab)
        if tp.exists():
            tok2idx = load_vocab(tp)
    # fallback to checkpoint
    if tok2idx is None:
        for key in ("token_to_idx", "tok2idx", "tokenizer"):
            if key in ck:
                tok2idx = ck[key]
                break
    idx2tok = {i: t for t, i in tok2idx.items()} if tok2idx else None

    # find sos/eos indices
    sos_idx = None; eos_idx = None
    if idx2tok:
        for i, t in idx2tok.items():
            if t in ("<start>", "<s>", "[START]", "<bos>"):
                sos_idx = i; break
        for i, t in idx2tok.items():
            if t in ("<end>", "</s>", "<eos>", "</end>"):
                eos_idx = i; break
    if sos_idx is None:
        sos_idx = 1

    # load refs (prefer provided pickle)
    refs = {}
    if args.refs_pkl and Path(args.refs_pkl).exists():
        try:
            groups = pickle.load(open(args.refs_pkl, "rb"))
            # groups could be dict with 'test' DataFrame
            if isinstance(groups, dict) and "test" in groups:
                gdf = groups["test"]
                try:
                    for _, r in gdf.iterrows():
                        pid = r['painting']
                        if 'references' in r and isinstance(r['references'], (list, tuple)):
                            refs[pid] = r['references']
                        elif 'references_pre_vocab' in r and isinstance(r['references_pre_vocab'], (list, tuple)):
                            refs[pid] = r['references_pre_vocab']
                        else:
                            refs.setdefault(pid, []).extend(r.get('references', []) or r.get('references_pre_vocab', []) or [])
                except Exception:
                    pass
        except Exception as e:
            print("Could not load refs_pkl:", e)

    if not refs:
        print("Building refs from test CSV")
        refs = build_refs_from_test_csv(Path(args.test_csv))

    test_df = pd.read_csv(args.test_csv)
    images_root = Path(args.images_root)

    outputs = []
    start_time = time.time()

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        pid = row['painting']
        emo = int(row['emotion_label']) if 'emotion_label' in row else 0

        # load image npy or jpg
        img_npy = images_root / f"{pid}.npy"
        if img_npy.exists():
            arr = np.load(img_npy)
            if arr.ndim == 1:
                # precomputed features: fallback to zeros image to keep API consistent
                img_tensor = torch.zeros(1, 3, cfg.get("image_size",224), cfg.get("image_size",224)).to(device)
            else:
                img_tensor = torch.tensor(arr).permute(2,0,1).unsqueeze(0).float().to(device)
        else:
            jpg = images_root / f"{pid}.jpg"
            if jpg.exists():
                from PIL import Image
                im = Image.open(jpg).convert("RGB").resize((cfg.get("image_size",224), cfg.get("image_size",224)))
                arr = np.array(im).astype("float32")/255.0
                img_tensor = torch.tensor(arr).permute(2,0,1).unsqueeze(0).float().to(device)
            else:
                img_tensor = torch.zeros(1,3,cfg.get("image_size",224),cfg.get("image_size",224)).to(device)

        if args.beam:
            try:
                ids = beam_search_decode(model, img_tensor, emo, sos_idx, eos_idx, args.max_len, device, beam_width=args.beam_width)
            except Exception:
                ids = greedy_decode_model(model, img_tensor, emo, sos_idx, eos_idx, args.max_len, device)
        else:
            ids = greedy_decode_model(model, img_tensor, emo, sos_idx, eos_idx, args.max_len, device)

        pred_text = decode_ids_to_text(ids, idx2tok) if idx2tok else " ".join(map(str, ids))
        ref_texts = refs.get(pid, [""])
        outputs.append({"painting": pid, "emotion_label": emo, "pred_ids": ids, "pred_text": pred_text, "references": ref_texts})

    elapsed = time.time() - start_time

    # save predictions csv
    out_csv = Path(args.out_dir) / "predictions.csv"
    with open(out_csv, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=["painting","emotion_label","pred_ids","pred_text","references"])
        writer.writeheader()
        for r in outputs:
            writer.writerow({
                "painting": r["painting"],
                "emotion_label": r["emotion_label"],
                "pred_ids": json.dumps(r["pred_ids"]),
                "pred_text": r["pred_text"],
                "references": json.dumps(r["references"])
            })

    # metrics
    hyps = [r["pred_text"] for r in outputs]
    refs_list = [r["references"] for r in outputs]

    print("Computing BLEU...")
    bleu = compute_corpus_bleu(refs_list, hyps)
    print("Computing ROUGE-L...")
    rouge_l = compute_rouge_l_avg(refs_list, hyps)
    meteor_avg = compute_meteor_avg(refs_list, hyps) if USE_METEOR else None

    summary = {
        "ckpt": str(ckpt_path),
        "n_samples": len(outputs),
        "elapsed_s": elapsed,
        "bleu_score": float(bleu.score),
        "bleu_verbose": str(bleu),
        "rouge_l_avg_f": rouge_l,
        "meteor_avg": meteor_avg
    }
    summary_path = Path(args.out_dir) / "eval_summary.json"
    with open(summary_path, "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EVAL SUMMARY ===")
    print(f"Samples: {len(outputs)}")
    print(f"BLEU (sacrebleu): {bleu.score:.4f}")
    print(f"ROUGE-L (avg best-ref F1): {rouge_l:.4f}")
    if meteor_avg is not None:
        print(f"METEOR (avg): {meteor_avg:.4f}")
    else:
        print("METEOR skipped (NLTK/WordNet not available).")
    print(f"Time elapsed (s): {elapsed:.1f}")
    print("====================")
    print(f"Predictions CSV: {out_csv}")
    print(f"Summary JSON: {summary_path}")

if __name__ == "__main__":
    main()
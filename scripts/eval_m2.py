#!/usr/bin/env python3
"""
scripts/eval_m2.py

No-CLI evaluator for VisionLanguageTransformer.

- Loads per-embedding best checkpoints (m2_best_{emb}.pt)
- Generates greedy captions on val set
- Computes BLEU (nltk), ROUGE-L (rouge_score), CIDEr (pycocoevalcap) if available
- Writes:
    eval_outputs/m2_eval_all.json   <- unified metrics for all embeddings
    eval_outputs/m2_{emb}_samples.csv  <- generated samples per embedding
"""

import json
import math
import csv
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# metric libs (optional)
try:
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    nltk_available = True
except Exception:
    nltk_available = False

try:
    from rouge_score import rouge_scorer
    rouge_available = True
except Exception:
    rouge_available = False

try:
    # pycocoevalcap may be installed; try to import CIDEr
    from pycocoevalcap.cider.cider import Cider
    cider_available = True
except Exception:
    cider_available = False

# import project model/dataset
from model2_vlt import VisionLanguageTransformer, CaptionDataset, DEFAULT_CONFIG

# -----------------------
# Helpers
# -----------------------
def load_vocab(path):
    p = Path(path)
    if not p.exists():
        return None, None
    import pickle
    v = pickle.load(open(p, "rb"))
    if not isinstance(v, dict) and hasattr(v, "token_to_idx"):
        tok2idx = v.token_to_idx
    else:
        tok2idx = v
    idx2tok = {i: t for t, i in tok2idx.items()}
    return tok2idx, idx2tok

def tokens_to_text(tokens, idx2tok):
    if idx2tok is None:
        return " ".join(str(t) for t in tokens if t != 0)
    words = []
    for t in tokens:
        if t == 0:
            continue
        words.append(idx2tok.get(int(t), "<unk>"))
    return " ".join(words)

def greedy_generate_batch(model, imgs, emos, tfidf_vecs, device, sos_idx=1, eos_idx=None, max_len=None):
    # Generates lists of token ids per image (greedy) using model.greedy_decode for each image
    outs = []
    model.eval()
    with torch.no_grad():
        B = imgs.size(0)
        for i in range(B):
            img = imgs[i].to(device)
            emo = int(emos[i].item()) if isinstance(emos[i], torch.Tensor) else int(emos[i])
            if model.embedding_type == "tfidf":
                tfv = tfidf_vecs[i].unsqueeze(0).to(device) if tfidf_vecs is not None else None
                gen = model.greedy_decode(img, emo, sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_len, device=device, tfidf_vec=tfv)
            else:
                gen = model.greedy_decode(img, emo, sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_len, device=device)
            outs.append(gen)
    return outs

# -----------------------
# Metric wrappers
# -----------------------
def compute_bleu_corpus(references, hypotheses):
    # references: list of list of reference token lists (tokenized words)
    # hypotheses: list of hypothesis token lists (tokenized words)
    if not nltk_available:
        return None
    # nltk corpus_bleu expects list of list of references (each reference is list of tokens)
    # we use SmoothingFunction method1 to avoid zero scores.
    ch = SmoothingFunction()
    # Use cumulative 4-gram BLEU
    try:
        score = corpus_bleu(references, hypotheses, smoothing_function=ch.method1)
    except Exception:
        # fallback: micro-average with uniform weights
        score = 0.0
    return float(score)

def compute_rouge_l(references_texts, hypotheses_texts):
    if not rouge_available:
        return None
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for ref, hyp in zip(references_texts, hypotheses_texts):
        sc = scorer.score(ref, hyp)
        scores.append(sc['rougeL'].fmeasure)
    return float(sum(scores) / max(1, len(scores)))

def compute_cider(references_texts, hypotheses_texts):
    if not cider_available:
        return None
    # pycocoevalcap Cider expects dicts mapping id->list of references / candidates
    refs_dict = {}
    hyps_dict = {}
    for i, (r, h) in enumerate(zip(references_texts, hypotheses_texts)):
        # references must be list
        refs_dict[i] = [r]
        hyps_dict[i] = h
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(refs_dict, hyps_dict)
    # cider returns (score, scores_list)
    return float(score)

# -----------------------
# Main (no CLI)
# -----------------------
def main():
    print("eval_m2.py starting...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    tok2idx, idx2tok = load_vocab(DEFAULT_CONFIG["vocab_path"])
    if idx2tok is None:
        print("Warning: vocab not loaded; generated text will be token ids.")

    embedding_types = ["random", "glove", "fasttext", "tfidf"]
    ckpt_dir = Path(DEFAULT_CONFIG["checkpoint_root"]) / "model2"
    out_root = Path("eval_outputs")
    out_root.mkdir(exist_ok=True)

    unified = {}

    # detect tfidf dir if present
    tfidf_dir = None
    tfidf_dim = None
    cand = Path("data_preprocessed/tfidf_npy")
    if cand.exists() and any(cand.glob("*.npy")):
        tfidf_dir = cand
        sample = next(cand.glob("*.npy"))
        tfidf_dim = int(np.load(sample).shape[-1])
        print("Detected tfidf dir:", tfidf_dir, "dim=", tfidf_dim)

    val_csv = Path(DEFAULT_CONFIG["val_csv"])
    if not val_csv.exists():
        raise RuntimeError("Val CSV not found: " + str(val_csv))

    for emb in embedding_types:
        ckpt_path = ckpt_dir / f"m2_best_{emb}.pt"
        if not ckpt_path.exists():
            print(f"[{emb}] best ckpt not found: {ckpt_path} — skipping.")
            continue
        print(f"\n=== Evaluating embedding: {emb} ===")
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        cfg = ckpt.get("config", DEFAULT_CONFIG)

        # build dataset & loader
        ds = CaptionDataset(str(val_csv), cfg["images_features_root"],
                            max_len=cfg["max_seq_len"], embedding_type=emb,
                            tfidf_dir=(str(tfidf_dir) if (emb == "tfidf" and tfidf_dir is not None) else None))
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

        # build model and load state
        model = VisionLanguageTransformer(cfg,
                                         pretrained_token_emb_weights=None,
                                         embedding_type=emb,
                                         tfidf_dim=(tfidf_dim if emb == "tfidf" else None))
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # iterate, compute loss and collect references + hyps for metrics
        model.eval()
        total_loss = 0.0
        n_examples = 0
        total_correct = 0
        total_tokens = 0

        references_for_bleu = []   # list of list of token lists (nltk format)
        hypotheses_for_bleu = []   # list of token lists
        references_texts = []
        hypotheses_texts = []
        generated_rows = []  # for CSV: [ref_tokens, ref_text, gen_tokens, gen_text]

        start = time.time()
        for batch in loader:
            if emb == "tfidf":
                imgs, token_ids, emos, tfidf_vecs = batch
                tfidf_vecs = tfidf_vecs.to(device)
            else:
                imgs, token_ids, emos = batch
                tfidf_vecs = None

            imgs = imgs.to(device)
            token_ids = token_ids.to(device)
            emos = emos.to(device)

            # forward for loss
            if emb == "tfidf":
                logits = model(imgs, token_ids, emos, tfidf_vecs)
            else:
                logits = model(imgs, token_ids, emos)

            logits_in = logits[:, :-1]  # (B, T-1, V)
            targets = token_ids[:, 1:].contiguous()
            B, Tm, V = logits_in.shape
            loss = criterion(logits_in.reshape(B * Tm, V), targets.reshape(B * Tm))
            total_loss += loss.item() * B
            n_examples += B

            # token accuracy
            pred = logits_in.argmax(dim=-1)
            mask = (targets != 0)
            total_correct += ((pred == targets) & mask).sum().item()
            total_tokens += mask.sum().item()

            # generate greedy per image (uses model.greedy_decode which is per-image)
            gen_ids = greedy_generate_batch(model, imgs, emos, tfidf_vecs if emb == "tfidf" else None,
                                           device=device, sos_idx=1, eos_idx=None, max_len=cfg["max_seq_len"])

            for i in range(len(gen_ids)):
                ref_ids = token_ids[i].tolist()
                hyp_ids = gen_ids[i]
                ref_text = tokens_to_text(ref_ids, idx2tok)
                hyp_text = tokens_to_text(hyp_ids, idx2tok)
                # prepare BLEU format: references are list of tokenized strings
                ref_tokens = [r for r in ref_text.split()] if ref_text.strip() else []
                hyp_tokens = [h for h in hyp_text.split()] if hyp_text.strip() else []
                if len(ref_tokens) == 0:
                    # fall back to token ids as strings if vocab missing
                    ref_tokens = [str(x) for x in ref_ids if x != 0]
                if len(hyp_tokens) == 0:
                    hyp_tokens = [str(x) for x in hyp_ids if x != 0]

                references_for_bleu.append([ref_tokens])
                hypotheses_for_bleu.append(hyp_tokens)
                references_texts.append(ref_text)
                hypotheses_texts.append(hyp_text)
                generated_rows.append((ref_ids, ref_text, hyp_ids, hyp_text))

        elapsed = time.time() - start
        avg_loss = total_loss / max(1, n_examples)
        token_acc = total_correct / max(1, total_tokens) if total_tokens > 0 else 0.0
        ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")

        print(f"[{emb}] examples={n_examples} loss={avg_loss:.4f} ppl={ppl:.2f} token_acc={token_acc:.4f} time={elapsed:.1f}s")

        # compute metrics
        bleu = compute_bleu_corpus(references_for_bleu, hypotheses_for_bleu) if nltk_available else None
        rouge_l = compute_rouge_l(references_texts, hypotheses_texts) if rouge_available else None
        cider = compute_cider(references_texts, hypotheses_texts) if cider_available else None

        print("Metrics:", {"BLEU": bleu, "ROUGE-L": rouge_l, "CIDEr": cider})

        # write samples CSV for this embedding
        csv_path = out_root / f"m2_{emb}_samples.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf8") as f:
            w = csv.writer(f)
            w.writerow(["ref_tokens", "ref_text", "gen_tokens", "gen_text"])
            for r in generated_rows:
                w.writerow([json.dumps(r[0]), r[1], json.dumps(r[2]), r[3]])
        print("Saved samples ->", csv_path)

        # add to unified dict
        unified[emb] = {
            "checkpoint": str(ckpt_path),
            "n_examples": int(n_examples),
            "loss": float(avg_loss),
            "perplexity": float(ppl),
            "token_accuracy": float(token_acc),
            "BLEU": float(bleu) if bleu is not None else None,
            "ROUGE-L": float(rouge_l) if rouge_l is not None else None,
            "CIDEr": float(cider) if cider is not None else None,
            "samples_csv": str(csv_path)
        }

    # save unified JSON
    unified_path = out_root / "m2_eval_all.json"
    json.dump(unified, open(unified_path, "w"), indent=2)
    print("Saved unified metrics ->", unified_path)
    print("Done.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
eval_m1.py

Evaluate CNN Encoder + LSTM Decoder (Model 1) for different embedding types:
- random
- glove
- fasttext
- tfidf

Metrics:
- BLEU-1, BLEU-4
- ROUGE-1, ROUGE-L
- METEOR (optional, skipped if NLTK/resources missing)
- Qualitative examples (text output)

Assumptions:
- Checkpoints are stored as:
    eval_outputs/results_cnn_lstm/best_model_<emb_type>.pth
- Preprocessing produced:
    new_preprocessed/artemis_preprocessed.csv  (with 'split' column)
    new_preprocessed/features/<painting>.npy
    new_preprocessed/vocab.pkl
- SPECIAL_TOKENS order: ["<pad>", "<start>", "<end>", "<unk>"]
"""

import os
import os.path as osp
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- NLP metrics deps (optional for METEOR) ----------------
try:
    # Ensure NLTK packages are downloaded if running this for the first time
    # import nltk; nltk.download(['punkt', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger'])
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

# ----------------------- Model 1 Definition (CNN Encoder + LSTM Decoder) -----------------------

# --- CustomCNNEncoder (from M1.py) ---
class CustomCNNEncoder(nn.Module):
    def __init__(self, output_dim, input_size=128):
        super().__init__()
        
        final_spatial_dim = input_size // (2**4) 
        final_cnn_channels = 512
        OBSERVED_FLATTENED_SIZE = final_cnn_channels * final_spatial_dim * final_spatial_dim 
        
        self.cnn_blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(OBSERVED_FLATTENED_SIZE, 1024), 
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(1024, output_dim) 
        )

    def forward(self, x):
        # Permute from (B, H, W, 3) to (B, 3, H, W) 
        if x.dim() == 4 and x.shape[-1] == 3:
              x = x.permute(0, 3, 1, 2) 
        x = self.cnn_blocks(x)
        x = self.fc(x)
        return x # (B, output_dim)

# --- EmotionLSTMDecoder (from M1.py) ---
class EmotionLSTMDecoder(nn.Module):
    def __init__(self, cfg: Dict[str, Any], embedding_matrix=None):
        super().__init__()
        
        vocab_size = cfg["vocab_size"]
        embed_dim = cfg["embedding_dim"]
        hidden_size = cfg["hidden_size"]
        num_emotions = cfg["num_emotions"]
        image_feature_dim = cfg["image_feature_dim"]
        dropout_rate = cfg["dropout_rate"]
        
        self.hidden_size = hidden_size
        self.image_feature_dim = image_feature_dim 
        self.pad_idx = cfg.get("pad_idx", 0)

        # 1. Embeddings
        if embedding_matrix is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float), freeze=False
            )
            embed_dim = embedding_matrix.shape[1] 
        else:
            # Placeholder for loading from state dict later
            self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        self.emotion_embeddings = nn.Embedding(num_emotions, embed_dim) 

        self.lstm_input_size = embed_dim 

        # 2. Initial State Generators (h0 and c0)
        init_gen_input_dim = image_feature_dim + embed_dim 
        
        self.h0_generator = nn.Sequential(
            nn.Linear(init_gen_input_dim, hidden_size), 
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.c0_generator = nn.Sequential(
            nn.Linear(init_gen_input_dim, hidden_size), 
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 3. LSTM
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1, 
            dropout=dropout_rate if dropout_rate > 0 else 0.0
        )

        # 4. Output Layer
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.dropout_word = nn.Dropout(dropout_rate)

    def init_hidden_state(self, image_features, emotion_labels):
        """Generates initial h0 and c0 using image features and emotion embedding."""
        emotion_vec = self.emotion_embeddings(emotion_labels) 
        combined_context = torch.cat((image_features, emotion_vec), dim=1) 
        h0 = self.h0_generator(combined_context).unsqueeze(0) 
        c0 = self.c0_generator(combined_context).unsqueeze(0) 
        return h0, c0, emotion_vec

    def forward(self, image_features, emotion_labels, caption_input):
        
        h0, c0, _ = self.init_hidden_state(image_features, emotion_labels) 
        word_embeds = self.word_embeddings(caption_input) 
        word_embeds = self.dropout_word(word_embeds)
        
        lstm_out, _ = self.lstm(word_embeds, (h0, c0)) 

        output = self.output_layer(lstm_out)
        
        return output

    def greedy_decode(
        self,
        encoder: CustomCNNEncoder,
        image: torch.Tensor,
        emo_id: int,
        sos_idx: int,
        eos_idx: int,
        max_len: int,
        device: torch.device,
    ) -> List[int]:
        self.eval()
        encoder.eval()
        
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(device)
        emo = torch.tensor([emo_id], dtype=torch.long, device=device)

        with torch.no_grad():
            image_features = encoder(image)
            
            # Get initial hidden state
            h, c, _ = self.init_hidden_state(image_features, emo)
            
            # Start with <start> token
            current_word_id = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
            generated: List[int] = []

            for _ in range(max_len):
                word_embed = self.word_embeddings(current_word_id) # (1, 1, D)
                
                # Forward one step through the LSTM
                lstm_out, (h, c) = self.lstm(word_embed, (h, c)) # (1, 1, H)
                
                # Output projection
                logits = self.output_layer(lstm_out.squeeze(1)) # (1, V)
                
                # Greedy search: get next token
                next_id = torch.argmax(logits, dim=-1).item()
                generated.append(next_id)
                
                if next_id == eos_idx:
                    break
                
                # Prepare next input
                current_word_id = torch.tensor([[next_id]], dtype=torch.long, device=device)

        return generated


# ----------------------- Helper functions (same as your template) -----------------------

def load_vocab(vocab_path: str):
    import pickle
    with open(vocab_path, "rb") as f:
        tok2idx = pickle.load(f)
    if not isinstance(tok2idx, dict) and hasattr(tok2idx, "token_to_idx"):
        tok2idx = tok2idx.token_to_idx
    idx2tok = {i: t for t, i in tok2idx.items()}
    pad_idx = tok2idx.get("<pad>", 0)
    sos_idx = tok2idx.get("<start>", 1)
    eos_idx = tok2idx.get("<end>", 2)
    return tok2idx, idx2tok, pad_idx, sos_idx, eos_idx


def ids_to_tokens(ids: List[int], idx2tok: Dict[int, str],
                  eos_idx: int, pad_idx: int) -> List[str]:
    toks = []
    for i in ids:
        if i == eos_idx or i == pad_idx:
            break
        toks.append(idx2tok.get(i, "<unk>"))
    return toks


def simple_rouge_1(hyp: List[str], ref: List[str]) -> Tuple[float, float, float]:
    """ROUGE-1 (unigram) precision, recall, F1."""
    # Simplified implementation for ROUGE, as provided in the template
    hyp_set = hyp
    ref_set = ref
    ref_counts = {}
    for w in ref_set:
        ref_counts[w] = ref_counts.get(w, 0) + 1
    hyp_counts = {}
    for w in hyp_set:
        hyp_counts[w] = hyp_counts.get(w, 0) + 1
    overlap = 0
    for w, c in hyp_counts.items():
        overlap += min(c, ref_counts.get(w, 0))
    if overlap == 0:
        return 0.0, 0.0, 0.0
    prec = overlap / max(1, len(hyp_set))
    rec = overlap / max(1, len(ref_set))
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def lcs_length(x: List[str], y: List[str]) -> int:
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]


def simple_rouge_l(hyp: List[str], ref: List[str]) -> Tuple[float, float, float]:
    lcs = lcs_length(hyp, ref)
    if lcs == 0:
        return 0.0, 0.0, 0.0
    prec = lcs / max(1, len(hyp))
    rec = lcs / max(1, len(ref))
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# ----------------------- Evaluation core -----------------------

def evaluate_for_embedding(
    emb_type: str,
    args,
    tok2idx,
    idx2tok,
    pad_idx,
    sos_idx,
    eos_idx,
) -> Dict[str, Any]:
    device = torch.device(args.device)
    
    # Checkpoint path for Model 1 (CNN-LSTM)
    ckpt_dir = Path(args.checkpoints_root)
    ckpt_path = ckpt_dir / f"best_model_{emb_type}.pth" # Match M1.py saving scheme
    
    if not ckpt_path.exists():
        print(f"[{emb_type}] Best checkpoint not found at {ckpt_path}, skipping.")
        return {}

    print(f"\n========== Evaluating embedding type: {emb_type} ==========")
    print("Loading checkpoint:", ckpt_path)
    ck = torch.load(str(ckpt_path), map_location=device)

    # ----- reconstruct cfg from checkpoint -----
    if "config" not in ck:
        print("FATAL ERROR: Checkpoint is missing 'config' key. Cannot reconstruct model.")
        return {}
        
    cfg = ck["config"]
    print("Config loaded from checkpoint.")

    # ----- build model & load weights -----
    
    # The config snapshot from training needs a few additions for consistency
    cfg["vocab_size"] = len(tok2idx)
    cfg["pad_idx"] = pad_idx
    
    # Initialize Models
    encoder = CustomCNNEncoder(output_dim=cfg["IMAGE_FEATURE_DIM"], 
                               input_size=cfg.get("IMAGE_SIZE", 128)).to(device)
    
    # Embedding matrix is passed as None, state_dict loads the weights (including embedding weights)
    decoder = EmotionLSTMDecoder(cfg, embedding_matrix=None).to(device) 
    
    try:
        encoder.load_state_dict(ck["encoder_state_dict"], strict=True)
        decoder.load_state_dict(ck["decoder_state_dict"], strict=True)
    except RuntimeError as e:
        print(f"ERROR loading state dict: {e}")
        return {}

    encoder.eval()
    decoder.eval()

    # ----- load data (test split only) -----
    df = pd.read_csv(args.csv)
    df = df[df["split"] == args.split].reset_index(drop=True)
    print(f"[{emb_type}] Test rows: {len(df)}")

    references: List[List[str]] = []
    hypotheses: List[List[str]] = []
    meteor_refs: List[str] = []
    meteor_hyps: List[str] = []

    if NLTK_AVAILABLE:
        smoothie = SmoothingFunction().method4
    else:
        smoothie = None

    for _, row in df.iterrows():
        painting = row["painting"]
        emo = int(row["emotion_label"])

        feat_path = Path(args.features_root) / f"{painting}.npy"
        if not feat_path.exists():
            continue

        arr = np.load(feat_path)
        # Image feature array is HxWx3, need to convert to tensor
        if arr.ndim == 3:
            img = torch.tensor(arr).float() 
        else:
            img = torch.tensor(arr).float()

        # reference tokens (assuming 'utter_clean' or 'tokens_str' is the source)
        if isinstance(row.get("tokens_str", None), str):
            ref_tokens = row["tokens_str"].split()
        else:
            # Fallback to utter_clean if tokens_str is missing or not a string
            ref_tokens = str(row["utter_clean"]).split()

        # generate using the LSTM's greedy_decode method
        gen_ids = decoder.greedy_decode(
            encoder=encoder,
            image=img,
            emo_id=emo,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            max_len=args.max_gen_len,
            device=device,
        )
        hyp_tokens = ids_to_tokens(gen_ids, idx2tok, eos_idx, pad_idx)
        if len(hyp_tokens) == 0:
            continue

        references.append([ref_tokens]) # BLEU requires list of references
        hypotheses.append(hyp_tokens)
        
        # Prepare for ROUGE/METEOR (which usually take strings or single reference lists)
        ref_str = " ".join(ref_tokens)
        hyp_str = " ".join(hyp_tokens)

        if NLTK_AVAILABLE:
            meteor_refs.append(ref_str)
            meteor_hyps.append(hyp_str)

    print(f"[{emb_type}] Collected {len(hypotheses)} hypothesis–reference pairs.")

    # BLEU
    if references and hypotheses:
        # Note: corpus_bleu expects references as a list of lists of tokens: [[[r1_t1, r1_t2...]], [[r2_t1...]]]
        bleu1 = corpus_bleu(
            references, hypotheses,
            weights=(1.0, 0.0, 0.0, 0.0),
            smoothing_function=smoothie,
        )
        bleu4 = corpus_bleu(
            references, hypotheses,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie,
        )
    else:
        bleu1 = bleu4 = 0.0

    # ROUGE-1 / ROUGE-L (Using the simple average approach from your template)
    rouge1_f_list, rougeL_f_list = [], []
    # ROUGE takes single reference lists (tokens)
    for hyp, ref in zip(hypotheses, [r[0] for r in references]): 
        _, _, f1_1 = simple_rouge_1(hyp, ref)
        _, _, f1_L = simple_rouge_l(hyp, ref)
        rouge1_f_list.append(f1_1)
        rougeL_f_list.append(f1_L)

    rouge1_f = float(np.mean(rouge1_f_list)) if rouge1_f_list else 0.0
    rougeL_f = float(np.mean(rougeL_f_list)) if rougeL_f_list else 0.0

    # METEOR (optional)
    meteor_avg = None
    if NLTK_AVAILABLE and meteor_refs:
        try:
            # meteor_score takes a list of reference strings (or single string) and a hypothesis string
            scores = [meteor_score([r], h) for r, h in zip(meteor_refs, meteor_hyps)]
            meteor_avg = float(np.mean(scores))
        except Exception as e:
            print(f"Warning: METEOR calculation failed: {e}")
            meteor_avg = None

    # qualitative examples
    print(f"\n[{emb_type}] Sample qualitative generations:")
    num_show = min(args.num_examples, len(hypotheses))
    for i in range(num_show):
        # references are stored as [[ref_tokens]] for BLEU, need to extract first ref
        ref = " ".join(references[i][0])
        hyp = " ".join(hypotheses[i])
        print(f"  Example {i+1}:")
        print(f"      REF: {ref}")
        print(f"      HYP: {hyp}")

    results = {
        "embedding_type": emb_type,
        "bleu1": float(bleu1),
        "bleu4": float(bleu4),
        "rouge1_f": rouge1_f,
        "rougeL_f": rougeL_f,
        "meteor": meteor_avg,
        "num_pairs": len(hypotheses),
    }
    print(f"\n[{emb_type}] metrics:")
    print(json.dumps(results, indent=2))
    return results


# ----------------------- Main -----------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CNN-LSTM (Model 1) for multiple embedding types.")
    p.add_argument("--checkpoints-root", type=str, default="eval_outputs/results_cnn_lstm",
                   help="Root folder containing best_model_<emb_type>.pth")
    p.add_argument("--embedding-types", type=str, nargs="+",
                   default=["random", "glove", "fasttext", "tfidf"]) # Added tfidf
    p.add_argument("--csv", type=str, default="new_preprocessed/artemis_preprocessed.csv")
    p.add_argument("--features-root", type=str, default="new_preprocessed/features")
    p.add_argument("--vocab", type=str, default="new_preprocessed/vocab.pkl")
    p.add_argument("--split", type=str, default="test", help="Data split to evaluate (e.g., 'test' or 'val')")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_gen_len", type=int, default=25)
    p.add_argument("--num_examples", type=int, default=5)
    p.add_argument("--out-json", type=str, default="eval_outputs/m1_eval_summary.json")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(osp.dirname(args.out_json), exist_ok=True)

    # 1. Load Vocab
    tok2idx, idx2tok, pad_idx, sos_idx, eos_idx = load_vocab(args.vocab)
    print("Vocab size:", len(tok2idx))
    print("PAD idx:", pad_idx, "SOS idx:", sos_idx, "EOS idx:", eos_idx)

    # 2. Run Evaluation for each type
    all_results = []
    for emb in args.embedding_types:
        res = evaluate_for_embedding(
            emb, args, tok2idx, idx2tok, pad_idx, sos_idx, eos_idx
        )
        if res:
            all_results.append(res)

    # 3. Save Summary
    summary = {
        "results": all_results,
        "config": vars(args) # Save runtime arguments for reference
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved evaluation summary to:", args.out_json)


if __name__ == "__main__":
    main()
import os
import os.path as osp
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from scripts.M1_CNN import (
    CustomCNNEncoder,
    EmotionLSTMDecoder,
    ArtemisCaptioningDataset,
    BASE_CONFIG as CONFIG,
    device,
)

from embedding_utils import get_embedding_matrix


def load_vocab(path):
    with open(path, "rb") as f:
        vocab = pickle.load(f)

    if hasattr(vocab, "token_to_idx"):
        tok2idx = vocab.token_to_idx
    else:
        tok2idx = vocab

    idx2tok = {i: t for t, i in tok2idx.items()}
    return tok2idx, idx2tok


# we’ll fill these from vocab inside evaluate_all_embeddings()
PAD_TOKEN_ID = 0
START_TOKEN_ID = 1
END_TOKEN_ID = 2

EMOTION_WORDS = [
    "amusement", "contentment", "awe", "excitement", "fear",
    "anger", "sadness", "disgust", "something", "something else"
]


def clean_caption(text: str) -> str:
    """Remove asterisks and leading emotion word (e.g., 'something else')."""
    if text is None:
        return ""

    text = text.replace("*", "").strip()
    words = text.split()

    # dropped if first word is an emotion word
    if len(words) > 0 and words[0] in EMOTION_WORDS:
        words = words[1:]

    return " ".join(words).strip()

def greedy_decode(encoder, decoder, image, emo_id, idx2tok, max_len):
    """
    Greedy decoding for one image-emotion pair.
    Uses global START_TOKEN_ID / END_TOKEN_ID / PAD_TOKEN_ID,
    and removes leading emotion-word from the decoded sentence.
    """
    encoder.eval()
    decoder.eval()

    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    emo = torch.tensor([emo_id], dtype=torch.long, device=device)

    # initial LSTM state from encoder + emotion
    img_feat = encoder(image)
    h, c = decoder.init_hidden_state(img_feat, emo)

    cur = torch.tensor([[START_TOKEN_ID]], dtype=torch.long, device=device)
    result_ids = []

    for _ in range(max_len):
        # embed last token
        emb = decoder.word_embeddings(cur[:, -1:])
        emb = decoder.dropout_word(emb)

        # single LSTM step
        output, (h, c) = decoder.lstm(emb, (h, c))
        logits = decoder.output_layer(output[:, -1, :])
        next_id = torch.argmax(logits, dim=-1).item()

        # stop on EOS or PAD
        if next_id == END_TOKEN_ID or next_id == PAD_TOKEN_ID:
            break

        # skip emotion-word as *first* generated token
        if len(result_ids) == 0 and idx2tok.get(next_id, "") in EMOTION_WORDS:
            continue

        result_ids.append(next_id)
        cur = torch.cat(
            [cur, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )

    caption_raw = " ".join(idx2tok.get(i, "<unk>") for i in result_ids)
    return clean_caption(caption_raw)


def rouge_1_f(pred: str, ref: str) -> float:
    pred_tokens = pred.split()
    ref_tokens = ref.split()

    ref_counts = {}
    for w in ref_tokens:
        ref_counts[w] = ref_counts.get(w, 0) + 1

    pred_counts = {}
    for w in pred_tokens:
        pred_counts[w] = pred_counts.get(w, 0) + 1

    overlap = sum(min(pred_counts.get(w, 0), ref_counts.get(w, 0))
                  for w in pred_counts)

    if overlap == 0:
        return 0.0

    prec = overlap / len(pred_tokens) if pred_tokens else 0.0
    rec = overlap / len(ref_tokens) if ref_tokens else 0.0
    if prec + rec == 0:
        return 0.0

    return 2 * prec * rec / (prec + rec)


def lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            dp[i+1][j+1] = dp[i][j]+1 if a[i] == b[j] else max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]


def rouge_l_f(pred: str, ref: str) -> float:
    p_tokens = pred.split()
    r_tokens = ref.split()
    ln = lcs(p_tokens, r_tokens)
    if ln == 0:
        return 0.0
    prec = ln / len(p_tokens) if p_tokens else 0.0
    rec = ln / len(r_tokens) if r_tokens else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# final evaluation
def evaluate_all_embeddings():

    EMBEDDINGS = ["glove", "fasttext", "tfidf"]

    # setting token IDs
    tok2idx, idx2tok = load_vocab(CONFIG["VOCAB_PATH"])
    global PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID
    PAD_TOKEN_ID = tok2idx.get("<pad>", 0)
    START_TOKEN_ID = tok2idx.get("<start>", 1)
    END_TOKEN_ID = tok2idx.get("<end>", 2)

    # loading test.csv file
    df = pd.read_csv(CONFIG["PREPROCESSED_CSV"])
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    all_results = {}
    sample_outputs = {}

    for emb in EMBEDDINGS:
        print(f"Evaluating embedding: {emb.upper()}")

        # checkpoint
        ckpt_path = osp.join(CONFIG["RESULTS_DIR"], f"best_model_{emb}.pth")
        if not osp.exists(ckpt_path):
            print(f"[WARN] Missing checkpoint: {ckpt_path}")
            continue

        ck = torch.load(ckpt_path, map_location=device)
        cfg = ck["config"]

        # load embedding matrix from embedding_utils
        emb_matrix, emb_dim, _, _ = get_embedding_matrix(
            emb,
            vocab_path=CONFIG["VOCAB_PATH"],
            repr_dir=osp.dirname(CONFIG["PREPROCESSED_CSV"]),
        )

        # building models
        encoder = CustomCNNEncoder(cfg["IMAGE_FEATURE_DIM"]).to(device)
        decoder = EmotionLSTMDecoder(
            vocab_size=len(tok2idx),
            embed_dim=emb_dim,
            hidden_size=cfg["HIDDEN_SIZE"],
            num_emotions=cfg["NUM_EMOTIONS"],
            image_feature_dim=cfg["IMAGE_FEATURE_DIM"],
            dropout_rate=cfg["DROPOUT_RATE"],
            embedding_matrix=emb_matrix,
        ).to(device)

        encoder.load_state_dict(ck["encoder_state_dict"])
        decoder.load_state_dict(ck["decoder_state_dict"])

        print("Loaded model.")

        # metrics
        total_bleu4 = 0.0
        total_rouge1 = 0.0
        total_rougeL = 0.0
        n = 0

        samples = []
        ch = SmoothingFunction().method1

        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            painting = row["painting"]
            emo = int(row["emotion_label"])
            gt = clean_caption(str(row["utter_clean"]))

            npy_path = osp.join(CONFIG["IMAGE_FEAT_DIR"], f"{painting}.npy")
            if not osp.exists(npy_path):
                continue

            arr = np.load(npy_path)
            if arr.ndim == 3:
                # (H,W,3) -> (3,H,W)
                img = torch.tensor(arr).permute(2, 0, 1).float()
            else:
                img = torch.tensor(arr).float()

            pred = greedy_decode(
                encoder,
                decoder,
                img,
                emo,
                idx2tok,
                max_len=CONFIG["MAX_LEN"],
            )

            # metrics
            if gt.strip() and pred.strip():
                bleu4 = sentence_bleu(
                    [gt.split()],
                    pred.split(),
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=ch,
                )
                r1 = rouge_1_f(pred, gt)
                rL = rouge_l_f(pred, gt)

                total_bleu4 += bleu4
                total_rouge1 += r1
                total_rougeL += rL
                n += 1

            # Saving 5 samples
            if len(samples) < 5:
                samples.append({
                    "image": painting,
                    "emotion_id": emo,
                    "pred": pred,
                    "gt": gt,
                })

        # finalize scores
        all_results[emb] = {
            "bleu4": total_bleu4 / n if n else 0.0,
            "rouge1_f": total_rouge1 / n if n else 0.0,
            "rougeL_f": total_rougeL / n if n else 0.0,
            "num_eval_pairs": n,
        }

        sample_outputs[emb] = samples

        print(f"\n[{emb}] BLEU-4:   {all_results[emb]['bleu4']:.4f}")
        print(f"[{emb}] ROUGE-1: {all_results[emb]['rouge1_f']:.4f}")
        print(f"[{emb}] ROUGE-L: {all_results[emb]['rougeL_f']:.4f}")

    # JSON outputs

    out_dir = CONFIG["RESULTS_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    summary_path = osp.join(out_dir, "m1_eval_summary.json")
    samples_path = osp.join(out_dir, "m1_eval_samples.json")

    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    with open(samples_path, "w") as f:
        json.dump(sample_outputs, f, indent=2)

    print("\nSaved:")
    print(" ", summary_path)
    print(" ", samples_path)


if __name__ == "__main__":
    evaluate_all_embeddings()
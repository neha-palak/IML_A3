"""
evaluate_m1.py

Loads best checkpoint for Model1 and runs greedy decoding on the test split,
computes BLEU and ROUGE-L for a subset of images, prints sample predictions.
"""
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from model1_cnn import load_vocab, ArtEmisDataset, collate_fn, CaptionModel  # import from training script
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TFIDF_DIR = "data_preprocessed/tfidf_npy"
TEST_CSV = "data_preprocessed/test.csv"
FEATURES_DIR = "data_preprocessed/features"
VOCAB_PATH = "data_preprocessed/vocab.pkl"
MAX_SEQ_LEN = 25
checkpoint_path= "checkpoints/m1_pt/m1_best.pt"

def evaluate(checkpoint_path, embedding, max_samples=200):
    device = torch.device(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    tfidf_dim = config.get("tfidf_dim", 0)

    vocab = load_vocab(VOCAB_PATH)
    idx2tok = {v: k for k, v in vocab.items()}
    pad_idx = vocab.get("<pad>", 0)
    sos_idx = vocab.get("<start>", 1)
    eos_idx = vocab.get("<end>", None)

    model = CaptionModel(
        vocab_size=len(vocab),
        embedding_type=embedding,
        embedding_matrix_path=None if embedding not in ("glove","fasttext") else ("data_preprocessed/emb_glove_300d.npy" if embedding=="glove" else "data_preprocessed/emb_fasttext_300d.npy"),
        tfidf_dim=tfidf_dim if tfidf_dim>0 else None,
        embed_dim=config.get("embed_dim", 300),
        image_feat_dim=256,
        emo_dim=config.get("emo_dim", 64),
        lstm_hidden=config.get("lstm_hidden", 256),
        num_emotions=config.get("num_emotions", 9)
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds = ArtEmisDataset(TEST_CSV, FEATURES_DIR, "test", use_tfidf=(embedding=="tfidf"), tfidf_dir=TFIDF_DIR, max_len=MAX_SEQ_LEN)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    bleu_smooth = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    bleu_scores = []
    rouge_scores = []
    for i, (imgs, toks, emos, tfidf) in enumerate(tqdm(test_loader, total=min(len(test_ds), max_samples))):
        if i >= max_samples:
            break
        imgs = imgs[0]
        emo_id = int(emos[0].item())
        tfidf_vec = tfidf[0] if tfidf is not None else None

        pred_ids = model.greedy_decode(imgs, emo_id, sos_idx, eos_idx=eos_idx, max_len=MAX_SEQ_LEN, tfidf_vec=(tfidf_vec if embedding=="tfidf" else None), device=device)
        pred_tokens = [idx2tok.get(int(x), "<unk>") for x in pred_ids]
        ref_ids = toks[0].tolist()
        ref_tokens = [idx2tok.get(int(x), "<unk>") for x in ref_ids if x not in (pad_idx, sos_idx, eos_idx)]

        try:
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=bleu_smooth)
        except Exception:
            bleu = 0.0
        rouge_l = rouge.score(" ".join(ref_tokens), " ".join(pred_tokens))['rougeL'].fmeasure

        bleu_scores.append(bleu)
        rouge_scores.append(rouge_l)

        if i < 5:
            print("\nSample", i)
            print("Ref :", " ".join(ref_tokens))
            print("Pred:", " ".join(pred_tokens))

    print("\nAverage BLEU:", float(np.mean(bleu_scores)) if bleu_scores else 0.0)
    print("Average ROUGE-L:", float(np.mean(rouge_scores)) if rouge_scores else 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint (m1_best.pt)")
    parser.add_argument("--embedding", choices=["glove", "fasttext", "tfidf", "trainable"], default="glove")
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.embedding, max_samples=args.max_samples)

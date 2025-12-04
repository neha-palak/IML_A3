

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

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False


def sinusoidal_positional_encoding(n_pos: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(n_pos, d_model)
    position = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         (-float(np.log(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=32, in_chans=3, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B, E, H', W')
        x = self.proj(x)  
        B, E, Hn, Wn = x.shape
        # (B, N, E)
        return x.flatten(2).transpose(1, 2)  


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
            embed_dim, num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
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
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )

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
    def __init__(self, layer: DecoderLayerCustom, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [layer] +
            [DecoderLayerCustom(layer.self_attn.embed_dim,
                                layer.self_attn.num_heads,
                                layer.linear1.out_features,
                                layer.dropout.p)
             for _ in range(num_layers - 1)]
        )

    def forward(self, tgt, memory, tgt_mask=None):
        out = tgt
        for layer in self.layers:
            out = layer(out, memory, tgt_mask=tgt_mask)
        return out


class VisionLanguageTransformer(nn.Module):
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        D_enc = cfg.get("vit_embed_dim", 256)
        D_dec = cfg.get("decoder_embed_dim", 256)
        vocab_size = cfg["vocab_size"]
        num_emotions = cfg.get("num_emotions", 9)
        max_seq_len = cfg.get("max_seq_len", 25)
        dropout = cfg.get("dropout", 0.1)

        self.encoder = VisionTransformerEncoder(
            img_size=cfg.get("image_size", 128),
            patch_size=cfg.get("patch_size", 32),
            embed_dim=D_enc,
            depth=cfg.get("vit_depth", 2),
            num_heads=cfg.get("vit_num_heads", 4),
            dropout=dropout,
        )

        self.token_embed = nn.Embedding(vocab_size, D_dec)
        pos_tokens = sinusoidal_positional_encoding(max_seq_len + 1, D_dec)
        self.pos_embed_tokens = nn.Parameter(pos_tokens, requires_grad=False)

        if D_enc != D_dec:
            self.enc_to_dec = nn.Linear(D_enc, D_dec)
        else:
            self.enc_to_dec = nn.Identity()

        self.emotion_emb = nn.Embedding(num_emotions, D_dec)
        nn.init.xavier_uniform_(self.emotion_emb.weight)

        dec_layer = DecoderLayerCustom(
            d_model=D_dec,
            nhead=cfg.get("decoder_num_heads", 4),
            dim_feedforward=D_dec * 4,
            dropout=dropout,
        )
        self.decoder = TransformerDecoderCustom(
            dec_layer, cfg.get("decoder_depth", 2)
        )

        self.output_proj = nn.Linear(D_dec, vocab_size)
        self.pad_idx = cfg.get("pad_idx", 0)

    def forward(self, images, token_ids, emo_ids):
        """
        images: (B, 3, H, W)
        token_ids: (B, T)
        emo_ids: (B,)
        """
        device = images.device
        B, T = token_ids.shape

        enc = self.encoder(images)       
        enc = self.enc_to_dec(enc)       
        memory = enc.transpose(0, 1)      

        tok_emb = self.token_embed(token_ids)          
        emo_vec = self.emotion_emb(emo_ids).unsqueeze(1)  
        dec_in = torch.cat([emo_vec, tok_emb], dim=1)     

        pos = self.pos_embed_tokens[: T + 1].unsqueeze(0).to(device)
        dec_in = dec_in + pos                          
        dec_in = dec_in.transpose(0, 1)                  

        tgt_mask = torch.triu(
            torch.ones(T + 1, T + 1, device=device), diagonal=1
        ).bool()
        dec_out = self.decoder(dec_in, memory, tgt_mask=tgt_mask)
        dec_out = dec_out.transpose(0, 1)               
        logits = self.output_proj(dec_out)             
        return logits[:, 1:, :]                         

    def greedy_decode(
        self,
        image: torch.Tensor,
        emo_id: int,
        sos_idx: int,
        eos_idx: int,
        max_len: int,
        device: torch.device,
    ) -> List[int]:
        self.eval()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(device)
        emo = torch.tensor([emo_id], dtype=torch.long, device=device)

        with torch.no_grad():
            enc = self.encoder(image)
            enc = self.enc_to_dec(enc)
            memory = enc.transpose(0, 1)

            cur = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
            generated: List[int] = []

            for step in range(max_len):
                cur_padded = F.pad(cur, (0, max_len - cur.size(1)),
                                   value=self.pad_idx)
                logits = self.forward(image, cur_padded, emo)  # (1, T, V)
                step_idx = cur.size(1) - 1
                logit_step = logits[:, step_idx, :]            # (1, V)
                next_id = torch.argmax(logit_step, dim=-1).item()
                generated.append(next_id)
                if next_id == eos_idx:
                    break
                cur = torch.cat(
                    [cur, torch.tensor([[next_id]], device=device)], dim=1
                )

        return generated



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
    ckpt_dir = Path(args.checkpoints_root) / emb_type
    ckpt_path = ckpt_dir / f"m2_{emb_type}_best.pt"
    if not ckpt_path.exists():
        print(f"[{emb_type}] Best checkpoint not found at {ckpt_path}, skipping.")
        return {}

    print(f"\nEvaluating embedding type: {emb_type}")
    print("Loading checkpoint:", ckpt_path)
    ck = torch.load(str(ckpt_path), map_location=device)

    if "config" in ck:
        cfg = ck["config"]
        print("Config loaded from checkpoint.")
    else:
        print("No config in checkpoint; reconstructing from state_dict shapes.")
        sd = ck["model_state_dict"]

        # token embedding dim 
        dec_dim = sd["token_embed.weight"].shape[1]

        # encoder embed dim (should match decoder for training)
        enc_dim = sd["encoder.patch_embed.proj.weight"].shape[0]

        # num emotions
        num_emotions = sd["emotion_emb.weight"].shape[0]

        # max_seq_len+1 from positional embedding
        max_pos = sd["pos_embed_tokens"].shape[0]
        max_seq_len = max_pos - 1

        cfg = {
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "vit_embed_dim": int(enc_dim),
            "vit_depth": args.vit_depth,
            "vit_num_heads": args.vit_num_heads,
            "decoder_embed_dim": int(dec_dim),
            "decoder_depth": args.decoder_depth,
            "decoder_num_heads": args.decoder_num_heads,
            "max_seq_len": int(max_seq_len),
            "dropout": args.dropout,
            "num_emotions": int(num_emotions),
        }

    cfg["vocab_size"] = len(tok2idx)
    cfg["pad_idx"] = pad_idx

    model = VisionLanguageTransformer(cfg)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

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
        if arr.ndim == 3:
            img = torch.tensor(arr).permute(2, 0, 1).float()
        else:
            img = torch.tensor(arr).float()

        # reference tokens
        if isinstance(row.get("tokens_str", None), str):
            ref_tokens = row["tokens_str"].split()
        else:
            ref_tokens = str(row["utter_clean"]).split()

        gen_ids = model.greedy_decode(
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

        references.append(ref_tokens)
        hypotheses.append(hyp_tokens)

        if NLTK_AVAILABLE:
            meteor_refs.append(" ".join(ref_tokens))
            meteor_hyps.append(" ".join(hyp_tokens))

    print(f"[{emb_type}] Collected {len(hypotheses)} hypothesis–reference pairs.")

    # BLEU
    if references and hypotheses:
        bleu1 = corpus_bleu(
            [[r] for r in references], hypotheses,
            weights=(1.0, 0.0, 0.0, 0.0),
            smoothing_function=smoothie,
        )
        bleu4 = corpus_bleu(
            [[r] for r in references], hypotheses,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie,
        )
    else:
        bleu1 = bleu4 = 0.0

    # ROUGE-1 / ROUGE-L
    rouge1_f_list, rougeL_f_list = [], []
    for hyp, ref in zip(hypotheses, references):
        _, _, f1_1 = simple_rouge_1(hyp, ref)
        _, _, f1_L = simple_rouge_l(hyp, ref)
        rouge1_f_list.append(f1_1)
        rougeL_f_list.append(f1_L)

    rouge1_f = float(np.mean(rouge1_f_list)) if rouge1_f_list else 0.0
    rougeL_f = float(np.mean(rougeL_f_list)) if rougeL_f_list else 0.0

    # METEOR
    # meteor_avg = None
    # if NLTK_AVAILABLE and meteor_refs:
    #     try:
    #         scores = [meteor_score([r], h) for r, h in zip(meteor_refs, meteor_hyps)]
    #         meteor_avg = float(np.mean(scores))
    #     except Exception:
    #         meteor_avg = None

    # qualitative examples
    print(f"\n[{emb_type}] Sample qualitative generations:")
    num_show = min(args.num_examples, len(hypotheses))
    for i in range(num_show):
        ref = " ".join(references[i])
        hyp = " ".join(hypotheses[i])
        print(f"  Example {i+1}:")
        print(f"    REF: {ref}")
        print(f"    HYP: {hyp}")

    results = {
        "embedding_type": emb_type,
        "bleu1": float(bleu1),
        "bleu4": float(bleu4),
        "rouge1_f": rouge1_f,
        "rougeL_f": rougeL_f,
        # "meteor": meteor_avg,
        "num_pairs": len(hypotheses),
    }
    print(f"\n[{emb_type}] metrics:")
    print(json.dumps(results, indent=2))
    return results



def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VLT (Model 2) for multiple embedding types.")
    p.add_argument("--checkpoints-root", type=str, default="new_checkpoints",
                   help="Root folder with subdirs glove/, fasttext/, /tfidf")
    p.add_argument("--embedding-types", type=str, nargs="+",
                   default=["glove", "fasttext", "tfidf"])
    p.add_argument("--csv", type=str, default="new_preprocessed/artemis_preprocessed.csv")
    p.add_argument("--features-root", type=str, default="new_preprocessed/features")
    p.add_argument("--vocab", type=str, default="new_preprocessed/vocab.pkl")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_gen_len", type=int, default=25)
    p.add_argument("--num_examples", type=int, default=5)

    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--patch_size", type=int, default=32)
    p.add_argument("--vit_embed_dim", type=int, default=256)
    p.add_argument("--vit_depth", type=int, default=2)
    p.add_argument("--vit_num_heads", type=int, default=4)
    p.add_argument("--decoder_embed_dim", type=int, default=256)
    p.add_argument("--decoder_depth", type=int, default=2)
    p.add_argument("--decoder_num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_emotions", type=int, default=9)

    p.add_argument("--out-json", type=str, default="new_checkpoints/m2_eval_summary.json")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(osp.dirname(args.out_json), exist_ok=True)

    tok2idx, idx2tok, pad_idx, sos_idx, eos_idx = load_vocab(args.vocab)
    print("Vocab size:", len(tok2idx))
    print("PAD idx:", pad_idx, "SOS idx:", sos_idx, "EOS idx:", eos_idx)

    all_results = []
    for emb in args.embedding_types:
        res = evaluate_for_embedding(
            emb, args, tok2idx, idx2tok, pad_idx, sos_idx, eos_idx
        )
        if res:
            all_results.append(res)

    summary = {
        "results": all_results,
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved evaluation summary to:", args.out_json)


if __name__ == "__main__":
    main()
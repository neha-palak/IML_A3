#!/usr/bin/env python3
"""
attention_visualiser.py

High-resolution attention visualisation for Model 2 (VLT).

- Loads a trained VisionLanguageTransformerEmotion checkpoint
- Runs greedy decoding for a (painting, emotion) pair
- Hooks into decoder cross-attention to extract attention weights
- For each generated word, shows an image+heatmap overlay

Run example:

python3 scripts/attention_visualiser.py \
  --embedding-type tfidf \
  --csv new_preprocessed/artemis_preprocessed.csv \
  --features-root new_preprocessed/features \
  --checkpoint-root new_checkpoints \
  --painting arkhip-kuindzhi_birches-1879 \
  --emotion-id 6 \
  --max-words 8
"""

import os
import os.path as osp
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# ----------------- import model class -----------------
try:
    # if used as package (e.g. from notebooks with scripts. prefix)
    from scripts.M2_transformers import VisionLanguageTransformerEmotion
except ImportError:
    # when running from scripts/ directly
    from M2_transformers import VisionLanguageTransformerEmotion


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------- small helpers -----------------

def load_ckpt(ckpt_path: str) -> Tuple[VisionLanguageTransformerEmotion, Dict[str, int], Dict[int, str]]:
    """
    Load VLT + vocab from a best checkpoint.
    Expects keys: "cfg", "tok2idx", "model_state_dict".
    """
    ck = torch.load(ckpt_path, map_location=DEVICE)

    cfg = ck.get("cfg", ck.get("config", None))
    if cfg is None or "tok2idx" not in ck:
        raise RuntimeError(f"Checkpoint {ckpt_path} missing cfg or tok2idx")

    tok2idx = ck["tok2idx"]
    idx2tok = {i: t for t, i in tok2idx.items()}

    vocab_size = len(tok2idx)
    pad_idx = tok2idx.get("<pad>", 0)
    cfg["pad_idx"] = pad_idx

    # infer embedding dim if not stored
    if "decoder_embed_dim" in cfg:
        token_embed_dim = cfg["decoder_embed_dim"]
    else:
        sd = ck["model_state_dict"]
        token_embed_dim = sd["token_embed.weight"].shape[1]

    model = VisionLanguageTransformerEmotion(
        cfg=cfg,
        vocab_size=vocab_size,
        token_embed_dim=token_embed_dim,
        token_emb_matrix=None,
        freeze_token_emb=True,
    )
    model.load_state_dict(ck["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, tok2idx, idx2tok


def load_image_npy(features_root: str, painting: str) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load (H,W,3) float image from <features_root>/<painting>.npy
    and convert to tensor (1,3,H,W).
    """
    path = Path(features_root) / f"{painting}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Feature not found: {path}")
    arr = np.load(path)
    if arr.ndim == 3:
        img_t = torch.tensor(arr).permute(2, 0, 1).float()
    else:
        img_t = torch.tensor(arr).float()
    return img_t.unsqueeze(0), arr  # (1,3,H,W), (H,W,3)


def greedy_decode(
    model: VisionLanguageTransformerEmotion,
    img: torch.Tensor,
    emo_id: int,
    tok2idx: Dict[str, int],
    idx2tok: Dict[int, str],
    max_len: int,
) -> List[int]:
    """
    Greedy decode returning *token ids* (including <start> and <end>).
    """
    pad_idx = tok2idx.get("<pad>", 0)
    sos_idx = tok2idx.get("<start>", 1)
    eos_idx = tok2idx.get("<end>", 2)

    img = img.to(DEVICE)
    emo = torch.tensor([emo_id], dtype=torch.long, device=DEVICE)

    cur = torch.tensor([[sos_idx]], dtype=torch.long, device=DEVICE)
    generated: List[int] = [sos_idx]

    model.eval()
    with torch.no_grad():
        for _ in range(max_len):
            # pad to max_len used in training so shapes match
            padded = F.pad(cur, (0, max_len - cur.size(1)), value=pad_idx)
            logits = model(img, padded, emo)  # (1, T, V)
            step_idx = cur.size(1) - 1
            logits_t = logits[:, step_idx, :]  # (1,V)
            next_id = int(torch.argmax(logits_t, dim=-1).item())
            generated.append(next_id)
            cur = torch.cat([cur, torch.tensor([[next_id]], device=DEVICE)], dim=1)
            if next_id == eos_idx:
                break

    return generated


def ids_to_words(ids: List[int], idx2tok: Dict[int, str],
                 eos_idx: int, pad_idx: int) -> List[str]:
    words = []
    for i in ids:
        if i == eos_idx or i == pad_idx:
            break
        w = idx2tok.get(i, "<unk>")
        if w in ("<start>", "<end>", "<pad>"):
            continue
        words.append(w)
    return words


# ----------------- attention extraction -----------------

def collect_cross_attention(
    model: VisionLanguageTransformerEmotion,
    img: torch.Tensor,
    token_ids: torch.Tensor,
    emo_id: int,
):
    """
    Run a single forward pass while collecting cross-attention weights
    from each DecoderLayerCustom.cross_attn via forward hooks.

    Returns:
        attn_per_layer: List[Tensor] of length L
            each: (T+1, S)   averaged over heads, batch dim removed.
    """
    img = img.to(DEVICE)
    emo = torch.tensor([emo_id], dtype=torch.long, device=DEVICE)
    token_ids = token_ids.to(DEVICE)

    attn_per_layer: List[torch.Tensor] = []
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inp, out):
            # out is (attn_output, attn_weights)
            attn_weights = out[1]  # (B, tgt_len, src_len)
            attn_per_layer.append(attn_weights[0].detach().cpu())  # (T+1, S)
        return hook

    # register hooks on each decoder layer's cross_attn
    for i, layer in enumerate(model.decoder.layers):
        h = layer.cross_attn.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        _ = model(img, token_ids, emo)

    # remove hooks
    for h in hooks:
        h.remove()

    return attn_per_layer


def upsample_attention(
    patch_attn: torch.Tensor,
    img_h: int,
    img_w: int,
    num_patches: int,
) -> np.ndarray:
    """
    patch_attn: (P,) attention over patches (excluding CLS)
    Returns: (img_h, img_w) numpy heatmap in [0,1]
    """
    side = int(num_patches ** 0.5)
    patch_grid = patch_attn.reshape(1, 1, side, side)  # (1,1,Hp,Wp)
    up = F.interpolate(
        patch_grid,
        size=(img_h, img_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze().numpy()
    up = up - up.min()
    if up.max() > 0:
        up = up / up.max()
    return up


# ----------------- main visualisation logic -----------------

def visualize_sample(
    embedding_type: str,
    csv_path: str,
    features_root: str,
    checkpoint_root: str,
    painting: str,
    emotion_id: int,
    max_words: int = 8,
):
    # 1) Load checkpoint & model
    ckpt_path = Path(checkpoint_root) / embedding_type / f"m2_{embedding_type}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    print(f"Loading model from {ckpt_path}")
    model, tok2idx, idx2tok = load_ckpt(str(ckpt_path))
    max_len = model.cfg["max_seq_len"]
    pad_idx = tok2idx.get("<pad>", 0)
    eos_idx = tok2idx.get("<end>", 2)

    # 2) Load image
    img_t, img_arr = load_image_npy(features_root, painting)  # (1,3,H,W), (H,W,3)
    img_h, img_w, _ = img_arr.shape

    # 3) Greedy decode to get caption token ids
    gen_ids = greedy_decode(
        model, img_t, emotion_id, tok2idx, idx2tok, max_len=max_len
    )
    words = ids_to_words(gen_ids, idx2tok, eos_idx, pad_idx)
    print("Generated caption:", " ".join(words))

    # 4) Build full token sequence padded to max_len
    #    (same ids as used for words, padded with <pad>)
    token_ids = gen_ids[:max_len]
    if len(token_ids) < max_len:
        token_ids = token_ids + [pad_idx] * (max_len - len(token_ids))
    token_ids_tensor = torch.tensor([token_ids], dtype=torch.long)

    # 5) Collect cross-attention maps (one per decoder layer)
    attn_layers = collect_cross_attention(model, img_t, token_ids_tensor, emotion_id)
    if not attn_layers:
        print("No attention maps collected – check hooks.")
        return

    # use last layer attention
    attn_last = attn_layers[-1]  # (T+1, S)
    # ignore position 0 (emotion token); we want positions 1.. for words
    num_patches = model.encoder.patch_embed.num_patches


    print("\nVisualizing attention maps...")
    for wi, w in enumerate(words[:max_words]):
        t_pos = wi + 1  # +1 because 0 is emotion position
        if t_pos >= attn_last.size(0):
            break

        # attention over encoder tokens (CLS + patches)
        att_vec = attn_last[t_pos]  # (S,)
        # drop CLS token (index 0), keep patch tokens
        patch_attn = att_vec[1:1 + num_patches]  # (P,)

        heat = upsample_attention(patch_attn, img_h, img_w, num_patches)

        print(f"  Word {wi}: {w}")
        plt.figure(figsize=(4, 4))
        plt.imshow(img_arr)
        plt.imshow(heat, cmap="jet", alpha=0.45)
        plt.axis("off")
        plt.title(f"Attention for: '{w}'")
        plt.show()


# ----------------- CLI -----------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="High-resolution attention visualisation for VLT"
    )
    parser.add_argument("--embedding-type", type=str,
                        choices=["glove", "fasttext", "tfidf"],
                        default="glove")
    parser.add_argument("--csv", type=str, default="new_preprocessed/artemis_preprocessed.csv")
    parser.add_argument("--features-root", type=str, default="new_preprocessed/features")
    parser.add_argument("--checkpoint-root", type=str, default="new_checkpoints")
    parser.add_argument("--painting", type=str, required=True,
                        help="Painting id (without .npy)")
    parser.add_argument("--emotion-id", type=int, required=True)
    parser.add_argument("--max-words", type=int, default=8)

    args = parser.parse_args()

    visualize_sample(
        embedding_type=args.embedding_type,
        csv_path=args.csv,
        features_root=args.features_root,
        checkpoint_root=args.checkpoint_root,
        painting=args.painting,
        emotion_id=args.emotion_id,
        max_words=args.max_words,
    )


if __name__ == "__main__":
    main()
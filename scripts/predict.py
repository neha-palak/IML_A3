
#!/usr/bin/env python3
"""
predict.py  (Transformer-only)

Use trained Vision-Language Transformer with emotion conditioning to generate
captions for a few (image, emotion) inputs.

Assumptions:
- Features at: new_preprocessed/features/<painting>.npy  (H,W,3 in [0,1])
- Best checkpoints at:
    new_checkpoints/glove/m2_glove_best.pt
    new_checkpoints/fasttext/m2_fasttext_best.pt
- Each checkpoint contains:
    - "cfg"      : config dict used in training
    - "tok2idx"  : vocabulary mapping {token -> idx}
    - "model_state_dict"

Run:
    python3 scripts/predict.py
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from transformers import VisionLanguageTransformerEmotion

# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------
FEATURES_ROOT = "new_preprocessed/features"

CKPT_VLT_GLOVE    = "new_checkpoints/glove/m2_glove_best.pt"
CKPT_VLT_FASTTEXT = "new_checkpoints/fasttext/m2_fasttext_best.pt"

EMO_ID2NAME = {
    0: "amusement", 1: "contentment", 2: "awe", 3: "excitement",
    4: "fear", 5: "anger", 6: "sadness", 7: "disgust", 8: "something else"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_feature(features_root: str, painting_name: str) -> torch.Tensor:
    """
    Load precomputed image feature from .npy:
        <features_root>/<painting_name>.npy
    Returns shape (1,3,H,W).
    """
    path = Path(features_root) / f"{painting_name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Feature not found: {path}")
    arr = np.load(path)
    if arr.ndim == 3:  # (H, W, 3)
        img = torch.tensor(arr).permute(2, 0, 1).float()
    else:              # assume already (3, H, W)
        img = torch.tensor(arr).float()
    return img.unsqueeze(0)  # (1,3,H,W)


def tokens_to_caption(
    token_ids: List[int],
    idx2tok: Dict[int, str],
    eos_idx: int,
    pad_idx: int
) -> str:
    """
    Convert token ids into text, skipping special tokens
    and stopping at EOS.
    """
    words = []
    for tid in token_ids:
        tid = int(tid)
        if tid == eos_idx:
            break
        if tid == pad_idx:
            continue
        w = idx2tok.get(tid, "<unk>")
        if w in ("<start>", "<end>", "<pad>"):
            continue
        words.append(w)
    return " ".join(words)


def load_vlt_from_ckpt(
    ckpt_path: str,
) -> Tuple[VisionLanguageTransformerEmotion, Dict[str, int]]:
    """
    Load VisionLanguageTransformerEmotion + tok2idx from a checkpoint:
      - "cfg"
      - "tok2idx"
      - "model_state_dict"
    """
    ck = torch.load(ckpt_path, map_location=DEVICE)

    if "cfg" not in ck or "tok2idx" not in ck:
        raise RuntimeError(f"Checkpoint {ckpt_path} missing 'cfg' or 'tok2idx' keys")

    cfg = ck["cfg"]
    tok2idx = ck["tok2idx"]
    vocab_size = len(tok2idx)
    token_embed_dim = cfg["decoder_embed_dim"]
    pad_idx = tok2idx.get("<pad>", 0)
    cfg["pad_idx"] = pad_idx   # ensure pad idx present

    model = VisionLanguageTransformerEmotion(
        cfg=cfg,
        vocab_size=vocab_size,
        token_embed_dim=token_embed_dim,
        token_emb_matrix=None,      # weights come from state_dict
        freeze_token_emb=True,
    )
    model.load_state_dict(ck["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, tok2idx


def greedy_vlt(
    model: VisionLanguageTransformerEmotion,
    img: torch.Tensor,
    emo_id: int,
    tok2idx: Dict[str, int],
    idx2tok: Dict[int, str],
) -> str:
    """
    Greedy decoding for the VLT model with emotion conditioning.
    Returns a caption string.
    """
    pad_idx = tok2idx["<pad>"]
    sos_idx = tok2idx["<start>"]
    eos_idx = tok2idx["<end>"]
    max_len = model.cfg["max_seq_len"]

    img = img.to(DEVICE)
    emo = torch.tensor([emo_id], dtype=torch.long, device=DEVICE)
    cur = torch.tensor([[sos_idx]], dtype=torch.long, device=DEVICE)

    model.eval()
    with torch.no_grad():
        for _ in range(max_len):
            # pad to max_len so shape matches training
            padded = F.pad(cur, (0, max_len - cur.size(1)), value=pad_idx)
            logits = model(img, padded, emo)      # (1, T, V)
            step_idx = cur.size(1) - 1
            next_id = logits[:, step_idx, :].argmax(dim=-1).item()
            cur = torch.cat([cur, torch.tensor([[next_id]], device=DEVICE)], dim=1)
            if next_id == eos_idx:
                break

    ids = cur.squeeze(0).tolist()
    return tokens_to_caption(ids, idx2tok, eos_idx, pad_idx)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("Using device:", DEVICE)

    # 1) Load transformer models (glove & fasttext)
    vlt_models = {}
    vlt_tok2idx = {}
    vlt_idx2tok = {}

    for name, path in {
        "glove": CKPT_VLT_GLOVE,
        "fasttext": CKPT_VLT_FASTTEXT,
    }.items():
        ckpt = Path(path)
        if not ckpt.exists():
            print(f"[WARN] VLT checkpoint for {name} not found at {ckpt}, skipping.")
            continue
        print(f"Loading VLT ({name}) from {ckpt} ...")
        model, tok2idx = load_vlt_from_ckpt(str(ckpt))
        vlt_models[name] = model
        vlt_tok2idx[name] = tok2idx
        vlt_idx2tok[name] = {i: t for t, i in tok2idx.items()}

    if not vlt_models:
        raise RuntimeError("No VLT models loaded – check CKPT paths.")
#     EMO_ID2NAME = {
#     0: "amusement", 1: "contentment", 2: "awe", 3: "excitement",
#     4: "fear", 5: "anger", 6: "sadness", 7: "disgust", 8: "something else"
# }

    # 2) Define the samples you want to show in the viva.
    #    Make sure the .npy exists at new_preprocessed/features/<painting>.npy
    samples = [
        # replace with names that definitely exist under new_preprocessed/features
        {"painting": "arkhip-kuindzhi_birches-1879", "emotion": 1}  
    ]

    results = []

    for s in samples:
        painting = s["painting"]
        emo_id = int(s["emotion"])
        emo_name = EMO_ID2NAME.get(emo_id, str(emo_id))

        print("=" * 90)
        print(f"Image:   {painting}")
        print(f"Emotion: {emo_id} ({emo_name})")

        try:
            img = load_feature(FEATURES_ROOT, painting)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            continue

        gen_entry = {
            "input": painting,
            "emotion": emo_id,
            "generations": {}
        }

        # generate captions for each embedding type
        for name, model in vlt_models.items():
            cap = greedy_vlt(
                model=model,
                img=img,
                emo_id=emo_id,
                tok2idx=vlt_tok2idx[name],
                idx2tok=vlt_idx2tok[name],
            )
            print(f"[VLT-{name}] {cap}")
            gen_entry["generations"][f"vlt_{name}"] = cap

        results.append(gen_entry)

    # 3) Save outputs for your report (optional)
    out_json = Path("predict_outputs_vlt.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved generations to {out_json}")


if __name__ == "__main__":
    main()
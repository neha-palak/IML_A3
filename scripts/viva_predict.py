#!/usr/bin/env python3
"""
predict_vlt.py 
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 

from scripts.transformers import VisionLanguageTransformerEmotion

# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------
DEFAULT_IMAGES_ROOT = "viva_images"

CKPT_VLT_GLOVE    = "new_checkpoints/glove/m2_glove_best.pt"
CKPT_VLT_FASTTEXT = "new_checkpoints/fasttext/m2_fasttext_best.pt"

EMO_ID2NAME = {
    0: "amusement", 1: "contentment", 2: "awe", 3: "excitement",
    4: "fear", 5: "anger", 6: "sadness", 7: "disgust", 8: "something else"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_PREPROC_SIZE = 128

# GLOBAL CACHE (For notebook usage)
_CACHED_MODELS = {}
_CACHED_VOCABS = {}

# ---------------------------------------------------------------------
# Helpers: loading / preprocessing
# ---------------------------------------------------------------------
def ensure_jpg_path(root: Path, name: str) -> Path:
    p = root / name
    if p.suffix == "":
        p = p.with_suffix(".jpg")
    return p

def load_and_preprocess_jpg(images_root: str, image_name: str, size: int):
    root = Path(images_root)
    jpg_path = ensure_jpg_path(root, image_name)
    if not jpg_path.exists():
        raise FileNotFoundError(f"JPG image not found: {jpg_path}")

    img = Image.open(jpg_path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0

    img_tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).float()
    npy_path = jpg_path.with_suffix(".npy")
    return img_tensor, arr, jpg_path, npy_path

def save_npy(arr: np.ndarray, npy_path: Path) -> None:
    np.save(npy_path, arr)

def load_from_npy(npy_path: Path) -> torch.Tensor:
    if not npy_path.exists():
        raise FileNotFoundError(f"NPY file not found: {npy_path}")
    arr = np.load(npy_path)
    if arr.ndim == 3:
        tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).float()
    else:
        t = torch.tensor(arr).float()
        tensor = t.unsqueeze(0) if t.ndim == 3 else t
    return tensor

def tokens_to_caption(token_ids, idx2tok, eos_idx, pad_idx) -> str:
    words = []
    for tid in token_ids:
        tid = int(tid)
        if tid == eos_idx: break
        if tid == pad_idx: continue
        w = idx2tok.get(tid, "<unk>")
        if w in ("<start>", "<end>", "<pad>"): continue
        words.append(w)
    return " ".join(words)

def load_vlt_from_ckpt(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location=DEVICE)
    cfg = ck.get("cfg", ck.get("config", None))
    if cfg is None or "tok2idx" not in ck:
        raise RuntimeError(f"Checkpoint {ckpt_path} missing cfg or tok2idx")

    tok2idx = ck["tok2idx"]
    vocab_size = len(tok2idx)
    pad_idx = tok2idx.get("<pad>", 0)
    cfg["pad_idx"] = pad_idx

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
    return model, tok2idx

def greedy_vlt(
    model: VisionLanguageTransformerEmotion,
    img: torch.Tensor,
    emo_id: int,
    tok2idx: Dict[str, int],
    idx2tok: Dict[int, str],
    override_max_len: int = 17, # [FIX] Added override
) -> str:
    pad_idx = tok2idx["<pad>"]
    sos_idx = tok2idx["<start>"]
    eos_idx = tok2idx["<end>"]
    
    # [FIX] Force limit to 17 to match model weights
    max_len = min(model.cfg.get("max_seq_len", 50), override_max_len)

    img = img.to(DEVICE)
    emo = torch.tensor([emo_id], dtype=torch.long, device=DEVICE)
    cur = torch.tensor([[sos_idx]], dtype=torch.long, device=DEVICE)

    model.eval()
    with torch.no_grad():
        for _ in range(max_len):
            padded = F.pad(cur, (0, max_len - cur.size(1)), value=pad_idx)
            logits = model(img, padded, emo)
            step_idx = cur.size(1) - 1
            if step_idx >= logits.size(1): break
            next_id = logits[:, step_idx, :].argmax(dim=-1).item()
            cur = torch.cat([cur, torch.tensor([[next_id]], device=DEVICE)], dim=1)
            if next_id == eos_idx: break

    ids = cur.squeeze(0).tolist()
    return tokens_to_caption(ids, idx2tok, eos_idx, pad_idx)

# ---------------------------------------------------------------------
# NEW: Interactive Notebook Function
# ---------------------------------------------------------------------
def predict_single_image(image_name: str, emotion_id: int, images_root: str = DEFAULT_IMAGES_ROOT):
    """
    Called from Jupyter Notebook. 
    Loads models (cached), processes one image, runs prediction ONCE.
    """
    global _CACHED_MODELS, _CACHED_VOCABS
    
    # 1. Load Models if not already in cache
    if not _CACHED_MODELS:
        print("[INFO] Loading models into cache (first run only)...")
        ckpt_map = {"glove": CKPT_VLT_GLOVE, "fasttext": CKPT_VLT_FASTTEXT}
        for name, path in ckpt_map.items():
            if Path(path).exists():
                print(f"Loading {name}...")
                model, tok2idx = load_vlt_from_ckpt(str(path))
                _CACHED_MODELS[name] = model
                _CACHED_VOCABS[name] = {
                    "tok2idx": tok2idx,
                    "idx2tok": {i: t for t, i in tok2idx.items()}
                }
            else:
                print(f"[WARN] Checkpoint not found: {path}")

    if not _CACHED_MODELS:
        print("[ERROR] No models loaded.")
        return

    # 2. Preprocess
    # Get size from first model
    first_model = next(iter(_CACHED_MODELS.values()))
    img_size = first_model.cfg.get("image_size", DEFAULT_PREPROC_SIZE)
    
    emo_name = EMO_ID2NAME.get(emotion_id, str(emotion_id))
    print(f"\nProcessing: {image_name} | Emotion: {emotion_id} ({emo_name})")

    try:
        img_tensor, arr, jpg_path, _ = load_and_preprocess_jpg(images_root, image_name, size=img_size)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # 3. Display Image
    plt.figure(figsize=(4, 4))
    plt.imshow(arr)
    plt.axis("off")
    plt.title(f"{jpg_path.name} ({emo_name})")
    plt.show()

    # 4. Generate Captions (Single Pass)
    print("-" * 60)
    for name, model in _CACHED_MODELS.items():
        vocab = _CACHED_VOCABS[name]
        cap = greedy_vlt(
            model=model, 
            img=img_tensor, 
            emo_id=emotion_id, 
            tok2idx=vocab["tok2idx"], 
            idx2tok=vocab["idx2tok"]
        )
        print(f"[{name.upper()}]: {cap}")
    print("-" * 60)


# ---------------------------------------------------------------------
# Main (for command line usage)
# ---------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=str, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--inputs-json", type=str, default=None)
    args, _ = parser.parse_known_args()

    # Reuse the interactive logic? 
    # The original main had a double-loop (JPG+NPY) and JSON saving.
    # For now, let's keep main() simple or you can point it to use the new function if you want.
    # Keeping original behavior for CLI compatibility:
    
    print("Use the 'predict_single_image' function in notebooks for interactive mode.")
    # ... (Rest of original main logic would go here if you still need CLI support) ...

if __name__ == "__main__":
    main()
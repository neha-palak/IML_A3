import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from M2_transformers import VisionLanguageTransformerEmotion
from M1_CNN import CustomCNNEncoder, EmotionLSTMDecoder


DEFAULT_IMAGES_ROOT = "viva_images"

# VLT checkpoints
CKPT_VLT_GLOVE    = "new_checkpoints/glove/m2_glove_best.pt"
CKPT_VLT_FASTTEXT = "new_checkpoints/fasttext/m2_fasttext_best.pt"
CKPT_VLT_TFIDF = "new_checkpoints/tfidf/m2_tfidf_best.pt"

# CNN+LSTM checkpoints 
CKPT_CNN_ROOT = "eval_outputs/results_cnn_lstm" 

EMO_ID2NAME = {
    0: "amusement", 1: "contentment", 2: "awe", 3: "excitement",
    4: "fear", 5: "anger", 6: "sadness", 7: "disgust", 8: "something else"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_PREPROC_SIZE = 128

# GLOBAL CACHES (for notebook usage)
_CACHED_VLT_MODELS: Dict[str, VisionLanguageTransformerEmotion] = {}
_CACHED_VLT_VOCABS: Dict[str, Dict[str, Dict]] = {}

_CACHED_CNN_MODELS: Dict[str, Dict[str, Any]] = {}  
_CACHED_CNN_VOCABS: Dict[str, Dict[str, Dict]] = {}  



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
        if tid == eos_idx:
            break
        if tid == pad_idx:
            continue
        w = idx2tok.get(tid, "<unk>")
        if w in ("<start>", "<end>", "<pad>"):
            continue
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
    override_max_len: int = 17,
) -> str:
    
    pad_idx = tok2idx["<pad>"]
    sos_idx = tok2idx["<start>"]
    eos_idx = tok2idx["<end>"]

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
            if step_idx >= logits.size(1):
                break
            next_id = logits[:, step_idx, :].argmax(dim=-1).item()
            cur = torch.cat([cur, torch.tensor([[next_id]], device=DEVICE)], dim=1)
            if next_id == eos_idx:
                break

    ids = cur.squeeze(0).tolist()
    return tokens_to_caption(ids, idx2tok, eos_idx, pad_idx)


def load_cnn_from_ckpt(emb_type: str):
    
    ckpt_path = Path(CKPT_CNN_ROOT) / f"best_model_{emb_type}.pth"
    ck = torch.load(ckpt_path, map_location=DEVICE)

    cfg: Dict[str, Any] = ck["config"]
    tok2idx: Dict[str, int] = ck["tok2idx"]

    vocab_size = len(tok2idx)
    dec_sd = ck["decoder_state_dict"]
    emb_dim = dec_sd["word_embeddings.weight"].shape[1]

    encoder = CustomCNNEncoder(output_dim=cfg["IMAGE_FEATURE_DIM"]).to(DEVICE)
    decoder = EmotionLSTMDecoder(
        vocab_size=vocab_size,
        embed_dim=emb_dim,
        hidden_size=cfg["HIDDEN_SIZE"],
        num_emotions=cfg["NUM_EMOTIONS"],
        image_feature_dim=cfg["IMAGE_FEATURE_DIM"],
        dropout_rate=cfg["DROPOUT_RATE"],
        embedding_matrix=None, 
    ).to(DEVICE)

    encoder.load_state_dict(ck["encoder_state_dict"])
    decoder.load_state_dict(dec_sd)

    encoder.eval()
    decoder.eval()

    return encoder, decoder, tok2idx


from typing import Dict, List

EMOTION_WORDS = [
    "amusement", "contentment", "awe", "excitement", "fear",
    "anger", "sadness", "disgust", "something", "something else"
]

def greedy_cnn(
    encoder: CustomCNNEncoder,
    decoder: EmotionLSTMDecoder,
    img: torch.Tensor,
    emo_id: int,
    tok2idx: Dict[str, int],
    max_len: int = 25,
) -> str:
   
    idx2tok = {i: t for t, i in tok2idx.items()}

    pad_idx = tok2idx.get("<pad>", 0)
    sos_idx = tok2idx.get("<start>", 1)
    eos_idx = tok2idx.get("<end>", 2)

    if img.ndim == 3:
        img = img.unsqueeze(0)
    img = img.to(DEVICE)

    emo = torch.tensor([emo_id], dtype=torch.long, device=DEVICE)

    result_ids: List[int] = []

    with torch.no_grad():
        img_feat = encoder(img)             
        h, c = decoder.init_hidden_state(img_feat, emo)

        cur = torch.tensor([[sos_idx]], dtype=torch.long, device=DEVICE)

        for _ in range(max_len):
            # last token only
            emb = decoder.word_embeddings(cur[:, -1:])
            emb = decoder.dropout_word(emb)

            out, (h, c) = decoder.lstm(emb, (h, c))        
            logits = decoder.output_layer(out[:, -1, :])   
            next_id = int(torch.argmax(logits, dim=-1).item())

            if next_id in (eos_idx, pad_idx):
                break

            word = idx2tok.get(next_id, "")

            if len(result_ids) == 0 and word in EMOTION_WORDS:
                cur = torch.cat(
                    [cur, torch.tensor([[next_id]], device=DEVICE)],
                    dim=1
                )
                continue

            # normal case: keep token
            result_ids.append(next_id)
            cur = torch.cat(
                [cur, torch.tensor([[next_id]], device=DEVICE)],
                dim=1
            )

    # Convert ids -> tokens
    tokens = [idx2tok.get(i, "<unk>") for i in result_ids]
    caption = " ".join(tokens)

    # strip asterisks and tidy whitespace
    caption = caption.replace("*", "").strip()

    return caption


#  Interactive Notebook Function
def predict_single_image(image_name: str, emotion_id: int, images_root: str = DEFAULT_IMAGES_ROOT):
    
    global _CACHED_VLT_MODELS, _CACHED_VLT_VOCABS
    global _CACHED_CNN_MODELS, _CACHED_CNN_VOCABS

    if not _CACHED_VLT_MODELS:
        print("[INFO] Loading VLT models into cache (first run only)...")
        ckpt_map = {"glove": CKPT_VLT_GLOVE, "fasttext": CKPT_VLT_FASTTEXT, "tfidf": CKPT_VLT_TFIDF}
        for name, path in ckpt_map.items():
            p = Path(path)
            if p.exists():
                print(f"  - loading VLT-{name} from {p}")
                model, tok2idx = load_vlt_from_ckpt(str(p))
                _CACHED_VLT_MODELS[name] = model
                _CACHED_VLT_VOCABS[name] = {
                    "tok2idx": tok2idx,
                    "idx2tok": {i: t for t, i in tok2idx.items()},
                }
            else:
                print(f"[WARN] VLT checkpoint not found: {p}")

    if not _CACHED_CNN_MODELS:
        print("[INFO] Loading CNN+LSTM models into cache (first run only)...")
        for name in ["glove", "fasttext", "tfidf"]:
            ckpt_path = Path(CKPT_CNN_ROOT) / f"best_model_{name}.pth"
            if ckpt_path.exists():
                print(f"  - loading CNN-{name} from {ckpt_path}")
                enc, dec, tok2idx = load_cnn_from_ckpt(name)
                _CACHED_CNN_MODELS[name] = {"encoder": enc, "decoder": dec}
                _CACHED_CNN_VOCABS[name] = {
                    "tok2idx": tok2idx,
                    "idx2tok": {i: t for t, i in tok2idx.items()},
                }
            else:
                print(f"[WARN] CNN checkpoint not found: {ckpt_path}")

    if not _CACHED_VLT_MODELS and not _CACHED_CNN_MODELS:
        print("[ERROR] No models loaded (neither VLT nor CNN).")
        return

    if _CACHED_VLT_MODELS:
        first_model = next(iter(_CACHED_VLT_MODELS.values()))
        img_size = first_model.cfg.get("image_size", DEFAULT_PREPROC_SIZE)
    else:
        img_size = DEFAULT_PREPROC_SIZE

    emo_name = EMO_ID2NAME.get(emotion_id, str(emotion_id))
    print(f"\nProcessing: {image_name} | Emotion: {emotion_id} ({emo_name})")

    try:
        img_tensor, arr, jpg_path, _ = load_and_preprocess_jpg(
            images_root, image_name, size=img_size
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    plt.figure(figsize=(4, 4))
    plt.imshow(arr)
    plt.axis("off")
    plt.title(f"{jpg_path.name} ({emo_name})")
    plt.show()

    print("-" * 60)
    print("VLT (Transformer) predictions:")
    for name, model in _CACHED_VLT_MODELS.items():
        vocab = _CACHED_VLT_VOCABS[name]
        cap = greedy_vlt(
            model=model,
            img=img_tensor,
            emo_id=emotion_id,
            tok2idx=vocab["tok2idx"],
            idx2tok=vocab["idx2tok"],
        )
        print(f"[VLT-{name}]   {cap}")

    print("-" * 60)
    print("CNN+LSTM predictions:")
    for name, mdl in _CACHED_CNN_MODELS.items():
        enc = mdl["encoder"]
        dec = mdl["decoder"]
        vocab = _CACHED_CNN_VOCABS[name]
        cap = greedy_cnn(
            encoder=enc,
            decoder=dec,
            img=img_tensor.squeeze(0),  
            emo_id=emotion_id,
            tok2idx=vocab["tok2idx"],
        )
        print(f"[CNN-{name}]   {cap}")
    print("-" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=str, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--image-name", type=str, default=None)
    parser.add_argument("--emotion-id", type=int, default=1)
    args, _ = parser.parse_known_args()

    if args.image_name is None:
        print("For interactive use, import predict_single_image in a notebook:")
        print("  from scripts.predict_vlt import predict_single_image")
        print('  predict_single_image("my_image", emotion_id=1)')
        return

    predict_single_image(args.image_name, args.emotion_id, images_root=args.images_root)


if __name__ == "__main__":
    main()
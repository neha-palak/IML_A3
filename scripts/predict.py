import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F

from M2_transformers import VisionLanguageTransformerEmotion
from M1_CNN import CustomCNNEncoder, EmotionLSTMDecoder

FEATURES_ROOT = "new_preprocessed/features"

CKPT_VLT_GLOVE    = "new_checkpoints/glove/m2_glove_best.pt"
CKPT_VLT_FASTTEXT = "new_checkpoints/fasttext/m2_fasttext_best.pt"

CKPT_CNN_ROOT = "eval_outputs/results_cnn_lstm"

EMO_ID2NAME = {
    0: "amusement", 1: "contentment", 2: "awe", 3: "excitement",
    4: "fear", 5: "anger", 6: "sadness", 7: "disgust", 8: "something else"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_feature(features_root: str, painting_name: str) -> torch.Tensor:
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
            padded = F.pad(cur, (0, max_len - cur.size(1)), value=pad_idx)
            logits = model(img, padded, emo)      # (1, T, V)
            step_idx = cur.size(1) - 1
            next_id = logits[:, step_idx, :].argmax(dim=-1).item()
            cur = torch.cat([cur, torch.tensor([[next_id]], device=DEVICE)], dim=1)
            if next_id == eos_idx:
                break

    ids = cur.squeeze(0).tolist()
    return tokens_to_caption(ids, idx2tok, eos_idx, pad_idx)

def load_cnn_from_ckpt(
    emb_type: str
) -> Tuple[CustomCNNEncoder, EmotionLSTMDecoder, Dict[str, int]]:

    ckpt_path = Path(CKPT_CNN_ROOT) / f"best_model_{emb_type}.pth"
    ck = torch.load(ckpt_path, map_location=DEVICE)

    cfg: Dict[str, Any] = ck["config"]
    tok2idx: Dict[str, int] = ck["tok2idx"]

    vocab_size = len(tok2idx)

    # Infer embedding dim from saved decoder weights
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
        embedding_matrix=None,   # weights will come from state_dict
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
        # 1) encode image + init hidden
        img_feat = encoder(img)               # (1, D_img)
        h, c = decoder.init_hidden_state(img_feat, emo)

        cur = torch.tensor([[sos_idx]], dtype=torch.long, device=DEVICE)

        for _ in range(max_len):
            # last token only
            emb = decoder.word_embeddings(cur[:, -1:])
            emb = decoder.dropout_word(emb)

            out, (h, c) = decoder.lstm(emb, (h, c))        # (1,1,H)
            logits = decoder.output_layer(out[:, -1, :])   # (1,V)
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

def main():
    print("Using device:", DEVICE)
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
        print("[WARN] No VLT models loaded – check CKPT paths.")
    cnn_encoders = {}
    cnn_decoders = {}
    cnn_tok2idx = {}
    cnn_idx2tok = {}

    for name in ["glove", "fasttext", "tfidf"]:
        ckpt_path = Path(CKPT_CNN_ROOT) / f"best_model_{name}.pth"
        if not ckpt_path.exists():
            print(f"[WARN] CNN checkpoint for {name} not found at {ckpt_path}, skipping.")
            continue
        print(f"Loading CNN+LSTM ({name}) from {ckpt_path} ...")
        enc, dec, tok2idx = load_cnn_from_ckpt(name)
        cnn_encoders[name] = enc
        cnn_decoders[name] = dec
        cnn_tok2idx[name] = tok2idx
        cnn_idx2tok[name] = {i: t for t, i in tok2idx.items()}

    if not cnn_encoders:
        print("[WARN] No CNN models loaded – check eval_outputs/results_cnn_lstm.")
       
    samples = [
        # Make sure these exist under new_preprocessed/features
        {"painting": "arkhip-kuindzhi_birches-1879", "emotion": 1},
        # {"painting": "gustave-dore_elijah-is-nourished-by-an-angel", "emotion": 2},
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
            "emotion_id": emo_id,
            "emotion_name": emo_name,
            "generations": {}
        }

        for name, model in vlt_models.items():
            cap = greedy_vlt(
                model=model,
                img=img,
                emo_id=emo_id,
                tok2idx=vlt_tok2idx[name],
                idx2tok=vlt_idx2tok[name],
            )
            print(f"[VLT-{name}]  {cap}")
            gen_entry["generations"][f"vlt_{name}"] = cap

        for name, enc in cnn_encoders.items():
            dec = cnn_decoders[name]
            tok2idx = cnn_tok2idx[name]
            cap_cnn = greedy_cnn(
                encoder=enc,
                decoder=dec,
                img=img.squeeze(0),
                emo_id=emo_id,
                tok2idx=tok2idx,
            )
            print(f"[CNN-{name}]  {cap_cnn}")
            gen_entry["generations"][f"cnn_{name}"] = cap_cnn

        results.append(gen_entry)

    out_json = Path("predict_outputs_vlt_cnn.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved generations to {out_json}")


if __name__ == "__main__":
    main()
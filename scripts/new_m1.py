#!/usr/bin/env python3
"""
M1.py: CNN + Emotion-aware LSTM Captioning model

Supports text embedding types via embedding_utils.get_embedding_matrix:
- random      : normal init, trainable
- glove       : uses new_preprocessed/emb_glove_300d.npy
- fasttext    : uses new_preprocessed/emb_fasttext_300d.npy
- tfidf       : uses TF-IDF + SVD projections (token-level) built earlier

Assumes:
- Preprocessed CSV: new_preprocessed/artemis_preprocessed.csv
  with columns: painting, emotion_label, token_ids_with_emotion_str, split, ...
- Image features: new_preprocessed/features/<painting>.npy  (H,W,3 in [0,1])
- Vocab: new_preprocessed/vocab.pkl
- embedding_utils.py is in Python path and implements get_embedding_matrix(...)

This script will run THREE experiments in sequence:
    1. GloVe
    2. FastText
    3. TF-IDF

and log them all into a shared history JSON.
"""

import os
import os.path as osp
import ast
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from embedding_utils import get_embedding_matrix  # <<< central embedding loader

# ----------------------------------------------------
# GLOBAL BASE CONFIG  (per-run copy will be made)
# ----------------------------------------------------
BASE_CONFIG = {
    "EMBEDDING_TYPE": "glove",              # will be overridden per run
    "PREPROCESSED_CSV": "new_preprocessed/artemis_preprocessed.csv",
    "VOCAB_PATH": "new_preprocessed/vocab.pkl",
    "IMAGE_FEAT_DIR": "new_preprocessed/features",
    "RESULTS_DIR": "eval_outputs/results_cnn_lstm",

    "IMAGE_SIZE": 128,                      # used by encoder conv assumptions
    "IMAGE_FEATURE_DIM": 256,               # encoder output dim
    "HIDDEN_SIZE": 256,
    "DROPOUT_RATE": 0.2,
    "NUM_EMOTIONS": 9,
    "MAX_LEN": 25,
    "BATCH_SIZE": 32,
    "NUM_EPOCHS": 20,                        # increase (e.g., 15–20) for real training
    "LEARNING_RATE": 1e-4,
    "SEED": 42,
}

torch.manual_seed(BASE_CONFIG["SEED"])
np.random.seed(BASE_CONFIG["SEED"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shared history path across all runs
HISTORY_LOG_PATH = osp.join(BASE_CONFIG["RESULTS_DIR"], "full_experiment_history.json")


# ----------------------------------------------------
# 1. DATASET
# ----------------------------------------------------

class ArtemisCaptioningDataset(Dataset):
    """
    Uses:
      - painting (for .npy image array)
      - emotion_label (0..8)
      - token_ids_with_emotion_str: a list-string like "[1, 333, 5, ...]"
        where the emotion token is already prepended in preprocessing.
    """

    def __init__(self, csv_path, split, feature_dir):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.feature_dir = feature_dir

        self.painting_ids = self.df["painting"].tolist()
        self.emotion_labels = self.df["emotion_label"].tolist()
        self.token_id_strs = self.df["token_ids_with_emotion_str"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        painting_id = self.painting_ids[idx]
        img_feat_path = osp.join(self.feature_dir, f"{painting_id}.npy")

        arr = np.load(img_feat_path)        # (H,W,3) in [0,1]
        image_data = torch.from_numpy(arr).float()  # (H,W,3)

        token_ids_list = ast.literal_eval(self.token_id_strs[idx])
        token_ids = torch.tensor(token_ids_list, dtype=torch.long)

        emotion_label = torch.tensor(self.emotion_labels[idx], dtype=torch.long)

        # teacher forcing: input is everything except last; target is everything except first
        caption_input = token_ids[:-1]
        caption_target = token_ids[1:]

        return image_data, emotion_label, caption_input, caption_target, painting_id


# ----------------------------------------------------
# 2. CNN ENCODER
# ----------------------------------------------------

class CustomCNNEncoder(nn.Module):
    """
    Simple CNN that takes an RGB image (B,H,W,3) or (B,3,H,W) and outputs a vector
    of dimension IMAGE_FEATURE_DIM.
    """

    def __init__(self, output_dim):
        super().__init__()

        self.cnn_blocks = nn.Sequential(
            # assume input ~ 128x128; if 224x224, still works (will end up 14x14)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # 128 -> 64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # 64 -> 32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # 32 -> 16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # 16 -> 8
        )

        final_spatial_dim = 8   # if input 128; if 224 it will be different but still consistent
        final_cnn_channels = 512
        OBSERVED_FLATTENED_SIZE = final_cnn_channels * final_spatial_dim * final_spatial_dim

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(OBSERVED_FLATTENED_SIZE, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, output_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # DataLoader batches: (B,H,W,3); convert to (B,3,H,W)
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = self.cnn_blocks(x)
        x = self.fc(x)
        return x  # (B, output_dim)


# ----------------------------------------------------
# 3. EMOTION-FUSED LSTM DECODER (2 LAYERS)
# ----------------------------------------------------
class EmotionLSTMDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_size,
        num_emotions,
        image_feature_dim,
        dropout_rate,
        embedding_matrix=None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.image_feature_dim = image_feature_dim
        self.num_layers = 2   # <--- keep track of this

        # 1. Word embedding
        if embedding_matrix is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float32),
                freeze=False,
            )
            embed_dim = embedding_matrix.shape[1]
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embed_dim)

        # 2. Emotion embedding
        self.emotion_embeddings = nn.Embedding(num_emotions, embed_dim)

        self.lstm_input_size = embed_dim

        # 3. Initial state generators
        context_dim = image_feature_dim + embed_dim
        self.h0_generator = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.c0_generator = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 4. LSTM with 2 layers
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
        )

        # 5. Output layer
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.dropout_word = nn.Dropout(dropout_rate)

    def init_hidden_state(self, image_features, emotion_labels):
        """
        image_features: (B, image_feature_dim)
        emotion_labels: (B,)
        Returns:
          h0, c0: (num_layers, B, H)
        """
        emotion_vec = self.emotion_embeddings(emotion_labels)  # (B, E)
        combined = torch.cat([image_features, emotion_vec], dim=1)  # (B, D_img+E)

        base_h = self.h0_generator(combined)  # (B, H)
        base_c = self.c0_generator(combined)  # (B, H)

        # repeat for all layers
        h0 = base_h.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (L,B,H)
        c0 = base_c.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (L,B,H)
        return h0, c0

    def forward(self, image_features, emotion_labels, caption_input):
        h0, c0 = self.init_hidden_state(image_features, emotion_labels)
        word_embeds = self.word_embeddings(caption_input)  # (B,T,E)
        word_embeds = self.dropout_word(word_embeds)

        lstm_out, _ = self.lstm(word_embeds, (h0, c0))     # (B,T,H)
        logits = self.output_layer(lstm_out)               # (B,T,V)
        return logits


# ----------------------------------------------------
# 4. TRAIN / VAL
# ----------------------------------------------------

def train_one_epoch(encoder, decoder, dataloader, criterion, optimizer, pad_idx):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    total_count = 0

    for image_data, emotion_labels, caption_input, caption_target, _ in tqdm(
        dataloader, desc="Training"
    ):
        image_data = image_data.to(device)
        emotion_labels = emotion_labels.to(device)
        caption_input = caption_input.to(device)
        caption_target = caption_target.to(device)

        optimizer.zero_grad()

        image_features = encoder(image_data)  # (B, image_feature_dim)
        logits = decoder(image_features, emotion_labels, caption_input)  # (B,T,V)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            caption_target.reshape(-1),
        )
        loss.backward()
        optimizer.step()

        batch_size = image_data.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(1, total_count)


def validate_one_epoch(encoder, decoder, dataloader, criterion, pad_idx):
    encoder.eval()
    decoder.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for image_data, emotion_labels, caption_input, caption_target, _ in tqdm(
            dataloader, desc="Validation"
        ):
            image_data = image_data.to(device)
            emotion_labels = emotion_labels.to(device)
            caption_input = caption_input.to(device)
            caption_target = caption_target.to(device)

            image_features = encoder(image_data)
            logits = decoder(image_features, emotion_labels, caption_input)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                caption_target.reshape(-1),
            )

            batch_size = image_data.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

    return total_loss / max(1, total_count)


# ----------------------------------------------------
# 5. HISTORY HELPERS
# ----------------------------------------------------

def load_full_history():
    if osp.exists(HISTORY_LOG_PATH):
        try:
            with open(HISTORY_LOG_PATH, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Warning: could not decode history JSON. Starting fresh.")
            return {}
    return {}


def save_full_history(history):
    os.makedirs(BASE_CONFIG["RESULTS_DIR"], exist_ok=True)
    with open(HISTORY_LOG_PATH, "w") as f:
        json.dump(history, f, indent=2)


# ----------------------------------------------------
# 6. SINGLE-RUN EXPERIMENT FUNCTION
# ----------------------------------------------------

def run_experiment(embedding_type: str):
    """
    Run a full training + validation experiment for a single embedding type:
    - "glove"
    - "fasttext"
    - "tfidf"
    (Uses embedding_utils.get_embedding_matrix, including TF-IDF pipeline.)
    """
    # Make a fresh config per run
    CONFIG = BASE_CONFIG.copy()
    CONFIG["EMBEDDING_TYPE"] = embedding_type

    print("\n========================================================")
    print(f"🚀 STARTING EXPERIMENT: {embedding_type.upper()}")
    print("========================================================")
    print(f"Using device: {device}")
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)

    # --- Load Embeddings via embedding_utils (this includes TF-IDF logic) ---
    print(f"Loading embeddings via embedding_utils: {embedding_type}...")
    emb_matrix, emb_dim, tok2idx, idx2tok = get_embedding_matrix(
        embedding_type,
        vocab_path=CONFIG["VOCAB_PATH"],
        repr_dir=osp.dirname(CONFIG["PREPROCESSED_CSV"]),
    )

    # If random: choose an embedding dimension manually
    if embedding_type == "random":
        if emb_dim is None:
            emb_dim = 256
        emb_matrix = None

    if tok2idx is None:
        print("ERROR: Failed to load vocab or embedding matrix. Check preprocessing + paths.")
        return

    pad_idx = tok2idx.get("<pad>", 0)
    sos_idx = tok2idx.get("<start>", 1)
    eos_idx = tok2idx.get("<end>", 2)
    unk_idx = tok2idx.get("<unk>", 3)

    vocab_size = len(tok2idx)
    CONFIG["EMBEDDING_DIM"] = emb_dim

    print(f"Vocab size: {vocab_size}")
    print(f"PAD idx: {pad_idx}  SOS idx: {sos_idx}  EOS idx: {eos_idx}  UNK idx: {unk_idx}")
    print(f"Embedding dim: {emb_dim}")

    # --- Models ---
    encoder = CustomCNNEncoder(output_dim=CONFIG["IMAGE_FEATURE_DIM"]).to(device)
    decoder = EmotionLSTMDecoder(
        vocab_size=vocab_size,
        embed_dim=CONFIG["EMBEDDING_DIM"],
        hidden_size=CONFIG["HIDDEN_SIZE"],
        num_emotions=CONFIG["NUM_EMOTIONS"],
        image_feature_dim=CONFIG["IMAGE_FEATURE_DIM"],
        dropout_rate=CONFIG["DROPOUT_RATE"],
        embedding_matrix=emb_matrix,
    ).to(device)

    # --- Data ---
    train_data = ArtemisCaptioningDataset(
        CONFIG["PREPROCESSED_CSV"], "train", CONFIG["IMAGE_FEAT_DIR"]
    )
    val_data = ArtemisCaptioningDataset(
        CONFIG["PREPROCESSED_CSV"], "val", CONFIG["IMAGE_FEAT_DIR"]
    )

    train_loader = DataLoader(
        train_data,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=False,
    )

    # --- Loss & Optimizer ---
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=CONFIG["LEARNING_RATE"])

    # --- Training loop ---
    best_val_loss = float("inf")
    current_run_history = {
        "embedding_type": embedding_type,
        "train_loss_history": [],
        "val_loss_history": [],
        "best_val_epoch": -1,
        "num_epochs": CONFIG["NUM_EPOCHS"],
        "config_snapshot": CONFIG,
    }

    for epoch in range(1, CONFIG["NUM_EPOCHS"] + 1):
        print(f"\n--- Epoch {epoch}/{CONFIG['NUM_EPOCHS']} ({embedding_type}) ---")
        train_loss = train_one_epoch(
            encoder, decoder, train_loader, criterion, optimizer, pad_idx
        )
        val_loss = validate_one_epoch(
            encoder, decoder, val_loader, criterion, pad_idx
        )

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        current_run_history["train_loss_history"].append(train_loss)
        current_run_history["val_loss_history"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            current_run_history["best_val_epoch"] = epoch
            print("Saving best model checkpoint...")

            ckpt_name = f"best_model_{embedding_type}.pth"
            ckpt_path = osp.join(CONFIG["RESULTS_DIR"], ckpt_name)
            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": CONFIG,
                    "tok2idx": tok2idx,
                },
                ckpt_path,
            )

    current_run_history["final_val_loss"] = val_loss

    # --- Save full history (shared JSON) ---
    full_history = load_full_history()
    if embedding_type not in full_history:
        full_history[embedding_type] = []
    full_history[embedding_type].append(current_run_history)
    save_full_history(full_history)

    print(f"\n✅ Experiment {embedding_type.upper()} complete.")
    print(f"Final Val Loss: {val_loss:.4f}")
    print(f"Results logged to {HISTORY_LOG_PATH}")


# ----------------------------------------------------
# 7. MAIN – MULTI-LOOP OVER THREE EMBEDDINGS
# ----------------------------------------------------

def main():
    # Three embedding types to run in sequence (like Script B),
    # but using embedding_utils + 2-layer LSTM from Script A.
    EMBEDDING_TYPES = ["glove", "fasttext", "tfidf"]

    for emb_type in EMBEDDING_TYPES:
        run_experiment(emb_type)

    print("\n--------------------------------------------------------")
    print("🎉 All three embedding experiments are complete.")
    print("--------------------------------------------------------")


if __name__ == "__main__":
    main()
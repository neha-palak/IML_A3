import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "data_preprocessed"
FEATURES_DIR = os.path.join(DATA_DIR, "features")
TFIDF_DIR = os.path.join(DATA_DIR, "tfidf_npy")
CHECKPOINT_DIR = "checkpoints/m1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_TYPE = "glove"  # 'glove', 'fasttext', or 'tfidf'
EMBED_DIM = 300
HIDDEN_SIZE = 256
NUM_EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
MAX_LEN = 20

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -----------------------------
# Load vocab & embeddings
# -----------------------------
with open(os.path.join(DATA_DIR, "vocab.pkl"), "rb") as f:
    token_to_idx = pickle.load(f)
idx_to_token = {v:k for k,v in token_to_idx.items()}
VOCAB_SIZE = len(token_to_idx)
PAD_IDX = token_to_idx["<pad>"]
START_IDX = token_to_idx["<start>"]
END_IDX = token_to_idx["<end>"]

if EMBED_TYPE in ["glove", "fasttext"]:
    emb_matrix = np.load(os.path.join(DATA_DIR, f"emb_{EMBED_TYPE}_300d.npy"))
    emb_matrix = torch.tensor(emb_matrix, dtype=torch.float32)
elif EMBED_TYPE == "tfidf":
    # TF-IDF is per-row, handled in dataset
    emb_matrix = None
else:
    raise ValueError("Unsupported embedding type")

# -----------------------------
# Dataset
# -----------------------------
class ArtEmisDataset(Dataset):
    def __init__(self, csv_file, features_dir, tfidf_dir=None):
        import pandas as pd
        self.df = pd.read_csv(csv_file)
        self.features_dir = features_dir
        self.tfidf_dir = tfidf_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_feat = np.load(os.path.join(self.features_dir, f"{row['painting']}.npy"))
        img_feat = torch.tensor(img_feat, dtype=torch.float32).permute(2,0,1)

        if self.tfidf_dir:
            word_emb = np.load(os.path.join(self.tfidf_dir, f"train__{idx:06d}.npy"))
            word_emb = torch.tensor(word_emb, dtype=torch.float32)
        else:
            word_emb = torch.tensor(eval(str(row['token_ids'])), dtype=torch.long)

        emotion = torch.tensor(row['emotion_label'], dtype=torch.long)
        return img_feat, word_emb, emotion

# -----------------------------
# Model
# -----------------------------
class CNNEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(128*16*16, out_dim)  # Adjust depending on input size

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class LSTMDecoder(nn.Module):
    def __init__(self, embed_matrix, hidden_size, vocab_size, emo_dim=32, tfidf_dim=None, dropout=0.3):
        super().__init__()
        if embed_matrix is not None:
            self.embed = nn.Embedding.from_pretrained(embed_matrix, freeze=False)
            input_dim = embed_matrix.size(1)
        elif tfidf_dim is not None:
            input_dim = tfidf_dim
            self.embed = nn.Identity()
        else:
            raise ValueError("Either embed_matrix or tfidf_dim must be provided")
        self.emotion_embed = nn.Embedding(10, emo_dim)
        self.lstm = nn.LSTM(input_dim + 256 + emo_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, word_input, img_feat, emo, hidden=None):
        emb = self.embed(word_input) if not isinstance(self.embed, nn.Identity) else word_input
        emo_emb = self.emotion_embed(emo).unsqueeze(1)
        img_feat = img_feat.unsqueeze(1).repeat(1, emb.size(1),1)
        lstm_input = torch.cat([emb, img_feat, emo_emb], dim=-1)
        out, hidden = self.lstm(lstm_input, hidden)
        out = self.fc(self.dropout(out))
        return out, hidden

# -----------------------------
# Training
# -----------------------------
def train():
    dataset = ArtEmisDataset(os.path.join(DATA_DIR,"train.csv"), FEATURES_DIR, TFIDF_DIR if EMBED_TYPE=="tfidf" else None)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    encoder = CNNEncoder().to(DEVICE)
    decoder = LSTMDecoder(emb_matrix, HIDDEN_SIZE, VOCAB_SIZE, tfidf_dim=512 if EMBED_TYPE=="tfidf" else None).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)

    all_losses = []
    for epoch in range(NUM_EPOCHS):
        encoder.train(); decoder.train()
        running_loss = 0
        for img, tokens, emo in tqdm(loader, desc=f"Epoch {epoch+1}"):
            img, emo = img.to(DEVICE), emo.to(DEVICE)
            tokens = tokens.to(DEVICE) if not isinstance(tokens, torch.FloatTensor) else tokens.to(DEVICE)
            optimizer.zero_grad()
            out, _ = decoder(tokens[:,:-1], encoder(img), emo)
            loss = criterion(out.view(-1, VOCAB_SIZE), tokens[:,1:].reshape(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss/len(loader)
        all_losses.append(avg_loss)
        torch.save({
            'encoder_state': encoder.state_dict(),
            'decoder_state': decoder.state_dict(),
            'losses': all_losses
        }, os.path.join(CHECKPOINT_DIR, f"epoch{epoch+1}.pth"))
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

if __name__=="__main__":
    train()

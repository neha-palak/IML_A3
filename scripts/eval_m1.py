import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make sure directories exist
os.makedirs("checkpoints/m1", exist_ok=True)

# Hyperparameters
HIDDEN_SIZE = 256
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Assuming your dataset class returns:
# img: [3,H,W], tokens: [seq_len], emo: int, tfidf: [tfidf_dim]
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Instantiate models
encoder = CNNEncoder().to(DEVICE)  # your custom CNN returning [batch, 256]
decoder = LSTMDecoder(emb_matrix, HIDDEN_SIZE, VOCAB_SIZE, tfidf_dim=512 if EMBED_TYPE=="tfidf" else None).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)

train_losses = []
val_losses = []

for epoch in range(1, EPOCHS+1):
    encoder.train()
    decoder.train()
    total_loss = 0

    for img, tokens, emo, tfidf_feat in tqdm(train_loader, desc=f"Epoch {epoch}"):
        img = img.to(DEVICE)
        tokens = tokens.to(DEVICE)
        emo = emo.to(DEVICE)
        if tfidf_feat is not None:
            tfidf_feat = tfidf_feat.to(DEVICE)

        optimizer.zero_grad()
        img_feat = encoder(img)  # [batch, 256]

        # Forward pass
        out = decoder(tokens[:, :-1], img_feat, emo, tfidf_feat)  # predict next word
        # tokens[:, 1:] = target sequence
        loss = criterion(out.reshape(-1, VOCAB_SIZE), tokens[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    encoder.eval()
    decoder.eval()
    val_loss_total = 0
    with torch.no_grad():
        for img, tokens, emo, tfidf_feat in val_loader:
            img = img.to(DEVICE)
            tokens = tokens.to(DEVICE)
            emo = emo.to(DEVICE)
            if tfidf_feat is not None:
                tfidf_feat = tfidf_feat.to(DEVICE)

            img_feat = encoder(img)
            out = decoder(tokens[:, :-1], img_feat, emo, tfidf_feat)
            loss = criterion(out.reshape(-1, VOCAB_SIZE), tokens[:, 1:].reshape(-1))
            val_loss_total += loss.item()

    avg_val_loss = val_loss_total / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # Save checkpoint
    torch.save({
        "epoch": epoch,
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses
    }, f"checkpoints/m1/epoch_{epoch}.pt")

# Save all losses to a JSON for easy plotting later
import json
with open("checkpoints/m1/losses.json", "w") as f:
    json.dump({"train": train_losses, "val": val_losses}, f)

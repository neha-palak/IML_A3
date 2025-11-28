from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn

class ArtEmisDataset(Dataset):
    def __init__(self, df, vocab, img_dir, max_len=20, transform=None):
        self.df = df
        self.vocab = vocab
        self.img_dir = img_dir
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.img_dir}/{row['art_style']}/{row['painting']}.jpg"
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        caption_ids = torch.tensor(row['token_ids'])
        return image, caption_ids

class CNN_Encoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class LSTM_Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, feature_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + feature_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, captions, features):
        embeddings = self.embedding(captions)                  
        features = features.unsqueeze(1).repeat(1, embeddings.size(1), 1)  
        lstm_input = torch.cat([embeddings, features], dim=-1)
        outputs, _ = self.lstm(lstm_input)
        return self.fc(outputs)

    # nn.Embedding(num_embeddings=len(vocab), embedding_dim=256)
    # outputs = decoder(captions[:, :-1], cnn(images))
    # loss = criterion(outputs.view(-1, vocab_size), captions[:,1:].reshape(-1))


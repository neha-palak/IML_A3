import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os.path as osp
import os
import ast
import json
import pickle
from tqdm import tqdm
from embedding_utils import get_embedding_matrix
# Assuming 'embedding_utils.py' is a separate file that contains get_embedding_matrix
# For 'tf-idf' logic, a placeholder is needed since the original file was not provided
# We will use the definition provided in the previous turn. 

# --- CONFIGURATION ---
CONFIG = {
    "EMBEDDING_TYPE": "fasttext",         # 'glove', 'fasttext', 'tf-idf' 
    "PREPROCESSED_CSV": "new_preprocessed/artemis_preprocessed.csv",
    "VOCAB_PATH": "new_preprocessed/vocab.pkl",
    "IMAGE_FEAT_DIR": "new_preprocessed/features",
    "RESULTS_DIR": "eval_outputs/results_cnn_lstm",
    "IMAGE_SIZE": 128,                     
    "IMAGE_FEATURE_DIM": 256,            
    "EMOTION_DIM": 300,                    
    "HIDDEN_SIZE": 256,                    
    "DROPOUT_RATE": 0.5,
    "NUM_EMOTIONS": 9,                     
    "MAX_LEN": 25,                         
    "BATCH_SIZE": 32,
    "NUM_EPOCHS": 3, # Set this higher (e.g., 20) for better results
    "LEARNING_RATE": 1e-4,
    "SEED": 42
}
torch.manual_seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])

# --- CONSTANTS DERIVED FROM VOCAB ---
PAD_TOKEN_ID = 0 
START_TOKEN_ID = 1 
END_TOKEN_ID = 2
UNK_TOKEN_ID = 3 
VOCAB_SIZE = 0 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HISTORY_LOG_PATH = osp.join(CONFIG["RESULTS_DIR"], "full_experiment_history.json")

# --- PLACEHOLDER for get_embedding_matrix (for 'glove', 'fasttext') ---
# NOTE: You MUST ensure your actual embedding_utils.py contains a working get_embedding_matrix
# For this script to be fully runnable, you need this function or must implement it.
def get_embedding_matrix(embedding_type, vocab_path, repr_dir):
    """
    Placeholder for loading pre-trained embeddings (GloVe/FastText). 
    In a real scenario, this would load the embeddings and the vocabulary.
    """
    try:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
            tok2idx = vocab.token_to_idx if hasattr(vocab, 'token_to_idx') else vocab
        
        # Load embedding matrix from pre-saved file (e.g., in repr_dir)
        # --- FIX: Use the correct filename based on text_representation.py ---
        if embedding_type == 'fasttext':
            matrix_path = osp.join(repr_dir, "emb_fasttext_300d.npy")
        elif embedding_type == 'glove':
            matrix_path = osp.join(repr_dir, "emb_glove_300d.npy")
        else: # Fallback to original attempt if type is unexpected
            matrix_path = osp.join(repr_dir, f"{embedding_type}_matrix.npy")

        # Load embedding matrix from pre-saved file
        emb_matrix = np.load(matrix_path)
        emb_dim = emb_matrix.shape[1]
        
        return emb_matrix, emb_dim, tok2idx, {i: t for t, i in tok2idx.items()}
    except Exception as e:
        print(f"Error loading {embedding_type} matrix: {e}")
        return None, None, None, None

# --- TF-IDF Specific Embedding Logic (As derived in previous turn) ---
def get_tfidf_vector_matrix(vocab_path, repr_dir):
    """Loads the TF-IDF matrix where each token's embedding is its TF-IDF vector."""
    try:
        tfidf_path = osp.join(repr_dir, "tfidf_matrix.npy")
        tfidf_matrix = np.load(tfidf_path)
        
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
            tok2idx = vocab.token_to_idx if hasattr(vocab, 'token_to_idx') else vocab

        assert tfidf_matrix.shape[0] == len(tok2idx), "TF-IDF matrix size mismatch."
        
        emb_dim = tfidf_matrix.shape[1] 
        return tfidf_matrix, emb_dim, tok2idx
    
    except Exception as e:
        print(f"Error loading TF-IDF matrix: {e}")
        return None, None, None

## ----------------------------------------------------
## 1. DATASET AND DATALOADER 
## ----------------------------------------------------

class ArtemisCaptioningDataset(Dataset):
    def __init__(self, csv_path, split, feature_dir, vocab_path):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.feature_dir = feature_dir
        
        self.painting_ids = self.df['painting'].tolist()
        self.emotion_labels = self.df['emotion_label'].tolist()
        self.token_id_strs = self.df['token_ids_with_emotion_str'].tolist() 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        painting_id = self.painting_ids[idx]
        img_feat_path = osp.join(self.feature_dir, f"{painting_id}.npy")
        # Load image data (assumed to be a normalized 3D array or features)
        image_data = torch.from_numpy(np.load(img_feat_path)).float() 

        token_ids_list = ast.literal_eval(self.token_id_strs[idx])
        token_ids = torch.tensor(token_ids_list, dtype=torch.long)
        
        emotion_label = torch.tensor(self.emotion_labels[idx], dtype=torch.long)

        # Input sequence (excluding <end>)
        caption_input = token_ids[:-1] 
        # Target sequence (excluding <start>)
        caption_target = token_ids[1:] 

        return image_data, emotion_label, caption_input, caption_target, painting_id


## ----------------------------------------------------
## 2. CUSTOM CNN ENCODER 
## ----------------------------------------------------

class CustomCNNEncoder(nn.Module):
    def __init__(self, output_dim): 
        super().__init__()
        
        self.cnn_blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(), # 128 -> 64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(), # 64 -> 32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(), # 32 -> 16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.ReLU(), # 16 -> 8
        )
        
        final_spatial_dim = 8
        final_cnn_channels = 512
        OBSERVED_FLATTENED_SIZE = final_cnn_channels * final_spatial_dim * final_spatial_dim 
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(OBSERVED_FLATTENED_SIZE, 1024), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, output_dim) 
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # Permute from (B, H, W, 3) to (B, 3, H, W) if necessary
        if x.dim() == 4 and x.shape[-1] == 3:
             x = x.permute(0, 3, 1, 2) 
        x = self.cnn_blocks(x)
        x = self.fc(x)
        return x # (B, output_dim)


## ----------------------------------------------------
## 3. EMOTION-FUSED LSTM DECODER (FIXED INPUT SIZE)
## ----------------------------------------------------

class EmotionLSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_emotions, image_feature_dim, dropout_rate, embedding_matrix=None):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.image_feature_dim = image_feature_dim 
        
        # 1. Embeddings
        if embedding_matrix is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float), freeze=False
            )
            # The actual embed_dim is determined by the matrix, but we keep the parameter name
            embed_dim = embedding_matrix.shape[1] 
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        self.emotion_embeddings = nn.Embedding(num_emotions, embed_dim) 

        # FIX: Total LSTM input size is only the Word Embedding dimension.
        # This addresses the repetition issue by removing the constant image/emotion context
        # from the sequential input, relying only on the h0/c0 initialization.
        self.lstm_input_size = embed_dim 

        # 2. Initial State Generators (h0 and c0)
        # Context size: D_img + D_emo (256 + 300)
        self.h0_generator = nn.Sequential(
            nn.Linear(image_feature_dim + embed_dim, hidden_size), 
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.c0_generator = nn.Sequential(
            nn.Linear(image_feature_dim + embed_dim, hidden_size), 
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 3. LSTM
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1, #dropouts cannot be used here 
            dropout=dropout_rate if dropout_rate > 0 else 0.0
        )

        # 4. Output Layer
        self.output_layer = nn.Linear(hidden_size, vocab_size)

        self.dropout_word = nn.Dropout(dropout_rate)

    def init_hidden_state(self, image_features, emotion_labels):
        """Generates initial h0 and c0 using image features and emotion embedding."""
        emotion_vec = self.emotion_embeddings(emotion_labels) 
        combined_context = torch.cat((image_features, emotion_vec), dim=1) 
        # Unsqueeze(0) for num_layers dimension (1)
        h0 = self.h0_generator(combined_context).unsqueeze(0) 
        c0 = self.c0_generator(combined_context).unsqueeze(0) 
        return h0, c0, emotion_vec

    def forward(self, image_features, emotion_labels, caption_input):
        
        h0, c0, _ = self.init_hidden_state(image_features, emotion_labels) 
        word_embeds = self.word_embeddings(caption_input) 
        word_embeds = self.dropout_word(word_embeds)
        
        # FIX: The input to the LSTM is only the word embedding.
        fused_input = word_embeds
        
        # Pass h0 and c0 to initialize the sequence
        lstm_out, _ = self.lstm(fused_input, (h0, c0)) 

        output = self.output_layer(lstm_out)
        
        return output

## ----------------------------------------------------
## 4. TRAINING & VALIDATION FUNCTIONS
## ----------------------------------------------------

def train_one_epoch(encoder, decoder, dataloader, criterion, optimizer):
    encoder.train()
    decoder.train()
    total_loss = 0
    
    for image_data, emotion_labels, caption_input, caption_target, _ in tqdm(dataloader, desc="Training"):
        image_data, emotion_labels = image_data.to(device), emotion_labels.to(device)
        caption_input, caption_target = caption_input.to(device), caption_target.to(device)

        optimizer.zero_grad()

        image_features = encoder(image_data)
        output = decoder(image_features, emotion_labels, caption_input)
        
        loss = criterion(
            output.contiguous().view(-1, output.size(-1)), 
            caption_target.contiguous().view(-1)
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * image_data.size(0)
        
    return total_loss / len(dataloader.dataset)

def validate_one_epoch(encoder, decoder, dataloader, criterion):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    
    with torch.no_grad():
        for image_data, emotion_labels, caption_input, caption_target, _ in tqdm(dataloader, desc="Validation"):
            image_data, emotion_labels = image_data.to(device), emotion_labels.to(device)
            caption_input, caption_target = caption_input.to(device), caption_target.to(device)

            image_features = encoder(image_data)
            output = decoder(image_features, emotion_labels, caption_input)

            loss = criterion(
                output.contiguous().view(-1, output.size(-1)), 
                caption_target.contiguous().view(-1)
            )
            
            total_loss += loss.item() * image_data.size(0)
            
    return total_loss / len(dataloader.dataset)

## ----------------------------------------------------
## 5. HISTORY MANAGEMENT 
## ----------------------------------------------------

def load_full_history():
    """Loads the persistent history log."""
    if osp.exists(HISTORY_LOG_PATH):
        try:
            with open(HISTORY_LOG_PATH, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Warning: Could not decode history file. Starting new log.")
            return {}
    return {}

def save_full_history(history):
    """Saves the persistent history log."""
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    with open(HISTORY_LOG_PATH, "w") as f:
        json.dump(history, f, indent=4)

## ----------------------------------------------------
## 6. MAIN FUNCTION
## ----------------------------------------------------

def main():
    print(f"Using device: {device}")
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    
    # --- Load Embeddings ---
    embedding_type = CONFIG['EMBEDDING_TYPE']
    print(f"Loading embeddings: {embedding_type}...")
    
    if embedding_type == 'tf-idf':
        emb_matrix, emb_dim, tok2idx = get_tfidf_vector_matrix(CONFIG["VOCAB_PATH"], osp.dirname(CONFIG["PREPROCESSED_CSV"]))
        _ = None # Dummy for idx2tok
    else:
        emb_matrix, emb_dim, tok2idx, _ = get_embedding_matrix(
            embedding_type, 
            vocab_path=CONFIG["VOCAB_PATH"],
            repr_dir=osp.dirname(CONFIG["PREPROCESSED_CSV"])
        )
    
    if emb_matrix is None:
        print("ERROR: Failed to load embedding matrix. Ensure pre-processing is complete.")
        return

    # Update global constants and CONFIG
    global VOCAB_SIZE
    VOCAB_SIZE = len(tok2idx)
    CONFIG["EMBEDDING_DIM"] = emb_dim 
    
    # --- Initialize Models ---
    encoder = CustomCNNEncoder(output_dim=CONFIG["IMAGE_FEATURE_DIM"]).to(device)
    decoder = EmotionLSTMDecoder(
        vocab_size=VOCAB_SIZE,
        embed_dim=CONFIG["EMBEDDING_DIM"],
        hidden_size=CONFIG["HIDDEN_SIZE"],
        num_emotions=CONFIG["NUM_EMOTIONS"],
        image_feature_dim=CONFIG["IMAGE_FEATURE_DIM"],
        dropout_rate=CONFIG["DROPOUT_RATE"],
        embedding_matrix=emb_matrix
    ).to(device)
    
    # --- DataLoaders ---
    train_data = ArtemisCaptioningDataset(CONFIG["PREPROCESSED_CSV"], 'train', CONFIG["IMAGE_FEAT_DIR"], CONFIG["VOCAB_PATH"])
    val_data = ArtemisCaptioningDataset(CONFIG["PREPROCESSED_CSV"], 'val', CONFIG["IMAGE_FEAT_DIR"], CONFIG["VOCAB_PATH"])
    train_loader = DataLoader(train_data, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)
    
    # --- Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=CONFIG["LEARNING_RATE"])
    
    # --- Training Loop ---
    best_val_loss = float('inf')
    current_run_history = {"train_loss_history": [], "val_loss_history": [], "best_val_epoch": -1}
    
    for epoch in range(1, CONFIG["NUM_EPOCHS"] + 1):
        print(f"\n--- Epoch {epoch}/{CONFIG['NUM_EPOCHS']} ---")
        train_loss = train_one_epoch(encoder, decoder, train_loader, criterion, optimizer)
        val_loss = validate_one_epoch(encoder, decoder, val_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        current_run_history["train_loss_history"].append(train_loss)
        current_run_history["val_loss_history"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            current_run_history["best_val_epoch"] = epoch
            print("Saving best model checkpoint...")
            
            checkpoint_filename = f"best_model_{embedding_type}.pth"
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'best_val_loss': best_val_loss,
                'config': CONFIG
            }, osp.join(CONFIG["RESULTS_DIR"], checkpoint_filename))

    # --- History Summary and Persistence ---
    current_run_history["embedding_type"] = embedding_type
    current_run_history["num_epochs"] = CONFIG["NUM_EPOCHS"]
    current_run_history["final_val_loss"] = val_loss
    
    full_history = load_full_history()
    
    if embedding_type not in full_history:
        full_history[embedding_type] = []
        
    full_history[embedding_type].append(current_run_history)
    
    save_full_history(full_history)
    print("Training complete.")
    print(f"Results for this run added to history log at {HISTORY_LOG_PATH}")

if __name__ == "__main__":
    main()
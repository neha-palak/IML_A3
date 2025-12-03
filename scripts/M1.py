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

# Import necessary functions from the utility file
# NOTE: This assumes embedding_utils.py is in the same directory or accessible via PYTHONPATH.
from embedding_utils import get_embedding_matrix, load_vocab 

# --- CONFIGURATION (Defaulted to a safe value) ---
GLOBAL_CONFIG = {
    "EMBEDDING_TYPE": "glove",           # This will be overridden in the loop
    "PREPROCESSED_CSV": "new_preprocessed/artemis_preprocessed.csv",
    "VOCAB_PATH": "new_preprocessed/vocab.pkl",
    "IMAGE_FEAT_DIR": "new_preprocessed/features",
    "RESULTS_DIR": "eval_outputs/results_cnn_lstm",
    "IMAGE_SIZE": 128,                   # Input image size (H=W)
    "IMAGE_FEATURE_DIM": 256,            # Output dimension of CNN Encoder
    "EMOTION_DIM": 300,                  # Used for Embedding size if no pre-trained matrix loads
    "HIDDEN_SIZE": 256,                  # LSTM hidden size
    "DROPOUT_RATE": 0.2,
    "NUM_EMOTIONS": 9,
    "MAX_LEN": 25,                       # Max sequence length (including <start>/<end>)
    "BATCH_SIZE": 32,
    "NUM_EPOCHS": 5,
    "LEARNING_RATE": 1e-4,
    "SEED": 42
}

torch.manual_seed(GLOBAL_CONFIG["SEED"])
np.random.seed(GLOBAL_CONFIG["SEED"])

# --- CONSTANTS DERIVED FROM VOCAB ---
PAD_TOKEN_ID = 0
START_TOKEN_ID = 1
END_TOKEN_ID = 2
UNK_TOKEN_ID = 3
VOCAB_SIZE = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HISTORY_LOG_PATH = osp.join(GLOBAL_CONFIG["RESULTS_DIR"], "full_experiment_history.json")


# ----------------------------------------------------
# 1. DATASET AND DATALOADER 
# ----------------------------------------------------

class ArtemisCaptioningDataset(Dataset):
    def __init__(self, csv_path, split, feature_dir, vocab_path):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.feature_dir = feature_dir
        
        self.painting_ids = self.df['painting'].tolist()
        self.emotion_labels = self.df['emotion_label'].tolist()
        # NOTE: Assumes the preprocessed CSV has the column 'token_ids' 
        self.token_id_strs = self.df['token_ids'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).tolist()


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        painting_id = self.painting_ids[idx]
        img_feat_path = osp.join(self.feature_dir, f"{painting_id}.npy")
        
        # Load image data (assumed to be a normalized 3D array HxWx3)
        try:
            image_data = torch.from_numpy(np.load(img_feat_path)).float() 
        except FileNotFoundError:
            print(f"Warning: Image feature file not found for {painting_id}")
            # Use a zero tensor as a placeholder for missing images
            image_data = torch.zeros((GLOBAL_CONFIG["IMAGE_SIZE"], GLOBAL_CONFIG["IMAGE_SIZE"], 3), dtype=torch.float)

        token_ids_list = self.token_id_strs[idx]
        token_ids = torch.tensor(token_ids_list, dtype=torch.long)
        
        emotion_label = torch.tensor(self.emotion_labels[idx], dtype=torch.long)

        # Input sequence (tokens 0 to L-1, i.e., from <start> up to the token before <end>)
        caption_input = token_ids[:-1] 
        # Target sequence (tokens 1 to L, i.e., from the first word up to <end>)
        caption_target = token_ids[1:] 

        return image_data, emotion_label, caption_input, caption_target, painting_id

# ----------------------------------------------------
# 2. CUSTOM CNN ENCODER 
# ----------------------------------------------------

class CustomCNNEncoder(nn.Module):
    def __init__(self, output_dim, input_size=GLOBAL_CONFIG["IMAGE_SIZE"]):
        super().__init__()
        
        # Calculate expected output size for the flattened layer
        final_spatial_dim = input_size // (2**4) # 128 -> 8
        final_cnn_channels = 512
        OBSERVED_FLATTENED_SIZE = final_cnn_channels * final_spatial_dim * final_spatial_dim 
        
        self.cnn_blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(OBSERVED_FLATTENED_SIZE, 1024), 
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(1024, output_dim) 
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Permute from (B, H, W, 3) to (B, 3, H, W) 
        if x.dim() == 4 and x.shape[-1] == 3:
              x = x.permute(0, 3, 1, 2) 
        x = self.cnn_blocks(x)
        x = self.fc(x)
        return x # (B, output_dim)

# ----------------------------------------------------
# 3. EMOTION-FUSED LSTM DECODER
# ----------------------------------------------------

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
            # Use the actual dimension from the loaded matrix
            embed_dim = embedding_matrix.shape[1] 
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        self.emotion_embeddings = nn.Embedding(num_emotions, embed_dim) 

        self.lstm_input_size = embed_dim 

        # 2. Initial State Generators (h0 and c0)
        init_gen_input_dim = image_feature_dim + embed_dim 
        
        self.h0_generator = nn.Sequential(
            nn.Linear(init_gen_input_dim, hidden_size), 
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.c0_generator = nn.Sequential(
            nn.Linear(init_gen_input_dim, hidden_size), 
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 3. LSTM
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2, #1 didnt work the best 
            dropout=dropout_rate if dropout_rate > 0 else 0.0
        )

        # 4. Output Layer
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.dropout_word = nn.Dropout(dropout_rate)

    def init_hidden_state(self, image_features, emotion_labels):
        """Generates initial h0 and c0 using image features and emotion embedding."""
        emotion_vec = self.emotion_embeddings(emotion_labels) 
        combined_context = torch.cat((image_features, emotion_vec), dim=1) 
        h0 = self.h0_generator(combined_context).unsqueeze(0) 
        c0 = self.c0_generator(combined_context).unsqueeze(0) 
        return h0, c0, emotion_vec

    def forward(self, image_features, emotion_labels, caption_input):
        
        h0, c0, _ = self.init_hidden_state(image_features, emotion_labels) 
        word_embeds = self.word_embeddings(caption_input) 
        word_embeds = self.dropout_word(word_embeds)
        
        fused_input = word_embeds
        
        lstm_out, _ = self.lstm(fused_input, (h0, c0)) 

        output = self.output_layer(lstm_out)
        
        return output

# ----------------------------------------------------
# 4. TRAINING & VALIDATION FUNCTIONS
# ----------------------------------------------------

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
        
        # Flatten and calculate loss
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


# ----------------------------------------------------
# 5. HISTORY MANAGEMENT
# ----------------------------------------------------

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
    os.makedirs(GLOBAL_CONFIG["RESULTS_DIR"], exist_ok=True)
    with open(HISTORY_LOG_PATH, "w") as f:
        json.dump(history, f, indent=4)

# ----------------------------------------------------
# 6. MAIN EXPERIMENT FUNCTION
# ----------------------------------------------------

def run_experiment(embedding_type):
    """Runs the training and validation process for a specific embedding type."""
    
    # 1. Update Configuration for the current run
    current_config = GLOBAL_CONFIG.copy()
    current_config["EMBEDDING_TYPE"] = embedding_type
    
    print(f"\n========================================================")
    print(f"🚀 STARTING EXPERIMENT: {embedding_type.upper()}")
    print(f"========================================================")
    print(f"Using device: {device}")
    os.makedirs(current_config["RESULTS_DIR"], exist_ok=True)
    
    # 2. Load Embeddings (Using the function imported from embedding_utils.py)
    print(f"Loading embeddings: {embedding_type}...")
    
    # Use the imported get_embedding_matrix which handles all types ('glove', 'fasttext', 'tfidf')
    emb_matrix, emb_dim, tok2idx, _ = get_embedding_matrix(
        embedding_type, 
        vocab_path=current_config["VOCAB_PATH"],
        repr_dir=osp.dirname(current_config["PREPROCESSED_CSV"])
    )
    
    # Check for loading failure (matrix is None) for non-random types
    if emb_matrix is None and embedding_type != 'random':
        print(f"ERROR: Failed to load {embedding_type} embedding matrix. Skipping run.")
        return

    # 3. Update Run-specific Constants
    global VOCAB_SIZE
    
    # Ensure vocab is loaded to get VOCAB_SIZE even if emb_matrix is None ('random')
    if tok2idx is None:
        tok2idx, _ = load_vocab(current_config["VOCAB_PATH"])
        if tok2idx is None:
             print("FATAL ERROR: Could not load vocabulary. Exiting.")
             return

    VOCAB_SIZE = len(tok2idx)
    
    # Set the final embedding dimension
    if emb_matrix is not None:
         current_config["EMBEDDING_DIM"] = emb_dim 
    else: # For 'random'
         current_config["EMBEDDING_DIM"] = current_config.get("EMOTION_DIM", 300) 

    # 4. Initialize Models
    encoder = CustomCNNEncoder(output_dim=current_config["IMAGE_FEATURE_DIM"]).to(device)
    decoder = EmotionLSTMDecoder(
        vocab_size=VOCAB_SIZE,
        embed_dim=current_config["EMBEDDING_DIM"],
        hidden_size=current_config["HIDDEN_SIZE"],
        num_emotions=current_config["NUM_EMOTIONS"],
        image_feature_dim=current_config["IMAGE_FEATURE_DIM"],
        dropout_rate=current_config["DROPOUT_RATE"],
        embedding_matrix=emb_matrix
    ).to(device)
    
    # 5. DataLoaders
    train_data = ArtemisCaptioningDataset(current_config["PREPROCESSED_CSV"], 'train', current_config["IMAGE_FEAT_DIR"], current_config["VOCAB_PATH"])
    val_data = ArtemisCaptioningDataset(current_config["PREPROCESSED_CSV"], 'val', current_config["IMAGE_FEAT_DIR"], current_config["VOCAB_PATH"])
    train_loader = DataLoader(train_data, batch_size=current_config["BATCH_SIZE"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=current_config["BATCH_SIZE"], shuffle=False)
    
    # 6. Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=current_config["LEARNING_RATE"])
    
    # 7. Training Loop
    best_val_loss = float('inf')
    current_run_history = {"embedding_type": embedding_type, "train_loss_history": [], "val_loss_history": [], "best_val_epoch": -1, "config_snapshot": current_config}
    
    for epoch in range(1, current_config["NUM_EPOCHS"] + 1):
        print(f"\n--- Epoch {epoch}/{current_config['NUM_EPOCHS']} ({embedding_type}) ---")
        train_loss = train_one_epoch(encoder, decoder, train_loader, criterion, optimizer)
        val_loss = validate_one_epoch(encoder, decoder, val_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        current_run_history["train_loss_history"].append(train_loss)
        current_run_history["val_loss_history"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            current_run_history["best_val_epoch"] = epoch
            print(f"Saving best model checkpoint for {embedding_type}...")
            
            checkpoint_filename = f"best_model_{embedding_type}.pth"
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'best_val_loss': best_val_loss,
                'config': current_config
            }, osp.join(current_config["RESULTS_DIR"], checkpoint_filename))

    # 8. History Summary and Persistence
    current_run_history["num_epochs"] = current_config["NUM_EPOCHS"]
    current_run_history["final_val_loss"] = val_loss
    
    full_history = load_full_history()
    
    if embedding_type not in full_history:
        full_history[embedding_type] = []
        
    full_history[embedding_type].append(current_run_history)
    
    save_full_history(full_history)
    print(f"\nExperiment {embedding_type.upper()} complete. Final Val Loss: {val_loss:.4f}")
    print(f"Results logged to {HISTORY_LOG_PATH}")

def main():
    # FIXED: The name "tf-idf" must be "tfidf" to match the logic in embedding_utils.py
    EMBEDDING_TYPES = ["glove", "fasttext", "tfidf"] 
    
    for emb_type in EMBEDDING_TYPES:
        run_experiment(emb_type)
        
    print("\n--------------------------------------------------------")
    print("✅ All three embedding experiments are complete.")
    print("--------------------------------------------------------")

if __name__ == "__main__":
    main()
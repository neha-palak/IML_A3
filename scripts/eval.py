import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os.path as osp
import json
from collections import defaultdict
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction 
import os

# --- Import models and dataset from M1.py (Ensure M1.py contains the correct definitions) ---
from M1 import (
    CustomCNNEncoder, 
    EmotionLSTMDecoder, 
    ArtemisCaptioningDataset, 
    get_embedding_matrix, 
    CONFIG, 
    device, 
    PAD_TOKEN_ID, 
    START_TOKEN_ID, 
    END_TOKEN_ID, 
    VOCAB_SIZE,
    get_tfidf_vector_matrix 
)

# Define the file path for history storage
HISTORY_PATH = osp.join(CONFIG["RESULTS_DIR"], "evaluation_history.json")
# Define the path for the full prediction/GT output
FULL_PRED_PATH = osp.join(CONFIG["RESULTS_DIR"], "all_predictions_with_gt.json")

# Global function to load vocab
def load_vocab_dict(vocab_path):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    # Handle both old (dict) and new (class) vocab formats
    tok2idx = vocab.token_to_idx if hasattr(vocab, "token_to_idx") else vocab
    idx2tok = {i: t for t, i in tok2idx.items()}
    return tok2idx, idx2tok

## ----------------------------------------------------
## 1. CAPTION GENERATION (BEAM SEARCH)
## ----------------------------------------------------

def beam_search_decode(encoder, decoder, image_data, emotion_label, max_len, idx2tok, start_token_id, end_token_id, emotion_word_ids, beam_size=5):
    """
    Performs beam search decoding for a single image and emotion pair.
    
    FIXED: The input to the LSTM step now correctly uses only the word embedding.
    """
    encoder.eval()
    decoder.eval()
    
    # 1. Get Image Features (Pass single item)
    image_features = encoder(image_data.unsqueeze(0).to(device)) # (1, D_img)
    emotion_label_tensor = emotion_label.unsqueeze(0).to(device) # (1)
    
    # 2. Initialize LSTM state
    h, c, _ = decoder.init_hidden_state(image_features, emotion_label_tensor) 
    
    # Beams: [(log_prob, [word_ids], h, c)]
    beams = [(0.0, [start_token_id], h, c)] 
    final_captions = []
    
    for _ in range(max_len):
        candidates = []
        
        for score, current_caption, h_t, c_t in beams:
            last_word_id = current_caption[-1]
            
            if last_word_id == end_token_id: 
                # Normalize the score by length (length penalty/bonus)
                final_captions.append((score / len(current_caption), current_caption)) 
                continue 
            
            # 3. Single LSTM Step
            input_token = torch.tensor([[last_word_id]], device=device) # (1, 1)
            
            # Use the decoder's logic to get word embedding and apply dropout
            word_embed = decoder.word_embeddings(input_token) # (1, 1, EmbedDim)
            word_embed = decoder.dropout_word(word_embed)

            # --- Input to the LSTM must ONLY be the word embedding (FIXED) ---
            fused_input = word_embed 

            # Pass the word embedding and the previous hidden state
            output, (h_next, c_next) = decoder.lstm(fused_input, (h_t, c_t))
            
            log_probs = F.log_softmax(decoder.output_layer(output.squeeze(1)), dim=-1) # (1, VocabSize)
            
            topk_log_probs, topk_ids = log_probs.topk(beam_size)
            
            for i in range(beam_size):
                word_id = topk_ids.squeeze(0)[i].item()
                log_prob = topk_log_probs.squeeze(0)[i].item()
                
                new_score = score + log_prob
                new_caption = current_caption + [word_id]
                
                # Clone the states for each new candidate path
                h_next_clone = h_next.clone() 
                c_next_clone = c_next.clone()
                
                candidates.append((new_score, new_caption, h_next_clone, c_next_clone))

        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]
        
    # Add remaining beams to final_captions (applying length normalization here as well)
    final_captions.extend([(s / len(c), c) for s, c, _, _ in beams])
    final_captions.sort(key=lambda x: x[0], reverse=True)
    
    if not final_captions:
        return "Caption generation failed."
        
    best_caption_ids = final_captions[0][1]
    
    # Remove special tokens and the initial emotion word
    caption_tokens = []
    
    # Skip <start> token (index 0)
    # The first generated word is at index 1
    
    # Track if the first non-special token has been processed
    first_token_processed = False 
    
    for i, word_id in enumerate(best_caption_ids[1:]): # Start checking from the first generated token
        if word_id in [end_token_id, PAD_TOKEN_ID]:
            break
        
        # --- FIX: Remove the emotion word if it's the first generated token ---
        if not first_token_processed and word_id in emotion_word_ids:
            # Skip this token (the predicted emotion word)
            first_token_processed = True
            continue
        
        caption_tokens.append(idx2tok.get(word_id, '<UNK>'))
        first_token_processed = True # Mark that we've passed the potential emotion word position
    
    return " ".join(caption_tokens)


## ----------------------------------------------------
## 2. MAIN EVALUATION FUNCTION
## ----------------------------------------------------

def load_history():
    if osp.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_history(history):
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=4)

def save_full_predictions(predictions, gts):
    """Saves predictions and one ground truth per sample to a separate file."""
    full_output = []
    for pred in predictions:
        image_id = pred['image_id']
        # Get the list of all ground truths for this sample
        gt_list = gts.get(image_id, [])
        
        # Extract the first ground truth for display/storage
        first_gt = gt_list[0]['caption'] if gt_list else "N/A"
        
        full_output.append({
            "image_id": image_id,
            "emotion_target": pred['emotion'],
            "predicted_caption": pred['caption'],
            "ground_truth_sample": first_gt,
            "all_ground_truths": [item['caption'] for item in gt_list]
        })
    
    with open(FULL_PRED_PATH, 'w') as f:
        json.dump(full_output, f, indent=4)
    print(f"Saved full prediction and GT details to {FULL_PRED_PATH}")


def evaluate_model():
    print(f"Using device: {device}")
    
    tok2idx, idx2tok = load_vocab_dict(CONFIG["VOCAB_PATH"])
    
    # Try to dynamically load the checkpoint based on the config's EMBEDDING_TYPE
    checkpoint_filename = f"best_model_{CONFIG['EMBEDDING_TYPE']}.pth"
    checkpoint_path = osp.join(CONFIG["RESULTS_DIR"], checkpoint_filename)

    # Fallback to generic name
    if not osp.exists(checkpoint_path):
        checkpoint_path = osp.join(CONFIG["RESULTS_DIR"], "best_model.pth")

    if not osp.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}. Run training first.")
        return

    # Load checkpoint
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Re-initialize models using saved config
    current_config = checkpoint['config']
    
    # Load embeddings based on saved config
    if current_config['EMBEDDING_TYPE'] == 'tf-idf':
        # Need to ensure get_tfidf_vector_matrix is imported and correctly configured in M1.py
        emb_matrix, emb_dim, _ = get_tfidf_vector_matrix(current_config["VOCAB_PATH"], osp.dirname(current_config["PREPROCESSED_CSV"]))
    else:
        emb_matrix, emb_dim, _, _ = get_embedding_matrix(
            current_config["EMBEDDING_TYPE"], 
            vocab_path=current_config["VOCAB_PATH"],
            repr_dir=osp.dirname(current_config["PREPROCESSED_CSV"])
        )
    
    current_config["EMBEDDING_DIM"] = emb_dim if emb_dim is not None else current_config["EMOTION_DIM"]

    encoder = CustomCNNEncoder(output_dim=current_config["IMAGE_FEATURE_DIM"]).to(device)
    decoder = EmotionLSTMDecoder(
        vocab_size=len(tok2idx),
        embed_dim=current_config["EMBEDDING_DIM"],
        hidden_size=current_config["HIDDEN_SIZE"],
        num_emotions=current_config["NUM_EMOTIONS"],
        image_feature_dim=current_config["IMAGE_FEATURE_DIM"],
        dropout_rate=current_config["DROPOUT_RATE"],
        embedding_matrix=emb_matrix
    ).to(device)

    try:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
    except RuntimeError as e:
        print(f"CRITICAL ERROR loading Encoder state dict: {e}")
        print("ACTION REQUIRED: Ensure CustomCNNEncoder in M1.py includes nn.Dropout(0.5) to match the saved checkpoint.")
        return
        
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"Loaded model from Val Loss: {val_loss:.4f}")

    # Prepare Emotion ID set for filtering
    EMOTION_WORDS = ["amusement", "contentment", "awe", "excitement", "fear", "anger", "sadness", "disgust", "something else"]
    EMOTION_WORD_IDS = {tok2idx.get(w) for w in EMOTION_WORDS if tok2idx.get(w) is not None}
    
    # Load test data
    test_data = ArtemisCaptioningDataset(
        CONFIG["PREPROCESSED_CSV"], 'test', CONFIG["IMAGE_FEAT_DIR"], CONFIG["VOCAB_PATH"]
    )
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False) 
    
    predictions = []
    gts = defaultdict(list) 
    EMOTION_ID_TO_WORD = {i: t for t, i in zip(EMOTION_WORDS, range(current_config["NUM_EMOTIONS"]))}

    print("Generating captions for the test set...")
    
    for image_data, emotion_labels, _, caption_target_tokens, painting_id in tqdm(test_loader, desc="Generating Captions"):
        
        emotion_label = emotion_labels.squeeze(0).item() 
        image_data = image_data.squeeze(0)
        
        # 1. Generate Caption using beam search
        predicted_caption_str = beam_search_decode(
            encoder, decoder, 
            image_data, 
            torch.tensor(emotion_label, dtype=torch.long), 
            max_len=CONFIG["MAX_LEN"], 
            idx2tok=idx2tok,
            start_token_id=START_TOKEN_ID, 
            end_token_id=END_TOKEN_ID,
            emotion_word_ids=EMOTION_WORD_IDS, 
            beam_size=5
        )
        
        # 2. Get Ground Truths (from the original clean utterance column 'utter_clean')
        gt_df = test_data.df[(test_data.df['painting'] == painting_id[0]) & (test_data.df['emotion_label'] == emotion_label)]
        ground_truth_strs = gt_df['utter_clean'].tolist()
        
        # 3. Store Results
        result_key = f"{painting_id[0]}_{emotion_label}"
        predictions.append({
            "image_id": result_key,
            "caption": predicted_caption_str,
            "emotion": EMOTION_ID_TO_WORD.get(emotion_label, "Unknown")
        })
        
        for j, gt_text in enumerate(ground_truth_strs):
            gts[result_key].append({
                "image_id": result_key,
                "cap_id": j,
                "caption": gt_text
            })

    # Save predictions JSON (required format for pycocoevaltool)
    pred_path = osp.join(CONFIG["RESULTS_DIR"], "predictions.json")
    with open(pred_path, 'w') as f:
        json.dump(predictions, f, indent=4)
    print(f"Saved predictions (PyCOCO format) to {pred_path}")

    # Save ground truth JSON (required format for pycocoevaltool)
    gt_list = [item for sublist in gts.values() for item in sublist]
    gt_path = osp.join(CONFIG["RESULTS_DIR"], "ground_truths.json")
    with open(gt_path, 'w') as f:
        json.dump({"annotations": gt_list}, f, indent=4)

    # 4. Save full prediction output with emotion and GT 
    save_full_predictions(predictions, gts)
        
    ## 5. METRIC CALCULATION (Simplified NLTK BLEU-4)
    print("\n--- Evaluation Metrics ---")
    chencherry = SmoothingFunction()
    total_bleu4 = 0
    count = 0
    
    for pred in predictions:
        img_id = pred['image_id']
        pred_tokens = pred['caption'].split()
        references = [ann['caption'].split() for ann in gts[img_id]]
        
        if references and pred_tokens:
            total_bleu4 += sentence_bleu(references, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
            count += 1
            
    if count > 0:
        avg_bleu4 = total_bleu4 / count
        print(f"**Simplified Average NLTK BLEU-4:** {avg_bleu4:.4f}")
    else:
        avg_bleu4 = 0.0

    print("\nTo get official metrics (ROUGE-L, CIDEr), use the COCO Evaluation toolkit on the saved JSON files.")

    ## 6. HISTORY STORAGE
    eval_history = load_history()
    embedding_key = current_config["EMBEDDING_TYPE"]
    
    # Store the results under the embedding type used
    if embedding_key not in eval_history:
        eval_history[embedding_key] = []
        
    eval_history[embedding_key].append({
        "timestamp": pd.Timestamp.now().isoformat(),
        "val_loss": val_loss,
        "bleu_4": avg_bleu4,
        "image_feature_dim": current_config["IMAGE_FEATURE_DIM"],
        "hidden_size": current_config["HIDDEN_SIZE"],
        "embedding_type": embedding_key
    })
    
    save_history(eval_history)
    print(f"\nEvaluation history saved to {HISTORY_PATH}")

    ## 7. DISPLAY SAMPLES
    print("\n--- Sample Captions ---")
    num_samples = min(5, len(predictions))
    
    for i in range(num_samples):
        sample = predictions[i]
        gt_sample = gts[sample['image_id']][0]['caption']
        print(f"Image ID: {sample['image_id']}")
        print(f"Emotion Target: {sample['emotion']}")
        print(f"Predicted Caption: **{sample['caption']}**")
        print(f"Ground Truth:      {gt_sample}")
        print("-" * 30)

if __name__ == "__main__":
    evaluate_model()
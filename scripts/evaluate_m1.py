import argparse
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import json
from model1_cnn import CaptionModel   # your custom model builder
from tensorflow import keras

load_model = keras.models.load_model

def generate_caption(model, tokenizer, img_features, max_len=20):
    """Greedy decoding."""
    word_to_id = tokenizer.word_index
    id_to_word = {v:k for k,v in word_to_id.items()}

    seq = ["<start>"]

    for _ in range(max_len):
        seq_ids = [word_to_id.get(w, word_to_id["<unk>"]) for w in seq]
        seq_padded = np.pad(seq_ids, (0, max_len-len(seq_ids)))

        preds = model.predict([img_features, np.array([seq_padded])], verbose=0)
        next_id = preds.argmax()

        next_word = id_to_word.get(next_id, "<unk>")
        if next_word == "<end>":
            break

        seq.append(next_word)

    return " ".join(seq[1:])


def compute_bleu(ref, hyp):
    chencherry = SmoothingFunction()
    return {
        "BLEU-1": sentence_bleu([ref.split()], hyp.split(), weights=(1,0,0,0),
                                smoothing_function=chencherry.method1),
        "BLEU-2": sentence_bleu([ref.split()], hyp.split(), weights=(0.5,0.5,0,0),
                                smoothing_function=chencherry.method1),
        "BLEU-4": sentence_bleu([ref.split()], hyp.split(), weights=(0.25,0.25,0.25,0.25),
                                smoothing_function=chencherry.method1),
    }


def evaluate_set(model, tokenizer, features, references, max_len=20):
    rouge = Rouge()
    
    bleu_scores = {"BLEU-1": [], "BLEU-2": [], "BLEU-4": []}
    rouge_scores = []

    for img_id in features:
        cap = generate_caption(model, tokenizer, np.array([features[img_id]]), max_len)
        ref = references[img_id]

        # BLEU
        bleu_dict = compute_bleu(ref, cap)
        for k in bleu_dict:
            bleu_scores[k].append(bleu_dict[k])

        # ROUGE-L
        rouge_out = rouge.get_scores(cap, ref)[0]["rouge-l"]["f"]
        rouge_scores.append(rouge_out)

    return {
        "BLEU-1": np.mean(bleu_scores["BLEU-1"]),
        "BLEU-2": np.mean(bleu_scores["BLEU-2"]),
        "BLEU-4": np.mean(bleu_scores["BLEU-4"]),
        "ROUGE-L": np.mean(rouge_scores),
    }


def evaluate_all_embeddings(args):
    embedding_types = ["random", "glove", "fasttext"]

    # Load saved tokenizer
    tokenizer = np.load("data_preprocessed/tokenizer.npy", allow_pickle=True).item()

    # Load image features
    test_features = np.load("data_preprocessed/test_features.npy", allow_pickle=True).item()
    references = np.load("data_preprocessed/test_captions.npy", allow_pickle=True).item()

    results = {}

    for emb in embedding_types:
        print(f"\n🔵 Evaluating model with embedding: {emb.upper()}")

        model_path = f"checkpoints/m1_{emb}.keras"
        model = load_model(model_path)

        scores = evaluate_set(model, tokenizer, test_features, references, max_len=args.max_len)
        results[emb] = scores

        print(scores)

    print("\n==============================")
    print("FINAL COMPARISON TABLE")
    print("==============================")
    for emb, scores in results.items():
        print(f"\n{emb.upper()}:")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=20)
    args = parser.parse_args()
    evaluate_all_embeddings(args)

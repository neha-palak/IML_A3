ArtEmis — Emotion-Conditioned Image Captioning

This project implements two multimodal caption-generation models trained on the ArtEmis dataset:

Model-1: CNN + Emotion-Aware LSTM

Model-2: Vision-Language Transformer (VLT)

Each model supports three text embedding strategies:
✔ GloVe (300d)
✔ FastText (300d)
✔ TF-IDF (SVD-reduced)

The project includes full preprocessing, training, evaluation, prediction, and attention visualization tools.


1. Setup & Installation

Clone and install

2. Dataset Preprocessing

Run this for preprocessing
```
python3 scripts/preprocessing.py \
  --raw-csv artemis_dataset.csv \
  --wiki-root wikiart \
  --out-dir new_preprocessed \
  --copy-images \
  --subsample-size 5500 \
  --subsample-by-style
```
Generated files:
```
new_preprocessed/
    artemis_preprocessed.csv
    vocab.pkl
    emb_glove_300d.npy
    emb_fasttext_300d.npy
    tfidf_vectorizer.pkl
    tfidf_svd.pkl
    features/<painting>.npy
    images_subset/<painting>.jpg
```

3. Training

3.1 CNN + Emotion-Aware LSTM (Model-1)

Trains all three embeddings automatically:
```
python3 scripts/new_m1.py
```
Checkpoints saved here:
```
eval_outputs/results_cnn_lstm/
    best_model_glove.pth
    best_model_fasttext.pth
    best_model_tfidf.pth
    full_experiment_history.json
```
3.2 Vision-Language Transformer (Model-2)

Train with chosen embedding (glove, fasttext, tfidf)
```
python3 scripts/M2_transformers.py \
  --embedding-type {emb_type}
```
Checkpoints saved:
```
new_checkpoints/<embedding>/m2_<embedding>_best.pt
```

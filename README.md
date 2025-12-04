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
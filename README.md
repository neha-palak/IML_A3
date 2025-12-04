ArtEmis — Emotion-Conditioned Image Captioning

This project implements two multimodal caption-generation models trained on the ArtEmis dataset:

Model-1: CNN + Emotion-Aware LSTM

Model-2: Vision-Language Transformer (VLT)

Each model supports three text embedding strategies:
✔ GloVe (300d)
✔ FastText (300d)
✔ TF-IDF (SVD-reduced)

The project includes full preprocessing, training, evaluation, prediction, and attention visualization tools.

REPO STRUCTURE
* ```eval_outputs/results_cnn_lstm```: contains best CNN+LSTM model for each embedding type and training history, summary, metrics
* ``` new_checkpoints```: contains a folder for each embedding type, which contains the best model and training history for the respective embedding type. It has overall the model metrics

2. Dataset Preprocessing

Image:
* Performed stratified subsampling per art style to get 5500 images
* The 5500 images were saved to new_preprocessed/images_subset
* Images were then resized to 128x128
* Pizel normalized to 0,1 and saved as .npy files in new_preprocessed/features

Text:
* removed punctuation, converted into lowercase
* tokenized
* Later, we embedded the token vectors with emotions to feed into the models

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
4. Evaluation
1: cnn + lstm
2: transformer

```
python scripts/eval_m1.py
python scripts/eval_m2.py


```
Outputs:
```
eval_outputs/results_cnn_lstm/m1_eval_summary.json
new_checkpoints/m2_eval_summary.json
```
Metrics:
	•	BLEU-4
    •	BLEU-1
	•	ROUGE-1 F
	•	ROUGE-L F
	•	5 clean sample predictions per embedding

5. Predict

```ArtEmis_Caption_Generation.ipynb``` calls the ```predict_single_image``` function from ```viva_predict.py``` file, which we created to generate captions for any images, and not just images from our subsampled folder.

```predict.py``` can only be used to generate captions for images in the ```new_preprocessed/images_subset``` folder

Steps to generate captions:
* Inside the ```ArtEmis_Caption_Generation.ipynb```, scroll to the 'FINAL CAPTION GENERATION' section
* Save the desired image in 'viva_images' folder as .jpg
* Put the name of the image in 'IMAGE_NAME' variable
* Imput the desired EMOTION_ID
* Run the script
* It generated captions for both the models per embedding type

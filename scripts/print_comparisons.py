#!/usr/bin/env python3
"""
scripts/print_comparisons.py

Read eval_outputs/m2_*_samples.csv and data_preprocessed/val.csv and show
compact terminal blocks like:

<ASCII thumbnail>

GT:   the painting and colors ...
TF-IDF:  the man in the center looks ...
W2V:     the man in the painting looks ...
GloVe:   the man looks like he is in a battle

Run:
    python3 scripts/print_comparisons.py
"""

from pathlib import Path
import csv
import json
import sys
import numpy as np

# try Pillow for image->ASCII
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# Adjust how many comparisons to print
PRINT_FIRST_N = 20
ASCII_WIDTH = 48

# paths
EVAL_DIR = Path("eval_outputs")
TFIDF_CSV = EVAL_DIR / "m2_tfidf_samples.csv"
W2V_CSV = EVAL_DIR / "m2_fasttext_samples.csv"   # your word2vec/fasttext file
GLOVE_CSV = EVAL_DIR / "m2_glove_samples.csv"
RANDOM_CSV = EVAL_DIR / "m2_random_samples.csv"
VAL_CSV = Path("data_preprocessed/val.csv")
IMAGES_ROOT = Path("data_preprocessed/features")

# ASCII helper (small, re-uses approach from eval script)
ASCII_CHARS = list(" .:-=+*#%@")

def pil_from_array(arr):
    # arr may be (H,W,3) float [0,1], (3,H,W), or uint8
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow not installed - install with `pip install pillow`")
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] == 3:
        a = np.transpose(a, (1,2,0))
    if a.dtype != np.uint8:
        if a.max() <= 1.0:
            a = (a * 255.0).astype(np.uint8)
        else:
            a = a.astype(np.uint8)
    if a.ndim == 2:
        im = Image.fromarray(a, mode="L").convert("RGB")
    else:
        im = Image.fromarray(a).convert("RGB")
    return im

def pil_to_ascii(im, width=ASCII_WIDTH):
    im_g = im.convert("L")
    w, h = im_g.size
    aspect = h / w
    char_aspect = 2.0
    new_w = width
    new_h = max(1, int(aspect * new_w / char_aspect))
    im_small = im_g.resize((new_w, new_h))
    arr = np.array(im_small)
    bins = np.linspace(0, 255, len(ASCII_CHARS), endpoint=True)
    lines = []
    for row in arr:
        line = "".join(ASCII_CHARS[np.searchsorted(bins, int(v), side="right") - 1] for v in row)
        lines.append(line)
    return "\n".join(lines)

def load_samples(csv_path):
    """
    returns mapping painting_name -> generated_text (we prefer gen_text column if present)
    expects CSV header like: painting,ref_tokens,ref_text,gen_tokens,gen_text
    """
    d = {}
    if not csv_path.exists():
        return d
    with csv_path.open("r", encoding="utf8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # try to find which column is painting and which is gen_text
        # expected header: painting, ref_tokens, ref_text, gen_tokens, gen_text
        for row in reader:
            if len(row) < 5:
                continue
            painting = row[0]
            # gen_text likely in column 4
            gen_text = row[4]
            if gen_text is None or gen_text.strip()=="":
                # fallback: try to parse gen_tokens column (JSON)
                try:
                    gen_tokens = json.loads(row[3])
                    gen_text = " ".join(str(x) for x in gen_tokens)
                except Exception:
                    gen_text = ""
            d[painting] = gen_text
    return d

def load_gt(val_csv):
    """
    return mapping painting -> ground truth text
    val.csv should contain 'painting' and some text column (we try 'caption', 'tokens' or similar)
    We'll attempt to find 'caption' or 'tokens' columns; else we return empty string.
    """
    import pandas as pd
    df = pd.read_csv(val_csv)
    # find candidate columns
    text_col = None
    for c in ["caption", "caption_text", "ref_text", "tokens", "token_ids", "text"]:
        if c in df.columns:
            text_col = c
            break
    # If there is a 'token' column in tokenized form, just read 'ref_text' if available, else join tokens
    mapping = {}
    if text_col:
        for _, row in df.iterrows():
            mapping[str(row["painting"])] = str(row[text_col])
    else:
        # fallback: try 'ref_text' or build from token list columns
        if "ref_text" in df.columns:
            for _, row in df.iterrows():
                mapping[str(row["painting"])] = str(row["ref_text"])
        else:
            # try tokens column like 'token_ids' or 'tokens'
            tokcol = None
            for c in ["token_ids", "tokens", "token_ids_c1"]:
                if c in df.columns:
                    tokcol = c
                    break
            if tokcol:
                for _, row in df.iterrows():
                    val = row[tokcol]
                    # convert repr "[1,2,3]" to words? we can't convert ids -> words here, so show ids
                    mapping[str(row["painting"])] = str(val)
            else:
                # give empty fallback
                for _, row in df.iterrows():
                    mapping[str(row["painting"])] = ""
    return mapping

def main():
    # load sample predictions
    tfidf_map = load_samples(TFIDF_CSV)
    w2v_map = load_samples(W2V_CSV)
    glove_map = load_samples(GLOVE_CSV)
    random_map = load_samples(RANDOM_CSV)

    # load GT
    gt_map = load_gt(VAL_CSV)

    # build set of painting ids to show (intersection or union)
    paintings = list(gt_map.keys())
    if not paintings:
        # try union of CSV keys
        s = set(tfidf_map) | set(w2v_map) | set(glove_map) | set(random_map)
        paintings = list(s)

    if not paintings:
        print("No paintings found in val.csv or sample CSVs. Ensure eval_outputs CSVs exist and val.csv has 'painting' col.")
        return

    count = 0
    for p in paintings:
        if count >= PRINT_FIRST_N:
            break
        gt = gt_map.get(p, "")
        tf_pred = tfidf_map.get(p, "")
        w2v_pred = w2v_map.get(p, "")
        glove_pred = glove_map.get(p, "")
        random_pred = random_map.get(p, "")

        # render image if exists
        p_npy = IMAGES_ROOT / f"{p}.npy"
        printed_header = False
        if p_npy.exists() and PIL_AVAILABLE:
            try:
                arr = np.load(p_npy)
                pil = pil_from_array(arr)
                ascii_art = pil_to_ascii(pil, width=ASCII_WIDTH)
                print(ascii_art)
                printed_header = True
            except Exception as e:
                # ignore and continue to print text
                pass
        elif PIL_AVAILABLE:
            # maybe there is a file with different naming (like indices). skip for now.
            pass

        # print compact block like your screenshot
        if not printed_header:
            # print a small separator
            print("\n" + "-"*80)

        # GT line
        print("GT:     ", gt)
        # predictions (TF-IDF label first)
        print("TF-IDF: ", tf_pred)
        # label names you used earlier: W2V corresponds to fasttext/word2vec
        print("W2V:    ", w2v_pred)
        print("GloVe:  ", glove_pred)
        print("Random: ", random_pred)

        print("\n" + "="*80 + "\n")
        count += 1

if __name__ == "__main__":
    main()
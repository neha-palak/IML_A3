import pickle, numpy as np
vocab = pickle.load(open("data_preprocessed/vocab.pkl", "rb"))  # token->idx dict
vocab_size = len(vocab)
print("Vocab size:", vocab_size)

def build_and_save_embedding_matrix(emb_lookup, emb_dim, save_path, vocab=vocab):
    # emb_lookup: dict-like with emb_lookup[token] -> np.array(shape=(emb_dim,))
    mat = np.random.normal(scale=0.6, size=(len(vocab), emb_dim)).astype("float32")
    covered = 0
    for tok, idx in vocab.items():
        vec = emb_lookup.get(tok)
        if vec is not None:
            mat[idx] = vec
            covered += 1
    print("Embedding dim:", emb_dim, "Covered tokens:", covered, "Coverage:", covered/len(vocab))
    np.save(save_path, mat)

glove_dict = {...}
# else load glove text file (this takes time)
def load_glove(path):
    d={}
    with open(path,'r',encoding='utf8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            w = parts[0]
            vec = np.asarray(parts[1:], dtype='float32')
            d[w]=vec
    return d

glove_dict = load_glove("data/glove/glove.6B.300d.txt")   # adjust path
build_and_save_embedding_matrix(glove_dict, 300, "data_preprocessed/emb_glove_300.npy")

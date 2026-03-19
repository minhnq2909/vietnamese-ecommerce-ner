import numpy as np
import torch
from gensim.models import KeyedVectors

def load_phow2v_matrix(word2idx, phow2v_path, embedding_dim=300):
    """
    Load pre-trained PhoW2V and map to vocabulary of train set.
    """
    phow2v_model = KeyedVectors.load_word2vec_format(phow2v_path)
    
    vocab_size = len(word2idx)
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim))
    
    if "<PAD>" in word2idx:
        embedding_matrix[word2idx["<PAD>"]] = np.zeros(embedding_dim)

    hit_count = 0
    for word, idx in word2idx.items():
        if word in ["<PAD>", "<UNK>"]:
            continue
        if word in phow2v_model:
            embedding_matrix[idx] = phow2v_model[word]
            hit_count += 1
        elif word.lower() in phow2v_model:
            embedding_matrix[idx] = phow2v_model[word.lower()]
            hit_count += 1

    print(f"Mapping completed {hit_count}/{vocab_size} words ({(hit_count/vocab_size)*100:.2f}%).")
    return torch.tensor(embedding_matrix, dtype=torch.float32)
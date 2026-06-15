import torch
from torch.utils.data import Dataset
import gensim.models
import numpy as np
import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

class MRDataset(Dataset):
    """
     Dataset class for Movie Reviews (MR).
    Handles loading raw text, building vocabulary, and loading Word2Vec embeddings.
    """
    def __init__(self, pos_file, neg_file, w2v_path, max_len=60):
        print("--- Initializing MRDataset ---")
        
        # 1. Load and clean raw text data
        print("[Info] Loading text files...")
        with open(pos_file, 'r', encoding='latin-1') as f:
            pos_text = [clean_str(line) for line in f.readlines()]
        with open(neg_file, 'r', encoding='latin-1') as f:
            neg_text = [clean_str(line) for line in f.readlines()]
            
        self.sentences = pos_text + neg_text
        # Labels: 1 for Positive, 0 for Negative
        self.labels = [1]*len(pos_text) + [0]*len(neg_text)
        self.max_len = max_len
        
        # 2. Build Vocabulary from the dataset
        print("[Info] Building vocabulary...")
        # 0 is reserved for padding, 1 for unknown words
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        for sent in self.sentences:
            for word in sent.split():
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        
        # 3. Load Google News Pre-trained Vectors using Gensim
        print(f"[Info] Loading Word2Vec from {w2v_path} (This may take time)...")
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        
        # 4. Create Embedding Matrix
        # This matrix will initialize the model's embedding layer.
        vocab_size = len(self.vocab)
        embed_dim = 300
        self.embedding_matrix = np.zeros((vocab_size, embed_dim))
        
        found_words = 0
        for word, idx in self.vocab.items():
            if word in w2v_model:
                # Use pre-trained vector if present
                self.embedding_matrix[idx] = w2v_model[word]
                found_words += 1
            else:
                # Initialize randomly if the word is not in Google News W2V
                self.embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embed_dim, ))
                
        print(f"[Result] Found {found_words}/{vocab_size} words in Word2Vec.")
        # Convert to Tensor
        self.embedding_matrix = torch.tensor(self.embedding_matrix, dtype=torch.float32)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx].split()
        # Convert words to numerical indices
        sent_indices = [self.vocab.get(w, 1) for w in sent] # use 1 (<UNK>) if word not found
        
        # Padding: Ensure all sentences have the same length (max_len)
        if len(sent_indices) < self.max_len:
            sent_indices += [0] * (self.max_len - len(sent_indices))
        else:
            sent_indices = sent_indices[:self.max_len]
            
        return torch.tensor(sent_indices), torch.tensor(self.labels[idx], dtype=torch.float32)
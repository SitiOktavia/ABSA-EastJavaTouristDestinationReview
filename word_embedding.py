import numpy as np
import json 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# === Load tokenizer (word_index) ===
with open("word_index.json", "r") as f:
    word_index = json.load(f)
word_index = {str(k): int(v) for k, v in word_index.items()}

tokenizer = lambda text: [word_index.get(word, 0) for word in text.split()]

# === Tokenisasi untuk Word2Vec ===
#def tokenize_word2vec(texts, tokenizer=tokenizer, max_len=100):
 #   sequences = [tokenizer(text) for text in texts]
  #  return pad_sequences(sequences, maxlen=max_len, padding='post')

def tokenize_word2vec(text, tokenizer=tokenizer, max_len=100):
    sequence = tokenizer(text)  # hasilnya list angka, misal [12, 4, 56]
    padded = pad_sequences([sequence], maxlen=max_len, padding='post')  # harus dibungkus jadi list of list
    return padded


def bert_encode(texts, tokenizer, model_bert, max_len=100):
    if isinstance(texts, str):
        texts = [texts]  # ubah jadi list 1 elemen

    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors="tf"
    )
    outputs = model_bert(encodings['input_ids'], attention_mask=encodings['attention_mask'])
    return outputs.last_hidden_state.numpy()  # shape: (batch_size, max_len, 768)


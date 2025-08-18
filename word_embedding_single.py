import numpy as np
import json 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel

def encode_text(text, method="word2vec", max_len=100):
    if method == "word2vec":
        with open("word_index.json", "r") as f:
            word_index = json.load(f)
        word_index = {str(k): int(v) for k, v in word_index.items()}
        tokenizer = lambda text: [word_index.get(word, 0) for word in text.split()]
        sequence = tokenizer(text)
        padded = pad_sequences([sequence], maxlen=max_len, padding='post')
        return padded  # shape: (1, max_len)

    elif method == "indobert":        
        tokenizer = DistilBertTokenizer.from_pretrained("tokenizer_final")
        model_bert = TFDistilBertModel.from_pretrained("model_bert")
        if isinstance(text, str):
            text = [text]
            encodings = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_len,
            return_tensors="tf"
        )
        outputs = model_bert(encodings['input_ids'], attention_mask=encodings['attention_mask'])
        return outputs.last_hidden_state.numpy()  # shape: (1, max_len, 768)

    else:
        raise ValueError(f"Metode '{method}' tidak dikenali. Gunakan 'word2vec' atau 'bert'.")

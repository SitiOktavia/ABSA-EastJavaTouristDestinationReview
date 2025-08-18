from flask import Flask, render_template, request
import pandas as pd
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
from preprocessing import preprocess_text  # import fungsi preprocessing
import py3langid as langid  # untuk deteksi bahasa
from googletrans import Translator
import os
from collections import Counter

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Load tokenizer dan model BERT sekali di awal
tokenizer = DistilBertTokenizer.from_pretrained("tokenizer_final")
model_bert = TFDistilBertModel.from_pretrained("model_bert")

# Load model klasifikasi aspek dan sentimen
aspect_model = tf.keras.models.load_model('model_final.h5')
attraction_sentiment_model = tf.keras.models.load_model('attraction_final.h5')
amenities_sentiment_model = tf.keras.models.load_model('amenities_final.h5')
access_sentiment_model = tf.keras.models.load_model('access_final.h5')
price_sentiment_model = tf.keras.models.load_model('price_final.h5')

translator = Translator()

def encode_text(texts, method="indobert", max_len=100):
    if method == "word2vec":
        with open("word_index.json", "r") as f:
            word_index = json.load(f)
        word_index = {str(k): int(v) for k, v in word_index.items()}
        tokenizer_fn = lambda text: [word_index.get(word, 0) for word in text.split()]
        sequences = [tokenizer_fn(text) for text in texts]
        padded = pad_sequences(sequences, maxlen=max_len, padding='post')
        return padded

    elif method == "indobert":
        encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors="tf"
        )
        outputs = model_bert(encodings['input_ids'], attention_mask=encodings['attention_mask'])
        # output shape: (batch_size, seq_len, hidden_dim)
        return outputs.last_hidden_state.numpy()

    else:
        raise ValueError("Metode encoding tidak dikenali")

def detect_languages(texts):
    langs = []
    for text in texts:
        lang, _ = langid.classify(text)
        langs.append(lang)
    return langs

def detect_and_translate(texts):
    translated_texts = []
    for text in texts:
        lang, _ = langid.classify(text)
        if lang != 'id':
            try:
                translated = translator.translate(text, dest='id').text
                translated_texts.append(translated)
            except Exception as e:
                print(f"Translation failed for text: {text} - {e}")
                translated_texts.append(text)  # fallback ke asli kalau gagal translate
        else:
            translated_texts.append(text)
    return translated_texts

def classify_aspect(reviews):
    preprocessed = [preprocess_text(r, embedding="word2vec") for r in reviews]
    encoded = encode_text(preprocessed, method="word2vec")
    predictions = aspect_model.predict(encoded)
    print(predictions)
    batch_aspects = []
    for pred in predictions:
        aspects = []
        if pred[0] > 0.7: aspects.append("attraction")
        if pred[1] > 0.5: aspects.append("amenities")
        if pred[2] > 0.45: aspects.append("access")
        if pred[3] > 0.4: aspects.append("price")
        batch_aspects.append(aspects)
    return batch_aspects

def classify_sentiments(reviews, batch_aspects):
    preprocessed = [preprocess_text(r) for r in reviews]
    encoded = encode_text(preprocessed, method="indobert")

    sentiment_models = {
        "attraction": attraction_sentiment_model,
        "amenities": amenities_sentiment_model,
        "access": access_sentiment_model,
        "price": price_sentiment_model
    }

    batch_sentiments = []
    for i, aspects in enumerate(batch_aspects):
        sentiments = {}
        for aspect in sentiment_models.keys():
            if aspect in aspects:
                model = sentiment_models[aspect]
                pred = model.predict(np.expand_dims(encoded[i], axis=0))
                # pred shape biasanya (1,1), ambil nilai float
                score = pred[0][0] if hasattr(pred[0], '__getitem__') else pred[0]
                print(f"sentiment {aspect} : {score}")
                sentiments[aspect] = "positive" if score > 0.5 else "negative"
            else:
                sentiments[aspect] = "none"
        batch_sentiments.append(sentiments)
    return batch_sentiments

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        single_text = request.form.get('single_text', '').strip()
        batch_file = request.files.get('batch_file')

        if single_text:
            # Translate jika bukan bahasa Indonesia
            translated_texts = detect_and_translate([single_text])
            detected_aspects = classify_aspect(translated_texts)[0]
            sentiment_results = classify_sentiments([translated_texts[0]], [detected_aspects])[0]

            return render_template("index.html",
                                   original_text=single_text,
                                   detected_aspects=detected_aspects,
                                   sentiment_results=sentiment_results,
                                   translated_text=translated_texts[0])

        elif batch_file:
            try:
                import io
                filename = batch_file.filename.lower()
                if filename.endswith('.csv'):
                    df = pd.read_csv(batch_file)
                elif filename.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(batch_file)
                else:
                    return "Format file tidak didukung, gunakan CSV atau Excel", 400

                # Ambil input lokasi dari form (bisa kosong)
                location_filter = request.form.get('location', '').strip().lower()

                # Pastikan kolom 'review' ada
                if 'review' not in df.columns:
                    return "File tidak memiliki kolom 'review'.", 400
                
                 # Jika pengguna mengisi lokasi, pastikan kolom 'location' ada dan filter
                if location_filter:
                    if 'location' not in df.columns:
                        return "Kolom 'location' tidak ditemukan dalam file, tapi input lokasi diberikan.", 400

                    df['location'] = df['location'].astype(str).str.lower()
                    df = df[df['location'] == location_filter]

                if df.empty:
                        return f"Tidak ada ulasan untuk lokasi: {location_filter}", 400


                original_texts = df['review'].astype(str).tolist()
                langs = detect_languages(original_texts)
                translated_texts = detect_and_translate(original_texts)

                batch_aspects = classify_aspect(translated_texts)
                batch_sentiments = classify_sentiments(translated_texts, batch_aspects)

                batch_results = []
                for original_text, translated_text, aspects, sentiments, lang in zip(original_texts, translated_texts, batch_aspects, batch_sentiments, langs):
                    batch_results.append({
                        "text": original_text,
                        "translated": translated_text if lang != 'id' else original_text,
                        "aspects": aspects,
                        "sentiments": sentiments
                    })

                # Hitung distribusi sentimen untuk chart
                aspects_list = ["attraction", "amenities", "access", "price"]
                chart_data = {aspect: Counter() for aspect in aspects_list}
                for s in batch_sentiments:
                    for aspect in aspects_list:
                        label = s.get(aspect, "none").lower()
                        chart_data[aspect][label] += 1

                preprocessed_texts = df['word'].astype(str).tolist()

                from collections import defaultdict
                def get_top_keywords_by_aspect_sentiment(batch_results, preprocessed_texts, top_n=20):
                    keywords = defaultdict(lambda: defaultdict(list))
                    for i, entry in enumerate(batch_results):
                        tokens = preprocessed_texts[i].split()
                        for aspect, sentiment in entry["sentiments"].items():
                            if sentiment in ['positive', 'negative']:
                                keywords[aspect][sentiment].extend(tokens)
                    top_keywords = {}
                    for aspect, sentiments in keywords.items():
                        top_keywords[aspect] = {}
                        for sentiment, tokens in sentiments.items():
                            counts = Counter(tokens)
                            top_keywords[aspect][sentiment] = counts.most_common(top_n)
                    return top_keywords

                top_keywords = get_top_keywords_by_aspect_sentiment(batch_results, preprocessed_texts)


                return render_template("index.html", batch_results=batch_results, chart_data=chart_data, location=location_filter,
                    top_keywords=top_keywords)

            except Exception as e:
                return f"Error memproses file: {e}", 500
        else:
            return "Mohon masukkan teks atau upload file batch.", 400

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

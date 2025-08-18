# preprocessing.py
from indoNLP.preprocessing import remove_html, remove_url, replace_word_elongation, emoji_to_words, replace_slang
import re
from Singkatan.SingkatanConverter import SingkatanConverter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from word_embedding import tokenize_word2vec

# Inisialisasi stemmer dan converter di sini supaya tidak berulang tiap fungsi dipanggil
sc = SingkatanConverter()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

normalization_dict = {
    'ngga': 'tidak', 'nggak': 'tidak', 'gak': 'tidak', 'ga': 'tidak', 'g': 'tidak',
    'enggak': 'tidak', 'ngak': 'tidak', 'kagak': 'tidak', 'tdk': 'tidak', 'gaada': 'tidak ada',
    'gakada': 'tidak ada'
}

stopwords = [
    'saya', 'aku', 'kamu', 'dia', 'kami', 'kita', 'mereka', 'nya', 'dan', 'atau', 'serta',
    'dengan', 'tanpa', 'agar', 'supaya', 'karena', 'sehingga', 'bila', 'jika', 'walaupun',
    'meskipun', 'padahal', 'di', 'ke', 'dari', 'pada', 'untuk', 'dalam', 'oleh', 'terhadap',
    'sudah', 'belum', 'akan', 'sedang', 'masih', 'telah', 'telat', 'yang', 'itu', 'ini',
    'adalah', 'yaitu', 'merupakan', 'sebagai', 'tersebut', 'pun', 'saja', 'lah', 'deh', 'dong',
    'nih', 'loh', 'ya', 'kok', 'kan', 'toh', 'lalu', 'bagi', 'hal', 'ketika', 'saat', 'bahwa',
    'secara', 'lain', 'anda', 'begitu', 'mengapa', 'kenapa', 'yakni', 'itulah', 'lagi', 'maka',
    'demi', 'dimana', 'kemana', 'pula', 'sambil', 'supaya', 'kah', 'pun', 'sampai', 'sedangkan',
    'selagi', 'sementara', 'apakah', 'kecuali', 'selain', 'seolah', 'seraya', 'seterusnya',
    'boleh', 'dapat', 'dahulu', 'dulunya', 'anu', 'demikian', 'mari', 'nanti', 'melainkan',
    'oh', 'seharusnya', 'sebetulnya', 'setiap', 'setidaknya', 'duga', 'sana', 'sini', 'pernah',
    'datang', 'sisi', 'juga'
]

def clean_text(text: str) -> str:
    text = remove_html(text)
    text = remove_url(text)
    text = replace_word_elongation(text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def casefold_text(text: str) -> str:
    return text.lower()

def normalize_text(text: str) -> str:
    text = sc.convert(text)
    text = replace_slang(text)
    tokens = text.split()
    tokens = [normalization_dict.get(token, token) for token in tokens]
    return ' '.join(tokens)

def remove_stopwords(text: str) -> str:
    tokens = text.lower().split()
    filtered = [word for word in tokens if word not in stopwords]
    return ' '.join(filtered)

def stem_text(text: str) -> str:
    return stemmer.stem(text)

def preprocess_text(text: str, embedding=None) -> str:
    # Pipeline lengkap preprocessing
    text = clean_text(text)
    text = casefold_text(text)
    text = normalize_text(text)
    text = remove_stopwords(text)
    #if embedding=="word2vec":
     #   text = stem_text(text)
    return text




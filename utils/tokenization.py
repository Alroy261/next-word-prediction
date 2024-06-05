from tensorflow.keras.preprocessing.text import Tokenizer


def tokenize(data):
    tokenizer = Tokenizer(oov_token="<oov>")
    tokenizer.fit_on_texts(data["cleaned"])
    total_words = len(tokenizer.word_index) + 1
    return tokenizer, total_words

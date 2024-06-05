import os
import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

remove_stop_words = os.getenv("REMOVE_STOP_WORDS", False)
remove_punctuation = os.getenv("REMOVE_PUNCTUATION", False)
convert_to_lowercase = os.getenv("CONVERT_TO_LOWERCASE", False)
lemmatization = os.getenv("LEMMATIZATION", False)


def preprocess(data_path="phrases_data.txt"):
    data = pd.read_csv(data_path, sep="\t", names=["phrases"])

    # Remove punctuation at the end
    data["cleaned"] = data["phrases"].apply(lambda x: re.sub(r"[.!?]+$", "", x))

    # Remove whitespace
    data["cleaned"] = data["cleaned"].str.strip()

    if convert_to_lowercase:
        data["cleaned"] = data["cleaned"].str.lower()

    if remove_stop_words:
        stop_words = set(stopwords.words("english"))
        data["cleaned"] = data["cleaned"].apply(
            lambda x: " ".join([word for word in x.split() if word not in stop_words])
        )
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        data["cleaned"] = data["cleaned"].apply(
            lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()])
        )

    return data

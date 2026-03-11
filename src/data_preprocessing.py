# text preprocessing utilities

import re
import nltk
from nltk.corpus import stopwords

# download stopwords
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


def clean_text(text):
    # convert to lowercase
    text = text.lower()

    # remove non alphabetic characters
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # tokenize words
    words = text.split()

    # remove stopwords
    words = [w for w in words if w not in stop_words]

    # join cleaned words
    return " ".join(words)
import re
import time
from collections import Counter
from typing import List

import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from rapidfuzz.distance import Levenshtein
from transformers import pipeline

# Read the data
df_train = pd.read_csv('data/yt_comments_train.csv', sep=',', header=0)
df_test = pd.read_csv('data/yt_comments_test.csv', sep=',', header=0)


def get_clean_text(text: str):
    """remove the https address, email, and number form the text"""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"\d{2,}", "", text)

    return text


#  Clean the data
df_train["clean_text"] = df_train["text"].apply(get_clean_text)

# Remove the column with empty string
df_train = df_train[df_train["clean_text"] != ""]

# Keyword extraction
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=sentence_model)

# Extract keywords by sentence-transformer
df_train['keywords'] = df_train['clean_text'].apply(lambda x: kw_model.extract_keywords(x, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=5))

# Get the most common keywords
keywords_list = df_train['keywords'].apply(lambda x: [i[0] for i in x]).tolist()
keywords_list = [item for sublist in keywords_list for item in sublist]
keywords_most_frequent = Counter(keywords_list).most_common(15)

ham_keywords = ["song", "love", "music", "like"]
spam_keywords = ["subscribe", "channel", "check"]


# Get the most similar words of the text to the keywords
def get_similarity(metric, keywords: List[str], text: str, threshold: float = 0.75):
    sim_metric = metric.normalized_similarity
    words = text.split()
    sim_values = [word for word in words for key_word in keywords if sim_metric(key_word, word) >= threshold]

    return sim_values


#  Zero shot classification
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

zs_labels = classifier(df_train["clean_text"].iloc[0:10].tolist(), ham_keywords + spam_keywords, multi_label=False)

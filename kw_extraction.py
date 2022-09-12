import re
from collections import Counter
from typing import List

import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from rapidfuzz.distance import Levenshtein
from top2vec import Top2Vec

# Read the data
df_train = pd.read_csv('data/yt_comments_train.csv', sep=',', header=0)


def get_clean_text(text: str):
    """remove the https address, email, and number form the text"""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"\d{2,}", "", text)

    return text


# Keyword extraction
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=sentence_model)

# Extract keywords
df_train["clean_text"] = df_train["text"].apply(get_clean_text)
df_train['keywords'] = df_train['clean_text'].apply(lambda x: kw_model.extract_keywords(x, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=5))

# Get the most common keywords
keywords_list = df_train['keywords'].apply(lambda x: [i[0] for i in x]).tolist()
keywords_list = [item for sublist in keywords_list for item in sublist]
keywords_most_frequent = Counter(keywords_list).most_common(10)

ham_keywords = ["video", "song", "love", "music"]
spam_keywords = ["subscribe", "channel", "check", "views", "like"]


# Get the most similar words of the text to the keywords
def get_similarity(metric, keywords: List[str], text: str, threshold: float = 0.75):
    sim_metric = metric.normalized_similarity
    words = text.split()
    sim_values = [word for word in words for key_word in keywords if sim_metric(key_word, word) >= threshold]

    return sim_values


get_similarity(Levenshtein, ham_keywords, "I am vide son musc", 0.75)

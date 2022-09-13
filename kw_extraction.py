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
df_train = pd.read_csv('data/yt_comments_train.csv', sep=',', header=0, index_col=0)
df_test = pd.read_csv('data/yt_comments_test.csv', sep=',', header=0, index_col=0)


def get_clean_text(text: str):
    """remove the https address, email, emoji, html tag, and number form the text"""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"\d{2,}", "", text)
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\ufeff", "", text)
    text = re.sub(r"\s+", " ", text)

    return text


#  Clean the data
def get_clean_data(train_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame):
    train_dataframe['clean_text'] = train_dataframe['text'].apply(get_clean_text)
    test_dataframe['clean_text'] = test_dataframe['text'].apply(get_clean_text)

    # Remove the column with empty string or only spaces
    train_dataframe = train_dataframe[train_dataframe['clean_text'] != '']
    test_dataframe = test_dataframe[test_dataframe['clean_text'] != '']

    return train_dataframe, test_dataframe


df_test, df_train = get_clean_data(df_train, df_test)

# Keyword extraction
# sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
sentence_model = SentenceTransformer("facebook/bart-large-mnli")
keybert_model = KeyBERT(model=sentence_model)


def get_kw(kw_model: KeyBERT, train_dataframe: pd.DataFrame, top_n: int = 5, n_common: int = 5):
    # Extract keywords by sentence-transformer
    train_dataframe['keywords'] = train_dataframe['clean_text'].apply(lambda x: kw_model.extract_keywords(x, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=top_n))

    # Get the most common keywords
    keywords_list = train_dataframe['keywords'].apply(lambda x: [i[0] for i in x]).tolist()
    keywords_list = [item for sublist in keywords_list for item in sublist]
    keywords_most_frequent = Counter(keywords_list).most_common(n_common)

    return keywords_most_frequent


frequent_keywords = get_kw(keybert_model, df_train, top_n=5, n_common=5)
ham_keywords = ["song", "love", "music", "like"]
spam_keywords = ["subscribe", "channel", "check"]


# Get the most similar words of the text to the keywords
def get_similarity(metric, keywords: List[str], text: str, threshold: float = 0.75):
    #  check whether metric has the method normalized_similarity
    if hasattr(metric, 'normalized_similarity'):
        sim_metric = metric.normalized_similarity
        words = text.split()
        sim_values = [word for word in words for key_word in keywords if sim_metric(key_word, word) >= threshold]
    else:
        raise ValueError("The metric does not have the method normalized_similarity")

    return sim_values


#  Zero shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
zs_labels = classifier(df_train["clean_text"].iloc[0:10].tolist(), ham_keywords + spam_keywords, multi_label=False)


#  Clustering

def get_clustered_date(model, training_dataframe: pd.DataFrame):
    corpus_sentences = training_dataframe["clean_text"].tolist()

    print("Start clustering")
    start_time = time.time()

    corpus_embeddings = model.encode(corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)

    # Two parameters to tune:
    # min_cluster_size: Only consider cluster that have at least 25 elements
    # threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
    clusters = util.community_detection(corpus_embeddings, min_community_size=10, threshold=0.75)

    print("Clustering done after {:.2f} sec".format(time.time() - start_time))

    clusters_sentences_list = [list(set([corpus_sentences[idx] for idx in cluster])) for cluster in clusters]

    # Print for all clusters the top 3 and bottom 3 elements
    for i, cluster in enumerate(clusters):
        print("\nCluster {}, #{} Elements ".format(i + 1, len(cluster)))
        for sentence_id in cluster[0:3]:
            print("\t", corpus_sentences[sentence_id])
        print("\t", "...")
        for sentence_id in cluster[-3:]:
            print("\t", corpus_sentences[sentence_id])

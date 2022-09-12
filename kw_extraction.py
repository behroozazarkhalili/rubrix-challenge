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
df_train["clean_text"] = df_train["text"].apply(get_clean_text)

# Remove the column with empty string or only spaces
df_train = df_train[df_train["clean_text"].str.strip() != ""]

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

#  Clustering
corpus_sentences = df_train["clean_text"].tolist()
print("Start clustering")
start_time = time.time()

corpus_embeddings = sentence_model.encode(corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)

# Two parameters to tune:
# min_cluster_size: Only consider cluster that have at least 25 elements
# threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
clusters = util.community_detection(corpus_embeddings, min_community_size=10, threshold=0.75)

print("Clustering done after {:.2f} sec".format(time.time() - start_time))

# Print for all clusters the top 3 and bottom 3 elements

clusters_sentences_list = [list(set([corpus_sentences[idx] for idx in cluster])) for cluster in clusters]

for i, cluster in enumerate(clusters):
    print("\nCluster {}, #{} Elements ".format(i + 1, len(cluster)))
    for sentence_id in cluster[0:3]:
        print("\t", corpus_sentences[sentence_id])
    print("\t", "...")
    for sentence_id in cluster[-3:]:
        print("\t", corpus_sentences[sentence_id])

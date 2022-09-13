import re
import time
from collections import Counter
from typing import List

import numpy as np
import rubrix as rb
import pandas as pd
from keybert import KeyBERT
from rubrix.labeling.text_classification import WeakLabels, load_rules, MajorityVoter
from sentence_transformers import SentenceTransformer, util
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import classification_report
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
    text = text.lower()

    return text


#  Clean the data
def get_clean_data(train_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame):
    train_dataframe['clean_text'] = train_dataframe['text'].apply(get_clean_text)
    test_dataframe['clean_text'] = test_dataframe['text'].apply(get_clean_text)

    # Remove the column with empty string or only spaces
    train_dataframe = train_dataframe[train_dataframe['clean_text'].str.strip() != '']
    test_dataframe = test_dataframe[test_dataframe['clean_text'].str.strip() != '']

    return train_dataframe, test_dataframe


df_train, df_test = get_clean_data(df_train, df_test)

# Keyword extraction
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
keybert_model = KeyBERT(model=sentence_model)


def get_kw(kw_model: KeyBERT, train_dataframe: pd.DataFrame, top_n: int = 5):
    # Extract keywords by sentence-transformer
    train_dataframe['keywords'] = train_dataframe['clean_text'].apply(lambda x: kw_model.extract_keywords(x, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=top_n))

    # Get the most common keywords
    keywords_list = train_dataframe['keywords'].apply(lambda x: [i[0] for i in x]).tolist()
    keywords_list = [item for sublist in keywords_list for item in sublist]

    return keywords_list


frequent_keywords = get_kw(keybert_model, df_train, top_n=5)
keywords_most_frequent = Counter(frequent_keywords).most_common(10)
ham_keywords = ["song", "love", "music", "like"]
spam_keywords = ["subscribe", "channel", "check", "youtube"]


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
corpus_sentences = df_train["clean_text"].tolist()
print("Start clustering")
start_time = time.time()

corpus_embeddings = sentence_model.encode(corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)

# Two parameters to tune:
# min_cluster_size: Only consider cluster that have at least 25 elements
# threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
clusters = util.community_detection(corpus_embeddings, min_community_size=10, threshold=0.75)

print("Clustering done after {:.2f} sec".format(time.time() - start_time))

clusters_sentences_list = [[corpus_sentences[idx] for idx in cluster] for cluster in clusters]

# Print for all clusters the top 3 and bottom 3 elements
for i, cluster in enumerate(clusters):
    print("\nCluster {}, #{} Elements ".format(i + 1, len(cluster)))
    for sentence_id in cluster[0:3]:
        print("\t", corpus_sentences[sentence_id])
    print("\t", "...")
    for sentence_id in cluster[-3:]:
        print("\t", corpus_sentences[sentence_id])


def get_kw_from_cluster(kw_model: KeyBERT, text_list: List[List[str]], top_n: int = 3):
    initial_kws = [kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=top_n) for text in text_list]
    kws_list = [[k[0] for kw in cluster for k in kw] for cluster in initial_kws]
    return kws_list


cluster_frequent_kws = get_kw_from_cluster(keybert_model, clusters_sentences_list, 5)
cluster_frequent_kws = [cluster_frequent_kw for cluster_frequent_kw in cluster_frequent_kws if len(cluster_frequent_kw) > 0]
cluster_most_frequent_kws_init = [Counter(cluster_keywords).most_common(3) for cluster_keywords in cluster_frequent_kws]
cluster_most_frequent_kws = [[item[0] for item in most_frequent_kws] for most_frequent_kws in cluster_most_frequent_kws_init]

cluster_ham_keywords = ["song", "love", "best"]
cluster_spam_keywords = ["subscribe", "channel", "check", "playlist"]

# build records from the train dataset
records = [
    rb.TextClassificationRecord(
        text=row.clean_text,
        metadata={"video": row.video, "author": row.author}
    )
    for i, row in df_train.iterrows()
]

# build records from the test dataset with annotation
labels = ["HAM", "SPAM"]

df_train["label"] = df_train["label"].astype(int)
records += [
    rb.TextClassificationRecord(
        text=row.clean_text,
        annotation=labels[row.label],
        metadata={"video": row.video, "author": row.author}
    )
    for i, row in df_train.iterrows()
]

# log records to Rubrix
rb.log(records, name="weak_supervision_yt")


# rules defined as Python labeling functions
def contains_subscribe(record: rb.TextClassificationRecord):
    if "subscribe" in record.inputs["text"]:
        return "SPAM"


def contains_check(record: rb.TextClassificationRecord):
    if "check" in record.inputs["text"]:
        return "SPAM"


def contains_channel(record: rb.TextClassificationRecord):
    if "channel" in record.inputs["text"]:
        return "SPAM"


def contains_playlist(record: rb.TextClassificationRecord):
    if "playlist" in record.inputs["text"]:
        return "SPAM"


def contains_youtube(record: rb.TextClassificationRecord):
    if "youtube" in record.inputs["text"]:
        return "SPAM"


def contains_song(record: rb.TextClassificationRecord):
    if "song" in record.inputs["text"]:
        return "HAM"


def contains_love(record: rb.TextClassificationRecord):
    if "love" in record.inputs["text"]:
        return "HAM"


def contains_best(record: rb.TextClassificationRecord):
    if "best" in record.inputs["text"]:
        return "HAM"


rules = [contains_subscribe, contains_check, contains_channel, contains_song, contains_love, contains_best]
rules += load_rules(dataset=df_train)

weak_labels = WeakLabels(rules=rules, dataset=df_train)

# show some stats about the rules, see the `summary()` docstring for details
weak_labels.summary()

# instantiate the majority vote label model by simply providing the weak labels object
majority_model = MajorityVoter(weak_labels)

# check its performance
print(majority_model.score(output_str=True))


def get_cluster_similarity(cluster_info_list: List[List[str]], text: str):
    cluster_embeddings_list = [sentence_model.encode(cluster_info, convert_to_tensor=True) for cluster_info in cluster_info_list]
    text_embeddings = sentence_model.encode(text, batch_size=64, convert_to_tensor=True)

    similarity_matrices = [util.cos_sim(text_embeddings, cluster_embedding).cpu().numpy() for cluster_embedding in cluster_embeddings_list]
    similarity_scores = [np.mean(similarity_matrix) for similarity_matrix in similarity_matrices]
    normalized_similarity_scores = [score / np.sum(similarity_scores) for score in similarity_scores]
    similar_clusters = np.argsort(similarity_scores)[::-1]
    return normalized_similarity_scores, similar_clusters.tolist()


similarity_ss, _ = get_cluster_similarity(clusters_sentences_list, df_test["clean_text"].iloc[0])
similarity_skw, _ = get_cluster_similarity(cluster_most_frequent_kws, df_test["clean_text"].iloc[0])

labels = ["HAM", "SPAM"]
cluster_labels = ["SPAM", "SPAM", "SPAM", "HAM", "HAM", "SPAM", "HAM", "HAM"]


def get_indices(cluster_class_labels: List[str], class_labels: List[str]):
    indices = {label: [index for index in range(len(cluster_class_labels)) if cluster_class_labels[index] == label] for label in class_labels}
    return indices


def get_class_labels(clusters_info: List[List[str]], text: str, cluster_class_labels: List[str], class_labels: List[str]):
    similarity_scores, similar_clusters = get_cluster_similarity(clusters_info, text)
    labels_indices = get_indices(cluster_class_labels, class_labels)
    class_labels_scores = {label: sum([similarity_scores[index] for index in indices]) for label, indices in labels_indices.items()}
    return class_labels_scores


get_class_labels(cluster_most_frequent_kws, df_test["clean_text"].iloc[0], cluster_labels, labels)


def get_df_class_labels(df: pd.DataFrame, clusters_info: List[List[str]], cluster_class_labels: List[str], class_labels: List[str]):
    df_class_labels = df.copy()
    df_class_labels["class_label_scores"] = df_class_labels["clean_text"].apply(lambda text: get_class_labels(clusters_info, text, cluster_class_labels, class_labels)[0])
    df_class_labels["class_label"] = df_class_labels["class_label_scores"].apply(lambda class_labels_scores: max(class_labels_scores, key=class_labels_scores.get))
    df_final = pd.concat([df_class_labels, df_class_labels['class_label_scores'].apply(pd.Series)], axis=1).drop('class_label_scores', axis=1)
    return df_final


df_pred = get_df_class_labels(df_test, cluster_most_frequent_kws, cluster_labels, labels)
df_pred["true_label"] = df_pred["class_label"].apply(lambda label: labels.index(label))

# Get the classification report from scikit learn
print(classification_report(df_pred["true_label"], df_pred["label"]))

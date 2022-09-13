import re
import time
from collections import Counter
from typing import List

import numpy as np
import rubrix as rb
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import classification_report

# Read the data
df_train = pd.read_csv('data/yt_comments_train.csv', sep=',', header=0, index_col=0)
df_test = pd.read_csv('data/yt_comments_test.csv', sep=',', header=0, index_col=0)


def get_clean_text(text):
    """
    remove the https address, email, emoji, html tag, and number form the text
    :param str text: the text to be cleaned
    :return: return the cleaned text
    """
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"\d{2,}", "", text)
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\ufeff", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.lower()

    return text


#  Clean the dataframes
def get_clean_data(train_dataframe, test_dataframe):
    """
    Clean the data
    :param pd.DataFrame train_dataframe: the train dataframe
    :param pd.DataFrame test_dataframe: the test dataframe
    :return: return the cleaned dataframes
    """
    train_dataframe['clean_text'] = train_dataframe['text'].apply(get_clean_text)
    test_dataframe['clean_text'] = test_dataframe['text'].apply(get_clean_text)

    # Remove the column with empty string or only spaces
    train_dataframe = train_dataframe[train_dataframe['clean_text'].str.strip() != '']
    test_dataframe = test_dataframe[test_dataframe['clean_text'].str.strip() != '']

    return train_dataframe, test_dataframe


df_train, df_test = get_clean_data(df_train, df_test)


def get_kw_model(model_name_path: str = "all-MiniLM-L6-v2"):
    """
    Get the keyword extraction model
    :param str model_name_path: the model name or path
    :return: return the keyword extraction model
    """
    sentence_model = SentenceTransformer(model_name_or_path=model_name_path)
    kw_model = KeyBERT(model=sentence_model)
    return kw_model, sentence_model


# Keyword extraction
keybert_model, _ = get_kw_model()


def get_kw(kw_model, df, top_n=5):
    """
    Extract keywords by sentence-transformer
    :param KeyBERT kw_model: the keyword extraction model
    :param pd.DataFrame df: the dataframe
    :param int top_n: the number of keywords to extract
    """
    df['keywords'] = df['clean_text'].apply(lambda x: kw_model.extract_keywords(x, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=top_n))

    # Get the most common keywords
    keywords_list = df['keywords'].apply(lambda x: [i[0] for i in x]).tolist()
    keywords_list = [item for sublist in keywords_list for item in sublist]

    return keywords_list


frequent_keywords = get_kw(keybert_model, df_train, top_n=5)
keywords_most_frequent = Counter(frequent_keywords).most_common(10)

ham_keywords = ["song", "love", "music", "like"]
spam_keywords = ["subscribe", "channel", "check", "youtube"]


# Get the most similar words of the text to the keywords
def get_similarity(metric, keywords, text, threshold=0.75):
    """
    check whether metric has the method normalized_similarity
    :param metric: the metric to check the similarity
    :param List[str] keywords: the keywords to check the similarity
    :param str text: the text to check the similarity
    :param float threshold: the threshold to check the similarity
    """
    if hasattr(metric, 'normalized_similarity'):
        sim_metric = metric.normalized_similarity
        words = text.split()
        sim_values = [word for word in words for key_word in keywords if sim_metric(key_word, word) >= threshold]
    else:
        raise ValueError("The metric does not have the method normalized_similarity")

    return sim_values


#  Clustering
corpus_sentences = df_train["clean_text"].tolist()
print("Start clustering")
start_time = time.time()

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeddings = sentence_model.encode(corpus_sentences, batch_size=64, convert_to_tensor=True)

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


def get_kw_from_cluster(kw_model, text_list, top_n=3):
    """
    Extract keywords from the cluster
    :param KeyBERT kw_model: the keyword extraction model
    :param List[List[str]] text_list: the list of text
    :param int top_n: the number of keywords to extract
    """
    initial_kws = [kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=top_n) for text in text_list]
    kws_list = [[k[0] for kw in cluster for k in kw] for cluster in initial_kws]
    return kws_list


cluster_frequent_kws = get_kw_from_cluster(keybert_model, clusters_sentences_list, 5)
cluster_frequent_kws = [cluster_frequent_kw for cluster_frequent_kw in cluster_frequent_kws if len(cluster_frequent_kw) > 0]
cluster_most_frequent_kws_init = [Counter(cluster_keywords).most_common(3) for cluster_keywords in cluster_frequent_kws]
cluster_most_frequent_kws = [[item[0] for item in most_frequent_kws] for most_frequent_kws in cluster_most_frequent_kws_init]

cluster_ham_keywords = ["song", "love", "best"]
cluster_spam_keywords = ["subscribe", "channel", "check", "playlist"]

# Build records from the train dataset
records = [
    rb.TextClassificationRecord(
        text=row.clean_text,
        metadata={"video": row.video, "author": row.author}
    )
    for i, row in df_train.iterrows()
]


# Develop new model based on clustering algorithm
def get_cluster_similarity(cluster_info_list, text):
    """
    Get the similarity between the text and each cluster
    :param List[List[str]] cluster_info_list: the list of cluster information which can be either list of text or keywords of each cluster
    :param str text:
    :return: return the similarity between the text and each cluster
    """
    # Encode the list of info in each cluster
    cluster_embeddings_list = [sentence_model.encode(cluster_info, convert_to_tensor=True) for cluster_info in cluster_info_list]

    # Encode the text
    text_embeddings = sentence_model.encode(text, batch_size=64, convert_to_tensor=True)

    # Compute the similarity between the text and the list of info in each cluster
    similarity_matrices = [util.cos_sim(text_embeddings, cluster_embedding).cpu().numpy() for cluster_embedding in cluster_embeddings_list]

    # Get the average similarity score between the text and each cluster
    similarity_scores = [np.mean(similarity_matrix) for similarity_matrix in similarity_matrices]

    # Get the normalized similarity score between the text and each cluster
    normalized_similarity_scores = [score / np.sum(similarity_scores) for score in similarity_scores]

    #  Sort the clusters based on the similarity score
    similar_clusters = np.argsort(similarity_scores)[::-1]
    return normalized_similarity_scores, similar_clusters.tolist()


similarity_ss, _ = get_cluster_similarity(clusters_sentences_list, df_test["clean_text"].iloc[0])
similarity_skw, _ = get_cluster_similarity(cluster_most_frequent_kws, df_test["clean_text"].iloc[0])

labels = ["HAM", "SPAM"]
cluster_labels = ["SPAM", "SPAM", "SPAM", "HAM", "HAM", "SPAM", "HAM", "HAM"]


def get_indices(cluster_class_labels, class_labels):
    """
    Get the dict of indices of the clusters that belong to the specific class label
    :param List[str] cluster_class_labels: the list of class labels of clusters
    :param List[str] class_labels: the list of unique class labels
    :return: return the dict of indices of the clusters that belong to the specific class label
    """
    indices = {label: [index for index in range(len(cluster_class_labels)) if cluster_class_labels[index] == label] for label in class_labels}
    return indices


def get_class_labels(clusters_info: List[List[str]], text: str, cluster_class_labels: List[str], class_labels: List[str]):
    """
    Get the class label of the text based on the clusters information
    :param clusters_info: the list of cluster information which can be either list of text or keywords of each cluster
    :param text: the text to be classified
    :param cluster_class_labels: the list of class labels of clusters
    :param class_labels: the list of unique class labels
    :return: return the class label of the text based on the clusters information
    """
    # Get the similarity score between the text and each cluster
    similarity_scores, similar_clusters = get_cluster_similarity(clusters_info, text)

    # Get the dict of indices of the clusters that belong to the specific class label
    labels_indices = get_indices(cluster_class_labels, class_labels)

    # Get the similarity score of each specific class label
    class_labels_scores = {label: round(sum([similarity_scores[index] for index in indices]), 3) for label, indices in labels_indices.items()}
    return class_labels_scores


print(get_class_labels(cluster_most_frequent_kws, df_test["clean_text"].iloc[0], cluster_labels, labels))


def get_df_class_labels(df: pd.DataFrame, clusters_info: List[List[str]], cluster_class_labels: List[str], class_labels: List[str]):
    """
    Get the class labels of the text column of a dataframe based on the clusters information
    :param pd.DataFrame df: the dataframe
    :param List[List[str]] clusters_info: the list of cluster information which can be either list of text or keywords of each cluster
    :param List[str] cluster_class_labels: the list of class labels of clusters
    :param List[str] class_labels: the list of unique class labels
    :return: List[str] return the class labels of the text column of a dataframe based on the clusters information
    """
    # Copy the dataframe
    df_class_labels = df.copy()

    # Add the column which is the dict of the scores of class labels
    df_class_labels["class_label_scores"] = df_class_labels["clean_text"].apply(lambda text: get_class_labels(clusters_info, text, cluster_class_labels, class_labels))

    # Add the column which is the class label
    df_class_labels["class_label"] = df_class_labels["class_label_scores"].apply(lambda class_labels_scores: max(class_labels_scores, key=class_labels_scores.get))

    #  Explode the dict column of the scores of class labels
    df_final = pd.concat([df_class_labels, df_class_labels['class_label_scores'].apply(pd.Series)], axis=1).drop('class_label_scores', axis=1)
    return df_final


df_pred = get_df_class_labels(df_test, cluster_most_frequent_kws, cluster_labels, labels)
df_pred["true_label"] = df_pred["class_label"].apply(lambda label: labels.index(label))

# Get the classification report from scikit learn
print(classification_report(df_pred["true_label"], df_pred["label"]))

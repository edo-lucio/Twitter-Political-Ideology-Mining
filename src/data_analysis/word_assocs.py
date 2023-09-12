from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re
import random

from itertools import combinations

emfd = pd.read_csv(
    "emfdscore\emfdscore\dictionaries\emfd_scoring.csv")

emfd_words = set(emfd["word"])


def find_frequent_words(word_list, n):
    frequent_words = []
    for word in word_list:
        if word_list.count(word) > n:
            frequent_words.append(word)

    return frequent_words


def preprocess(df):
    # Combine all the text data into a single string
    text_raw = " ".join(df["tweet_text"])
    text = re.sub('[^0-9a-zA-Z]+', ' ', text_raw)

    # Tokenize the text into words
    words_raw = word_tokenize(text)
    words = [w for w in words_raw if w.isalpha()]
    words = find_frequent_words(words, 10)

    # Get the set of English stopwords
    stop_words = set(stopwords.words("english"))

    # Add your custom words here
    custom_stopwords = ["brexit", "https",
                        "amp", "see", "back", "look", "co", "t", "youtube"]
    stop_words.update(custom_stopwords)

    filtered_words = [word for word in words if word.lower() not in stop_words]
    return filtered_words


def word_assocs(df):
    filtered_words = preprocess(df)

    # 1. find moral words
    moral_words = list(set(filtered_words) & emfd_words)

    # 2. link moral words to frequent words
    frequent_words = find_frequent_words(filtered_words)

    frequent_moral_words = find_frequent_words(moral_words)

    cooccurrence_matrix = {
        word: {word2: 0 for word2 in frequent_words} for word in moral_words}

    print(cooccurrence_matrix)

    fig = plt.figure(figsize=(30, 20))
    G = nx.Graph()

    for i in range(len(frequent_moral_words)):
        G.add_node(frequent_moral_words[i])
        for word in frequent_words[i]:
            G.add_edges_from([(frequent_moral_words[i], word)])

    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, with_labels=True, font_size=20)
    plt.show()


def word_assocs_new(df):

    vectorizer = CountVectorizer()
    tweets = preprocess(df)

    tdm = vectorizer.fit_transform(tweets)

    tdm_df = pd.DataFrame(
        tdm.toarray(), columns=vectorizer.get_feature_names_out())

    print(tdm_df)

    # Get word associations based on co-occurrence
    word_associations = {}
    for word1, word2 in combinations(tdm_df.columns, 2):
        cooccurrences = tdm_df[(tdm_df[word1] > 0) &
                               (tdm_df[word2] > 0)].shape[0]
        if cooccurrences > 0:
            word_associations[(word1, word2)] = cooccurrences

    # Create the graph
    G = nx.Graph()
    for (word1, word2), weight in word_associations.items():
        G.add_edge(word1, word2, weight=weight)

    # Plot the graph
    plt.figure(figsize=(10, 8))
    # You can use different layout algorithms here
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=2000,
            font_size=12, font_weight='bold')

    # Add edge labels
    edge_labels = {(word1, word2): weight for (word1, word2),
                   weight in word_associations.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("data\\tweets\\right-regular.csv")

    word_assocs_new(data)

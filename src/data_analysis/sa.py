from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import pandas as pd

from scipy.stats import ttest_ind

from numpy import std, mean, sqrt, var


task = 'sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)


# Preprocess text (username and link placeholders
def cohens_d(v1, v2):
    var1 = var(v1)
    var2 = var(v2)

    n1 = len(v1) - 1
    n2 = len(v2) - 1

    pooled_std = sqrt(((n1 * var1) + (n2 * var2)) / (n1 + n2))

    mean_v1 = mean(v1)
    mean_v2 = mean(v2)

    d = round(abs((mean_v1 - mean_v2) / pooled_std), 2)
    return d


def filter_tweets(tweets_df, keywords):
    filtered_df = tweets_df[tweets_df["tweet_text"].str.contains(
        "|".join(keywords), case=False)]

    return filtered_df


def preprocess(text):
    new_text = []

    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


# download label mapping
def setup_labels():
    labels = []
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

    return labels


def setup_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)
    tokenizer.save_pretrained(MODEL)
    return model


def score(df):
    model = setup_model()
    labels = setup_labels()

    sent_data = {"Text": [], "positive": [], "neutral": [], "negative": []}

    keywords = ["juststop_oil", "juststopoil", "xrebellion", "xrebellionuk", "#xrebellionuk",
                "just stop oil", "@juststop_oil", "x rebellion", "greenpeace", "ExtinctionR"]
    filtered_df = filter_tweets(df, keywords)

    for tweet in filtered_df["tweet_text"]:
        text = preprocess(tweet)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        sent_data["Text"] = tweet

        for i in range(scores.shape[0]):
            l = labels[ranking[i]]
            s = scores[ranking[i]]

            sent_data[l].append(s)

    out_df = pd.DataFrame(sent_data)
    return out_df


if __name__ == "__main__":
    paths = ["data\\tweets\\internal-regular.csv",
             "data\\tweets\\external-regular.csv"]

    data = [pd.read_csv(
        filepath_or_buffer=path, encoding="utf-8", on_bad_lines="skip") for path in paths]

    scores = [score(d) for d in data]

    print(scores[0]["negative"].mean())
    print(scores[1]["negative"].mean())

    t_stats, p_value = ttest_ind(scores[0]["negative"], scores[1]["negative"])
    effect_size = cohens_d(scores[0]["negative"], scores[1]["negative"])

    print("p-value", p_value)
    print("cohen's d", effect_size)

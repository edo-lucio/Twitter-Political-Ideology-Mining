import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import ttest_ind

from transformers import pipeline

from numpy import mean, sqrt, var
import torch

# Load the sentiment analysis pipeline with DistilBERT
sentiment_analyzer = pipeline(
    'sentiment-analysis', model='distilbert-base-uncased', )


def vader(data_frame):
    sid = SentimentIntensityAnalyzer()
    data_frame['Sentiment'] = data_frame['tweet_text'].apply(
        lambda tweet: sid.polarity_scores(tweet)['compound'])
    return data_frame


# Step 2: Perform sentiment analysis on each text in the DataFrame
def perform_sentiment_analysis(data_frame):
    sentiments = []
    confidence_scores = []

    for text in data_frame['tweet_text']:
        result = sentiment_analyzer(text)[0]
        sentiments.append(result['label'])
        confidence_scores.append(result['score'])

    data_frame['Sentiment'] = sentiments
    data_frame['Confidence'] = confidence_scores
    return data_frame


def filter_tweets(tweets_df, keywords):
    filtered_df = tweets_df[tweets_df["tweet_text"].str.contains(
        "|".join(keywords), case=False)]

    return filtered_df


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


def score(path):
    output_csv_file = 'sentiment_analysis_results.csv'

    df = pd.read_csv(
        filepath_or_buffer=path, encoding="utf-8", on_bad_lines="skip")

    filtered_df = filter_tweets(
        df, ["juststop_oil", "juststopoil", "xrebellion", "#xrebellion", "#juststopoil", "@juststop_oil", "@#xrebellion", "greenpeace", "scientistrebellion"])
    print(len(filtered_df))

    sentiment_df = perform_sentiment_analysis(filtered_df)
    print(len(sentiment_df))
    print(sentiment_df["Sentiment"].unique())

    return sentiment_df


if __name__ == "__main__":
    path_A = "data\\tweets\\left-regular.csv"
    path_B = "data\\tweets\\right-regular.csv"

    sent_A = score(path_A)
    sent_B = score(path_B)

    t_stat, p_value = ttest_ind(sent_A["Sentiment"], sent_B["Sentiment"])
    effect_size = cohens_d(sent_A["Sentiment"], sent_B["Sentiment"])

    print("p-value", p_value)
    print("effect size", effect_size)

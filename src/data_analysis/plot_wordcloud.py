import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
import re


emfd = pd.read_csv(
    "emfdscore\emfdscore\dictionaries\emfd_scoring.csv")

emfd_words = set(emfd["word"])


def filter_tweets(tweets_df, keyword):
    if keyword == "":
        return tweets_df

    filtered_df = tweets_df[tweets_df["tweet_text"].str.contains(
        keyword, case=False)]

    print(filtered_df.head())
    return filtered_df


def plot_word_cloud(df, keyword="", moral=False, count=0):
    filtered_df = filter_tweets(df, keyword)

    # Combine all the text data into a single string
    text_raw = " ".join(filtered_df["tweet_text"])
    text = re.sub('[^0-9a-zA-Z]+', ' ', text_raw)

    # Tokenize the text into words
    words = word_tokenize(text)

    # Get the set of English stopwords
    stop_words = set(stopwords.words("english"))

    # Add your custom words here
    custom_stopwords = ["brexit", "https",
                        "amp", "see", "back", "look", "t", "co", "u"]
    stop_words.update(custom_stopwords)

    # Remove stopwords from the list of words
    filtered_words = [word for word in words if word.lower(
    ) not in stop_words and words.count(word) > count]

    if moral:
        # moral_words = list(set(filtered_words) & emfd["word"])
        moral_words = emfd[emfd["word"].isin(filtered_words)]["word"]
        # Create a single string from the filtered words
        filtered_text = " ".join(moral_words)
        words = moral_words

    filtered_text = " ".join(filtered_words)
    words = filtered_text

    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400,
                          background_color="white").generate(filtered_text)

    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    path = "data\\tweets\\internal-regular.csv"

    # Read the CSV file into a DataFrame
    df = pd.read_csv(
        filepath_or_buffer=path, encoding="utf-8", on_bad_lines="skip")

    plot_word_cloud(df, "#climatechange", False, 20)

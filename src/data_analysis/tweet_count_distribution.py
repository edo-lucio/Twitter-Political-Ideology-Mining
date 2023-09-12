import pandas as pd
import matplotlib.pyplot as plt

def tweets_count_distribution(tweets_df):
    # Compute the tweet counts for each user
    tweet_counts = tweets_df.groupby('user_id').size()

    # Create a boxplot of the tweet counts
    fig, ax = plt.subplots()
    ax.boxplot(tweet_counts, whis=(0, 100))

    # Set the labels and title
    ax.set_ylabel("Number of tweets")
    ax.set_title("Distribution of tweet counts")

    # Add quartile values to the plot
    quartiles = tweet_counts.quantile([0.25, 0.5, 0.75])
    ax.text(1.1, quartiles[0.25], f"Q1: {quartiles[0.25]}")
    ax.text(1.1, quartiles[0.5], f"Q2: {quartiles[0.5]}")
    ax.text(1.1, quartiles[0.75], f"Q3: {quartiles[0.75]}")

    # Display the plot
    plt.show()

if __name__ == "__main__":
    # Load the CSV file
    tweets_df = pd.read_csv("data\\scores\\frame-axis\\raw\\thecarbontrust-regular-frame-axis.csv")
    tweets_count_distribution(tweets_df)

        
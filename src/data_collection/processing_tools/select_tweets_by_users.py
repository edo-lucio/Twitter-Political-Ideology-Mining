import pandas as pd


def select_tweets(tweets_path, users_path):
    tweets_df = pd.read_csv(
        tweets_path, encoding="utf-8", on_bad_lines="skip")

    users_df = pd.read_csv(
        users_path, encoding="utf-8", on_bad_lines="skip", header=None)[0].to_list()

    tweets_subset = tweets_df[~tweets_df["user_id"].isin(users_df)].to_csv()
    return tweets_subset


if __name__ == "__main__":
    tweets_path = "data\\tweets\\internal-regular.csv"
    users_path = "data\\users-list\\raw\\greenpeaceuk-followers-list.csv"

    select_tweets(tweets_path, users_path)

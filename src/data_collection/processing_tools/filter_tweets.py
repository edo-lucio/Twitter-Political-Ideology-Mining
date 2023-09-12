import pandas as pd


def filter_tweets(tweets_df, keyword):
    filtered_df = tweets_df[tweets_df["tweet_text"].str.contains(
        keyword, case=False)]

    print(filtered_df.head())
    return filtered_df


if __name__ == "__main__":
    files = ["data\\tweets\\left-regular.csv",
             "data\\tweets\\right-regular.csv"]
    keyword = "#BlackLivesMatter"

    for f in files:
        df = pd.read_csv(f, on_bad_lines='skip', encoding='utf-8')
        filtered_df = filter_tweets(df, keyword)
        filtered_df.to_csv(path_or_buf=f.split(".")[0]+keyword,
                           mode="a", index=False, header=None)

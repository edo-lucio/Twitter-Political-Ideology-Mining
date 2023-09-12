import pandas as pd


def aggregate_accounts_scores(df, output_file, number_of_tweets=0, specifics=None):
    ''' create a new csv file with mf scores means for each account '''

    output_file = f"{output_file}\\{name}-aggregated.csv"  # ! REMOVE /TESTING/

    columns = list(df.columns)

    user_ids = set()
    ids_list = df["user_id"].to_list()

    for id in ids_list:
        user_ids.add(id)

    result_df = pd.DataFrame(columns=columns)
    result_df.to_csv(path_or_buf=output_file, index=None)

    for user_id in list(user_ids):
        # select rows that matches the user id
        user_data = df.loc[df["user_id"] == user_id]
        if len(user_data) < number_of_tweets:
            continue

        # select the scores and take the row mean
        user_scores = user_data.loc[:, user_data.columns != 'user_id']
        score_means = list(user_scores.mean(axis=0))

        user_data = [user_id] + score_means
        result_df = pd.DataFrame(columns=columns, data=[user_data])

        result_df.to_csv(path_or_buf=output_file,
                         mode="a", header=None, index=None)


def process_scores():
    pass


if __name__ == "__main__":
    input_files = []
    output_file = "data\\scores\\frame-axis\\processed_test"

    for input_file in input_files:
        df = pd.read_csv(input_file, on_bad_lines='skip', encoding='utf-8')

        name = input_file.split("\\")[-1]
        # ! REMOVE /TESTING/
        output_file = f"{output_file}\\{name}-processed.csv"

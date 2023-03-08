# from processing_tools.tools import aggregate_accounts_scores
import pandas as pd

def aggregate_accounts_scores(input_file, output_path, number_of_tweets=0, specifics=None):
    ''' create a new csv file with mf scores means for each account '''

    print(f"Aggregating {input_file}")

    name = input_file.split("\\")[-1]
    input_file = f"{input_file}.csv" # ! REMOVE /TESTING/
    output_file = f"{output_path}\\{name}-aggregated.csv" # ! REMOVE /TESTING/

    original_df = pd.read_csv(input_file, on_bad_lines='skip', encoding='utf-8')
    columns = list(original_df.columns)

    user_ids = set()
    ids_list = original_df["user_id"].to_list()

    for id in ids_list:
        user_ids.add(id)

    result_df = pd.DataFrame(columns=columns)
    result_df.to_csv(path_or_buf=output_file, index=None)

    for user_id in list(user_ids):
        # select rows that matches the user id
        user_data  = original_df.loc[original_df["user_id"] == user_id]
        if len(user_data) < number_of_tweets: continue

        # select the scores and take the row mean
        user_scores = user_data.loc[:, user_data.columns != 'user_id']
        score_means = list(user_scores.mean(axis=0))

        user_data = [user_id] + score_means
        result_df = pd.DataFrame(columns=columns, data=[user_data])

        result_df.to_csv(path_or_buf=output_file, mode="a", header=None, index=None)

if __name__ == "__main__":
    input_files = ["data-collection\\data\\scores\\emfd-scoring\\UKLabour-regular", "data-collection\\data\\scores\\emfd-scoring\\Conservatives-regular"]
    output_path = "data-collection\\data\\scores\\emfd-scoring\\aggregated" # e.g. data-collection//data//folder_

    for file in input_files:
        aggregate_accounts_scores(file, output_path)





import os
import pandas as pd
import random
from pathlib import Path

from writers.api_wrapper.twitter_api_v1 import TwitterAPI

from dotenv import load_dotenv
load_dotenv()

dirname = os.path.dirname(__file__)
bearer_tokens = os.environ.get("BEARER1")
api = TwitterAPI(bearer_tokens)

def remove_intersections(input_files):
    '''
    - takes a tuple of paths of the data to process
    - creates 2 csv files named after the twitter accounts as inputs
    - the ouput csv files contain the symmetric difference between the elements of the two input files  
    '''

    # read data
    data = [pd.read_csv(input_file, header=None)[0] for input_file in input_files]

    # create output file
    file_names = [file.split("/")[-1].split(".")[0].split("-")[0] for file in input_files]
    dir = f"data-collection//data//users-list//no-intersection-pairs//{file_names[0] + '-' + file_names[1]}"
    os.mkdir(dir)
    output_files = [f"{dir}//{file_name}-followers-list.csv" for file_name in file_names]

    # remove intersections
    processed_array1 = list(set(data[0]) - set(data[1]))
    processed_array2 = list(set(data[1]) - set(data[0]))

    dataframes = [pd.DataFrame(processed_array1), pd.DataFrame(processed_array2)]

    dataframes[0].to_csv(path_or_buf=output_files[0], index=False, header=None)
    dataframes[1].to_csv(path_or_buf=output_files[1], index=False, header=None)

def tweets_subset_user_condition(input_path, output_folder, number_of_tweets=float("inf"), conditions=None, specifics=None):
    print("Reading {input} with {specifics} condition".format(input=input_path, specifics = specifics))

    name = input_path.split("\\")[-1].split(".")[0]

    output_folder = output_folder + '\\' + name
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    output_file = f"{output_folder}//{name}-{specifics}.csv" # ! REMOVE /TESTING/

    original_df = pd.read_csv(input_path, on_bad_lines='skip', encoding='utf-8')

    columns = list(original_df.columns)

    # raw list of ids (with repetitions)
    user_ids = original_df["user_id"].to_list()
    random.shuffle(user_ids)

    # instantiate ids set
    account_ids = set()

    # populate ids set
    for id in user_ids:
        account_ids.add(id)

    # array of tweets that meet users condition
    counter = 0

    # make chunks
    chunk_size = 100 
    chunks = [list(account_ids)[i:i+chunk_size] for i in range(0, len(account_ids), chunk_size)]
    
    result_df = pd.DataFrame(columns=columns)
    result_df.to_csv(path_or_buf=output_file, index=None)

    for user_ids in chunks:
        if counter >= number_of_tweets: break

        # get the user informations and filter them
        users_batch = api.show_users(user_ids)
        users_matching_conditions = [user for user in users_batch if all(condition.apply(user) for condition in conditions)]
        filtered_df_batches = [original_df[original_df['user_id'] == user["id"]] for user in users_matching_conditions]

        # update the counter
        counter += sum(map(len, filtered_df_batches))
        print(f"saved {counter} tweets")

        for df in filtered_df_batches:
            df.to_csv(path_or_buf=output_file, mode="a", header=None, index=None)

def aggregate_accounts_scores(input_path, output_path, number_of_tweets=0, specifics=None):
    ''' create a new csv file with mf scores means for each account '''

    print(f"Aggregating {input_path}")
    input_file = f"data_collection/data/{input_path}.csv" # ! REMOVE /TESTING/
    output_file = f"data_collection/data/{output_path}_{specifics}.csv" # ! REMOVE /TESTING/

    original_df = pd.read_csv(input_file, on_bad_lines='skip', encoding='utf-8')
    print(original_df.head())
    return
    columns = list(original_df.columns)

    user_ids = set()

    ids_list = original_df["user_id"].to_list()
    # random.shuffle(ids_list)

    for id in ids_list:
        user_ids.add(id)

    result_df = pd.DataFrame(columns=columns)
    result_df.to_csv(path_or_buf=output_file, index=None)

    print("Good News")

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

# if __name__ == "__main__":
#     # paths of the two csv to remove intersections from
#     path_1 = "data-collection//data//users-list//UKLabour-followers-list.csv"
#     path_2 = "data-collection//data//users-list//Conservatives-followers-list.csv"

#     paths = (path_1, path_2)
#     remove_intersections(paths)



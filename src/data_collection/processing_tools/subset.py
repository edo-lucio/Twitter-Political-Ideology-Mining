import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from writers.conditions import Conditions
from writers.api_wrapper.twitter_api_v1 import TwitterAPI

from pathlib import Path

import pandas as pd

import operator
import random

from dotenv import load_dotenv
load_dotenv()

dirname = os.path.dirname(__file__)
bearer_tokens = os.environ.get("BEARER1")
api = TwitterAPI(bearer_tokens)

def tweets_subset_user_condition(input_path, output_folder, number_of_tweets=float("inf"), conditions=None, specifics=None):
    print("Reading {input} with {specifics} condition".format(input=input_path, specifics = specifics))

    name = input_path.split("\\")[-1].split(".")[0]

    output_folder = output_folder + '\\' + name
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    output_file = f"{output_folder}\\{name}-{specifics}.csv" # ! REMOVE /TESTING/

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

def subset(df, conditions=None):

    print(df.head())

    # instantiate ids set
    account_ids = set()
    user_ids = df["user_id"].to_list()

    # populate ids set
    for id in user_ids:
        account_ids.add(id)

    # array of tweets that meet users condition
    counter = 0

    # make chunks
    chunk_size = 100 
    chunks = [list(account_ids)[i:i+chunk_size] for i in range(0, len(account_ids), chunk_size)]

    result_df = pd.DataFrame(columns=df.columns)
    
    for user_ids in chunks:
        # get the user informations and filter them
        users_batch = api.show_users(user_ids)
        users_matching_conditions = [user for user in users_batch if all(condition.apply(user) for condition in conditions)]
        filtered_df_batches = [df[df['user_id'] == user["id"]] for user in users_matching_conditions]

        # update the counter
        counter += sum(map(len, filtered_df_batches))
        print(f"saved {counter} tweets")

        for df in filtered_df_batches:
            pd.concat([result_df, df])

if __name__ == "__main__":
    input_files = ["data\\scores\\frame-axis\\aggregated\\tradbritgroup-regular-frame-axis-aggregated.csv", 
                   "data\\scores\\frame-axis\\aggregated\\hopenothate-regular-frame-axis-aggregated.csv"]
    
    output_path = "data\\scores\\frame-axis\\aggregated\\filtered" # e.g. data-collection//data//folder_

    # condition
    condition = "followers_count"
    threshold_values = [85, 170, 335, 770, 1440, 4000]

    for file in input_files:
        for value in threshold_values:
            conditions = [Conditions(condition, operator.lt, value, True)]
            tweets_subset_user_condition(file, output_path, conditions=conditions, specifics=f"{value}-{condition}")


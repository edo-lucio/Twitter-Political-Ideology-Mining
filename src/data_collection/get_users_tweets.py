import os
import pandas as pd
import random
from utilities.utils import write_file
from writers.writers import Writer
from writers.conditions import Conditions
import threading
import operator
import time

from dotenv import load_dotenv
load_dotenv()

# set tokens
bearer_tokens = os.environ.get("BEARER1")

# instantiate class to write tweets to csv file
writer = Writer(bearer_tokens)
dirname = os.path.dirname(__file__)

# lock to avoid problems when writing to csv files later on
lock = threading.Lock()

# constant thread count to use to write tweets
THREAD_COUNT = 40


def collect_tweets(input_files, min_tweets=float("inf"), conditions=None, specifics='regular', tweets_per_account=float("inf")):
    '''
    Collect tweets from a list of account ids and write them into a csv file.
    Uses 

    :param account_ids:
        List of Twitter account ids
    :type account_ids: ``int[]``

    :param min_tweets:
        Optional, minimum tweets to retrieve from all accounts
    :type min_tweets: ``int``

    :param \**kwargs:

    :Keyword Arguments:
        * *condition* (``function``) --
        Function that returns a boolean value if the inner condition is met.
        Conditions from condition.py module will check whether tweets met a 
        specific condition.

        * *condition_value* (``any``) --
          Value to compare the entry data to inside the condition
    '''

    threads = []

    for input_file in input_files:

        # create output file name
        file_name = input_file.split("\\")[-1].split(".")[0].split("-")[0]
        output_name = f"data\\tweets\\{file_name}-{specifics}.csv"

        output_file = open(output_name, "a", encoding="utf-8")

        # write file header
        write_file(output_file, "user_id", "tweet_text")
        output_file.close()

        # followers list
        account_ids = pd.read_csv(input_file, header=None)[0].to_list()
        random.shuffle(account_ids)

        # divide followers list into chunks
        chunk_size = len(account_ids) // THREAD_COUNT
        chunks = [account_ids[i:i+chunk_size]
                  for i in range(0, len(account_ids), chunk_size)]

        # add a thread for each chunk for each
        threads += [threading.Thread(target=writer.tweets_writer, args=(
            output_name, chunk, min_tweets, conditions, tweets_per_account, lock)) for chunk in chunks]

    for t in threads:
        t.start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    user_lists_paths = ["data\\users-list\\raw\\l-users.csv",
                        "data\\users-list\\raw\\r-users.csv"]

    # set condition
    # conditions = [Conditions("user.followers_count", operator.lt, 40, True)]

    start = time.time()
    collect_tweets(user_lists_paths, 4000000)
    end = time.time()

    print("Running time: ", end - start)

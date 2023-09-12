from writers.api_wrapper.twitter_api_v1 import TwitterAPIV2
import os
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

# set: 
bearer_tokens = os.environ.get("BEARER1") # tokens
api = TwitterAPIV2(bearer_tokens) # api

def collect_mentions(user_id):
    mentions = api.get_mention_timeline(user_id)
    result_df = pd.DataFrame(columns=["user_id", "tweet_text"])

    for mentions_batch in mentions:
        print(mentions_batch[0])
        user_ids = [batch["user"]["id"] for batch in mentions_batch]



if __name__ == "__main__":
    id = 1471232646294474763
    collect_mentions(id)

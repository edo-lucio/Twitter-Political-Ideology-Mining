from writers.api_wrapper.twitter_api_v1 import TwitterAPI
from writers.conditions import conditions_handler
import os
import pandas as pd

dirname = os.path.dirname(__file__) 

class Writer(TwitterAPI):
    def __init__(self, bearer_tokens_raw):
        super().__init__(bearer_tokens_raw)
        self.row_counts = {}

    def tweets_writer(self, output_file, account_ids, min_tweets, conditions, tweets_per_account, lock):
        output_name = output_file.split("/")[-1]
        self.row_counts[output_name] = 0

        # iterate through ids list
        for id in account_ids:
            # get tweets
            for tweets_raw_batch in self.new_get_tweets(id, tweets_per_account):

                # filter tweets
                filtered_batch = conditions_handler(tweets_raw_batch, conditions)
                if len(filtered_batch) == 0: break
                tweets_text_batch = ['"' + d["full_text"].replace('\r', '').replace('\n', '').replace("\r\n", '') + '"' for d in filtered_batch if d["full_text"]]

                # write into csv
                with lock:
                    user_id = [filtered_batch[0]["user"]["id"]] * len(tweets_text_batch)
                    
                    tweets_batch_df = pd.DataFrame({"user_id": user_id, "tweet_text": tweets_text_batch})
                    tweets_batch_df.to_csv(path_or_buf=output_file, mode="a", index=False, header=None, encoding="utf-8")

                    self.row_counts[output_name] += len(tweets_batch_df)
                    print(f"Gathered {self.row_counts[output_name]} tweets for {output_name}!")

                    if self.row_counts[output_name] >= min_tweets:
                        print(f"{output_name} gathered all tweets !")
                        return


    def followers_writer(self, account, number_of_followers, output_file, lock):
        output_name = output_file.split("/")[-1]
        self.row_counts[output_name] = 0

        for followers_ids in self.get_followers_ids(account):
            with lock:
                followers_df = pd.DataFrame(followers_ids)
                followers_df.to_csv(path_or_buf=output_file, mode="a", index=False, header=None)

                self.row_counts[output_name] += len(followers_ids)
                print(f"Gathered {self.row_counts[output_name]} followers for {output_name}!")

                if number_of_followers <= self.row_counts[output_name]:
                    print(f"Followers gathering for {output_name} is completed")
                    return


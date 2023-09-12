import requests
from tokenizer import Tokenizer
import time
import os

from dotenv import load_dotenv
load_dotenv()

# set tokens
bearer_tokens = os.environ.get("BEARER1")


class TwitterAPIV2(Tokenizer):
    def __init__(self, bearer_tokens_raw):
        super().__init__(bearer_tokens_raw)

    def get_mention_timeline(self, id, number_of_mentions=float("inf")):
        url = "https://api.twitter.com/2/users/{}/mentions".format(id)
        params = {
            "tweet.fields": "created_at",
            "max_results": 100
        }

        headers = {
            "Authorization": "Bearer {}".format(self.bearer_token)
        }

        pagination_token = True
        counter = 0

        while pagination_token:
            try:
                if counter >= number_of_mentions:
                    break
                response = requests.get(url, headers=headers, params=params)

                if response.status_code != 200:
                    raise Exception(response)

                response_json = response.json()

                # set pagination token for next request if any
                pagination_token = response_json["meta"]["next_token"] if "next_token" in response_json["meta"].keys(
                ) else None
                params["pagination_token"] = pagination_token

                # access data
                mentions = response_json["data"]
                yield mentions

            except Exception as e:
                # rate limit error: switch bearer token
                if str(e.status_code) == "429":
                    print(f"HTTP ERROR {e.status_code}: {e.text}")
                    headers = self._change_header()
                    time.sleep(8)

        # get tweets from id

    def get_tweets(self, id, number_of_tweets=("inf")):
        try:
            print(self.bearer_token)
            max_id = None

            url = "https://api.twitter.com/2/users/{}/tweets".format(id)
            params = {
                "count": 200,
                "exclude_replies": False,
                "include_rts": False,
                "max_id": max_id,
                "tweet_mode": "extended"
            }

            headers = {
                "Authorization": "Bearer {}".format(self.bearer_token)
            }

            # response = requests.get(url, params=params, headers=headers)
            response = requests.get(url, headers=headers, params=params)

            if (response.status_code != 200):
                raise Exception(response.status_code)

            tweets = response.json()
            return tweets

        except Exception as e:
            response = int(str(e))

            if response == 401:
                self._change_header()

            if response != 429:
                return []

            if self.bearer_index == len(self.bearer_tokens) - 1:
                self.bearer_index = 0
                self.bearer_token = self.bearer_tokens[self.bearer_index]
                self.get_tweets(id, max_id=None)

                self.bearer_index += 1
                self.bearer_token = self.bearer_tokens[self.bearer_index]
                self.get_tweets(id, max_id=None)


def test():
    api = TwitterAPIV2(bearer_tokens)
    mentions = api.get_tweets(1471232646294474763)

    i = 0
    for mentions_batch in mentions:
        print([(i, mention["text"]) for mention in mentions_batch])
        i += 1


if __name__ == "__main__":
    test()

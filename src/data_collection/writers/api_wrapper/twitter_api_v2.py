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
                if counter >= number_of_mentions: break
                response = requests.get(url, headers=headers, params=params)

                if response.status_code != 200:
                    raise Exception(response)

                response_json = response.json()

                # set pagination token for next request if any
                pagination_token = response_json["meta"]["next_token"] if "next_token" in response_json["meta"].keys() else None
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

        

def test():
    api = TwitterAPIV2(bearer_tokens)
    mentions = api.get_mention_timeline(1471232646294474763)

    i = 0
    for mentions_batch in mentions:
        print([(i, mention["text"]) for mention in mentions_batch])
        i += 1

if __name__ == "__main__":
    test()
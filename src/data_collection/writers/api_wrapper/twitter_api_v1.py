import requests
import json
import asyncio
from functools import partial
import time

loop = asyncio.get_event_loop()


class Tokenizer:
    def __init__(self, bearer_tokens_raw):
        self.bearer_tokens = [
            bearer for bearer in bearer_tokens_raw.split(".")]
        self.bearer_index = 0
        self.bearer_token = self.bearer_tokens[self.bearer_index]

    def _change_header(self):
        if self.bearer_index == len(self.bearer_tokens) - 1:
            self.bearer_index = 0
            self.bearer_token = self.bearer_tokens[self.bearer_index]
        else:
            self.bearer_index += 1
            self.bearer_token = self.bearer_tokens[self.bearer_index]

        headers = {
            "Authorization": "Bearer {}".format(self.bearer_token)
        }

        return headers


class TwitterAPI(Tokenizer):
    def __init__(self, bearer_tokens_raw):
        super().__init__(bearer_tokens_raw)

    def _set_rules(self, rules):
        # You can adjust the rules if needed

        try:
            payload = {"add": rules}
            response = requests.post(
                "https://api.twitter.com/2/tweets/search/stream/rules",
                auth=self.bearer_token,
                json=payload,
            )
            print(json.dumps(response.json()))

        except Exception as e:
            print("Cannot add rules (HTTP {}): {}".format(e.status_code, e.text))

    def _delete_rules(self, rules):
        if rules is None or "data" not in rules:
            return None

        try:
            ids = list(map(lambda rule: rule["id"], rules["data"]))
            payload = {"delete": {"ids": ids}}
            response = requests.post(
                "https://api.twitter.com/2/tweets/search/stream/rules",
                auth=self.bearer_token,
                json=payload
            )

            print(json.dumps(response.json()))

        except Exception as e:
            print("Cannot delete rules (HTTP {}): {}".format(
                e.status_code, e.text
            ))

    def _get_rules(self):
        try:
            response = requests.get(
                "https://api.twitter.com/2/tweets/search/stream/rules", auth=self.bearer_token)
            print(json.dumps(response.json()))
            return response.json()

        except Exception as e:
            print("Cannot get rules (HTTP {}): {}".format(e.status_code, e.text))

    def _build_value_string(self, ids):
        rule = "from:{}".format(ids[0])
        for id in ids:
            rule + "OR from:{}".format(id)

        return rule

    def stream_users_tweets(self, ids):
        headers = {
            'Authorization': 'Bearer ' + self.bearer_token,
            'User-Agent': 'MyApp/1.0.0'
        }

        rules = self._get_rules()
        delete = self._delete_rules(rules)
        new_rules = self._build_value_string(ids)

        self._set_rules([{"value": new_rules}])
        url = "https://api.twitter.com/2/tweets/search/stream"

        # Start streaming

        try:
            response = requests.get(url, headers=headers, stream=True)

            # Iterate over tweets in the stream
            for r in response.iter_lines():
                if r:
                    print(r)

        except Exception as e:
            if (e.status_code == 401 or e.status_code == 429):
                if self.bearer_index == len(self.bearer_tokens) - 1:
                    self.bearer_index = 0
                    return self.stream_users_tweets(self, ids)

                self.bearer_index += 1
                return self.stream_users_tweets(self, ids)

    # get tweets from id
    async def get_tweets(self, id, number_of_tweets=("inf")):
        try:
            max_id = None

            url = "https://api.twitter.com/1.1/statuses/user_timeline.json"
            params = {
                "user_id": id,
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
            response = await loop.run_in_executor(None, partial(requests.get, url, headers=headers, params=params))

            if (response.status_code != 200):
                raise Exception(response.status_code)

            tweets = response.json()
            return tweets

        except Exception as e:
            print(f"Request failed with status code {e}")
            response = int(str(e))

            if response != 429:
                return []

            if self.bearer_index == len(self.bearer_tokens) - 1:
                self.bearer_index = 0
                self.bearer_token = self.bearer_tokens[self.bearer_index]
                await self.get_tweets(id, max_id=None)

                self.bearer_index += 1
                self.bearer_token = self.bearer_tokens[self.bearer_index]
                await self.get_tweets(id, max_id=None)

    def get_followers_ids(self, account, number_of_followers=float("inf")):
        '''
        returns specified number of followers ids from account
        '''
        # variable for paginating through data
        cursor = -1

        url = "https://api.twitter.com/1.1/followers/ids.json"
        params = {
            "screen_name": account,
            "count": 5000,
            "cursor": cursor
        }

        headers = {
            "Authorization": "Bearer {}".format(self.bearer_token)
        }

        results_followers = []

        while cursor:
            try:
                if len(results_followers) >= number_of_followers:
                    break

                # start the requests
                # response = await loop.run_in_executor(None, partial(requests.get, url, headers=headers, params=params))
                response = requests.get(
                    url=url, params=params, headers=headers)

                # if response has gone wrong throw status code exception
                if (response.status_code != 200):
                    raise Exception(response.status_code)

                # parse the response
                response = response.json()
                followers_ids = response["ids"]
                results_followers += followers_ids

                print("Followers gathered", len(results_followers), account)

                # switch to next cursor
                cursor = response["next_cursor"]
                params["cursor"] = cursor
                print(cursor)

                # return results
                yield followers_ids

            except Exception as e:
                # rate limit error: switch bearer token
                if str(e) == "429":
                    print(f"HTTP ERROR {e}: too many requests")
                    headers = self._change_header()
                    time.sleep(8)

    def show_users(self, ids):
        try:
            url = "https://api.twitter.com/1.1/users/lookup.json"
            params = {
                "user_id": ids,
            }

            headers = {
                "Authorization": "Bearer {}".format(self.bearer_token)
            }

            response = requests.post(url, headers=headers, data=params)

            informations = response.json()
            return informations

        except Exception as e:

            print(e)
            # auth error: user is protected
            if str(e) == "401":
                print(f"HTTP ERROR {e}: user is protected")
                return

            # rate limit error: switch bearer token
            if str(e) == "429":
                print(f"HTTP ERROR {e}: too many requests")
                headers = self._change_header()
                time.sleep(8)

    # get tweets from id
    def new_get_tweets(self, id, number_of_tweets=float("inf")):
        url = "https://api.twitter.com/1.1/statuses/user_timeline.json"
        params = {
            "user_id": id,
            "count": 200,
            "exclude_replies": False,
            "include_rts": False,
            "max_id": None,
            "tweet_mode": "extended"
        }

        headers = {
            "Authorization": "Bearer {}".format(self.bearer_token)
        }

        max_id = True
        counter = 0

        while max_id:
            try:
                if counter >= number_of_tweets:
                    break

                # get tweets request
                # response = await loop.run_in_executor(None, partial(requests.get, url, headers=headers, params=params))
                response = requests.get(
                    url=url, headers=headers, params=params)

                # throw exception if response returns an http error
                if response.status_code != 200:
                    raise Exception(response.status_code)

                # parse the response
                tweets = response.json()

                # if no tweets are retrieved return
                if len(tweets) == 0:
                    break

                # update counter
                counter += len(tweets)

                # access last pagination token id
                max_id = tweets[len(tweets) - 1]["id"]

                # if new pagination token id is the same as last one return
                if params["max_id"] == max_id:
                    return

                time.sleep(0.8)

                # return the results
                params["max_id"] = max_id
                yield tweets

            except Exception as e:

                # auth error: user is protected
                if str(e) == "401":
                    print("HTTP ERROR {e}: user is protected")
                    return

                # rate limit error: switch bearer token
                if str(e) == "429":
                    print(f"HTTP ERROR {e}: too many requests")
                    headers = self._change_header()
                    time.sleep(8)

    async def get_tweet_info(self, account_name, tweet_id):
        try:
            url = "https://api.twitter.com/1.1/search/tweets.json"

            params = {
                "q": [f"to:{account_name}", f"sinceId: {tweet_id}"],
                "count": 100,
                "tweet_mode": "extended",
                "exclude_replies": False
            }

            headers = {
                "Authorization": "Bearer {}".format(self.bearer_token)
            }

            # response = requests.get(url, params=params, headers=headers)
            response = await loop.run_in_executor(None, partial(requests.get, url, headers=headers, params=params))

            if (response.status_code != 200):
                raise Exception(response.status_code)

            tweets = response.json()
            return tweets

        except Exception as e:
            print(f"Request failed with status code {e}")
            response = int(str(e))

            if response != 429:
                return []

            if self.bearer_index == len(self.bearer_tokens) - 1:
                self.bearer_index = 0
                self.bearer_token = self.bearer_tokens[self.bearer_index]
                await self.get_tweets(id, max_id=None)

                self.bearer_index += 1
                self.bearer_token = self.bearer_tokens[self.bearer_index]
                await self.get_tweets(id, max_id=None)


class TwitterAPIV2(Tokenizer):
    def __init__(self, bearer_tokens_raw):
        super().__init__(bearer_tokens_raw)

    def get_mention_timeline(self, id, number_of_mentions=float("inf")):
        url = "https://api.twitter.com/2/users/{}/mentions".format(id)
        params = {
            "tweet.fields": "created_at,author_id",
            "max_results": 100,
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


# accounts = pd.read_csv("data_collection/data/users/UKLabour-followers-list.csv",  header=None)
api = TwitterAPI(
    "AAAAAAAAAAAAAAAAAAAAAIlmlAEAAAAANxYXGRxIQUdG%2Bvm5QSIkBcBjjhY%3Dy7i34oggfw6Jz8X5D85FvyGxFJbY21Ff21K81f4u71vrr50BuI")
# async def test():
#     i = await api.show_users([307423967])
#     print(i[0]["statuses_count"])

#     result_tweets = []

#     async for tweets in api.new_get_tweets(307423967):
#         result_tweets += tweets

#     for tweet in result_tweets:
#         print(tweet["full_text"])

#     print(len(result_tweets))

# loop.run_until_complete(test())

########################################################################


def testV1():
    user_id = "1872495530"

    for tweets in api.new_get_tweets(user_id):
        print(tweets[0])

# testV1()


api_v2 = TwitterAPIV2(
    "AAAAAAAAAAAAAAAAAAAAAIlmlAEAAAAANxYXGRxIQUdG%2Bvm5QSIkBcBjjhY%3Dy7i34oggfw6Jz8X5D85FvyGxFJbY21Ff21K81f4u71vrr50BuI")


def testV2():
    user_id = "1872495530"
    number_of_mentions = 0

    for mentions in api_v2.get_mention_timeline(user_id):
        number_of_mentions += len(mentions)
        print(mentions[0]["text"])

    print(number_of_mentions)


# testV2()

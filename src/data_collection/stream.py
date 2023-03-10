import requests
import os
import json
from dotenv import load_dotenv
import pandas as pd
load_dotenv()

# To set your enviornment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = os.environ.get("BEARER1")
bearer_token = [i for i in bearer_token.split(".")][0]

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2FilteredStreamPython"
    return r


def get_rules():
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream/rules", auth=bearer_oauth
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot get rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    print(json.dumps(response.json()))
    return response.json()


def delete_all_rules(rules):
    if rules is None or "data" not in rules:
        return None

    ids = list(map(lambda rule: rule["id"], rules["data"]))
    payload = {"delete": {"ids": ids}}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        auth=bearer_oauth,
        json=payload
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot delete rules (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    print(json.dumps(response.json()))


def set_rules(rules):
    # You can adjust the rules if needed
    sample_rules = rules
    # [
    #     {"value": "dog has:images", "tag": "dog pictures"},
    #     {"value": "cat has:images -grumpy", "tag": "cat pictures"},
    # ]
    payload = {"add": sample_rules}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        auth=bearer_oauth,
        json=payload,
    )
    if response.status_code != 201:
        raise Exception(
            "Cannot add rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    print(json.dumps(response.json()))


def create_rule(ids):
    rule = "from:{}".format(ids[0])
    for id in ids:
        rule + "OR from:{}".format(id)

    return [{"value": rule}]

def get_stream(set):
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream", auth=bearer_oauth, stream=True,
    )
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Cannot get stream (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    for response_line in response.iter_lines():
        if response_line:
            json_response = json.loads(response_line)
            print(json.dumps(json_response, indent=4, sort_keys=True))


def main():
    rules = get_rules()

    delete = delete_all_rules(rules)

    accounts_df = pd.read_csv("data_collection/data/users/UKLabour-followers-list.csv", header=None)
    accounts = [1379114005487366153]
    new_rules = create_rule(accounts)

    set = set_rules(new_rules)
    get_stream(set)


if __name__ == "__main__":
    main()
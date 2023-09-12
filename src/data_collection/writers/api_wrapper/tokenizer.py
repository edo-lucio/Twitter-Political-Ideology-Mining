class Tokenizer:
    def __init__(self, bearer_tokens_raw):
        self.bearer_tokens = [bearer for bearer in bearer_tokens_raw.split(".")]
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
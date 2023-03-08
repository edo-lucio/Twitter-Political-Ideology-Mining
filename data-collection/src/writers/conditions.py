
class Conditions:
    def __init__(self, key, operator, threshold, on_single_data=False, function=lambda x: x):
        self.key = key
        self.operator = operator
        self.threshold = threshold
        self.on_single_data = on_single_data
        self.function = function

    def apply(self, data):
        value = self._get_value(data)
        return self.operator(self.function(value), self.threshold)

    def _get_value(self, data):
        keys = self.key.split('.')
        value = data
        for k in keys:
            if k in value:
                value = value[k]
            else:
                return None
        return value

def fast_filter(tweet, conditions):
    if conditions is None:
        return True
    
    conditions = [condition for condition in conditions if condition.on_single_data == True]
    conditions_results = [condition.apply(tweet) for condition in conditions]

    if False in conditions_results:
        print("Condition Failed")
        return False
    return True

def filter(tweets, conditions):
    if conditions is None:
        return tweets
    
    return [tweet for tweet in tweets if all(condition.apply(tweet) for condition in conditions if condition.on_single_data == False)]

def conditions_handler(d, conditions):
    condition_passed = fast_filter(d[0], conditions)

    if not condition_passed:
        return []
    
    return filter(d, conditions)

    


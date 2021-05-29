import time, datetime
class CompositionTest:
    """컴포지션 사용 예시"""

    def __init__(self, policy_data, **extra_data):
        self._data = {**policy_data, **extra_data}

    def change_in_policy(self, customer_id, **new_policy_data):
        self._data[customer_id].update(**new_policy_data)

    def __getitem__(self, customer_id):
        return self._data[customer_id]

    def __len__(self):
        return len(self._data)

new_policy = CompositionTest({
    "client001": {
        "fee": 10000,
        "expiration_data": datetime.datetime(2021, 3, 20),
    }
})

print(new_policy["client001"])
print("\n")

new_policy.change_in_policy("client001", expiration_data=datetime.datetime(2021, 3, 29))
print(new_policy["client001"])

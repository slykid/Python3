import numpy as np
import pandas as pd

train = pd.read_csv("data/titanic/train.csv")
test = pd.read_csv("data/titanic/test.csv")

train.head()
train.tail()

test.head()
test.tail()

train.columns
test.columns
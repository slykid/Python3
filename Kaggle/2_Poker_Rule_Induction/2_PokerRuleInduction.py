import numpy as np
import pandas as pd

train = pd.read_csv("Kaggle/2_Poker_Rule_Induction/data/train.csv")
test = pd.read_csv("Kaggle/2_Poker_Rule_Induction/data/test.csv")

train.head()
test.head()

train.info()
test.info()

train.describe()
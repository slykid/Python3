import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras

# 라벨링 데이터 전처리
labels = pd.read_csv("data/dog-breed-identification/labels.csv")
labels["path"] = labels["id"].apply(lambda x: "data/data/dog-breed-identification/train" + x + ".jpg")


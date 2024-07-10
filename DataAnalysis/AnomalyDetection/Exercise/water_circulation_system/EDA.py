import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt

import pyod

matplotlib.use("MacOSX")
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

df = pd.read_csv("Dataset/anomaly_detection/water_circulation_system/chapter01_df.csv", sep=";")
df.head()

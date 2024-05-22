import numpy as np
import pandas as pd

from pyculiarity import detect_ts
from pyculiarity.date_utils import date_format

from matplotlib import pyplot as plt
import seaborn as sns

co2 = [315.58, 316.39, 317.82, 318.45, 319.67, 320.78, 313.55, 315.02, 317.79, 1000, 1200, 1300, 318.28, 314.04, 311.25, 319.22]

co2 = pd.Series(co2, index=pd.date_range("1-1-2022", periods=len(co2), freq="M"), name="CO2")
co2.head()

co2 = pd.DataFrame(co2).reset_index()
co2.columns = ["Time", "CO2"]
co2["Time"] = np.int64(co2["Time"])  # "날짜 형식 -> 정수형" 으로 변환
co2.head()

def plot_ts_anoms(inDF, savepath):
    fig = plt.figure(figsize=(22, 5))

    plt.plot(inDF.index, inDF["value"], alpha=0.4, label="Value")
    plt.plot(inDF.index, inDF["anorm"], color="steelblue", alpha=0.1, marker='o', markersize='7', markeredgewidth=1, markerfacecolor=None, markeredgecolor="red", label="anormalies")

    if 'expected_value' in inDF.columns:
        plt.plot(inDF.index, inDF["expected_value"], color="c")


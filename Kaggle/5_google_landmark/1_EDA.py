import os
import glob
import cv2
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

from matplotlib import pyplot as plt

# make image path
label = pd.read_csv("data/google_landmark/train.csv")
label["file_path"] = os.path.join("data", "google_landmark", "train", label["id"][0], label["id"][1], label["id"][2], label["id"] + ".jpg")

train_files = label["file_path"].values
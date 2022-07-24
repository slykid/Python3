import os
import glob
import cv2
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

from matplotlib import pyplot as plt

# make image path
label = pd.read_csv("data/google_landmark/train.csv")
label["file_path"] = label["id"].apply(lambda x: "data/google_landmark/train/" + x[0] + "/" + x[1] + "/" + x[2] + "/" + x + ".jpg")

train_files = label["file_path"].values

# load sample image
sample_img = cv2.cvtColor(cv2.imread(np.random.choice(train_files)), cv2.COLOR_BGR2RGB)
plt.imshow(sample_img)
print("image shape:" + sample_img.shape + "\n")




import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

from sklearn.model_selection import StratifiedKFold


#Parameters for Plots
plt.rcParams['figure.figsize'] = (10,6)
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.frameon'] = True
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams["font.family"] = "monospace";

# Load Data
patients_note = pd.read_csv("data/nbme_notes/patient_notes.csv")
features = pd.read_csv("data/nbme_notes/features.csv")
train = pd.read_csv("data/nbme_notes/train.csv")
test = pd.read_csv("data/nbme_notes/test.csv")

patients_note.head()
features.head()
train.head()
train.info()

# 기초 통계
## patients_note
A = sns.countplot(patients_note['case_num'], color='gray', edgecolor='white', linewidth=1.5, saturation=1.5)

patch_h = []
for patch in A.patches:
    reading = patch.get_height()
    patch_h.append(reading)

idx_tallest = np.argmax(patch_h)
A.patches[idx_tallest].set_facecolor("red")

for p in A.patches:
    A.text(p.get_x() + p.get_width() / 2., p.get_height() + 3, p.get_height(), ha="center", size=10)

plt.ylabel('Count', weight='semibold', fontname='Georgia')
plt.xlabel('Cases', weight='semibold', fontname='Georgia')
plt.suptitle('Number of Cases', fontname='Georgia', weight='bold', size=18, color="#070c23")
plt.show()

## features
A = sns.countplot(features['case_num'], color='gray', edgecolor='white', linewidth=1.5, saturation=1.5)

patch_h = []
for patch in A.patches:
    reading = patch.get_height()
    patch_h.append(reading)

idx_tallest = np.argmax(patch_h)
A.patches[idx_tallest].set_facecolor("red")

for p in A.patches:
    A.text(p.get_x() + p.get_width() / 2., p.get_height() + 0.3, p.get_height(), ha="center", size=10)

plt.ylabel('Count', weight='semibold', fontname='Georgia')
plt.xlabel('Cases', weight='semibold', fontname='Georgia')
plt.suptitle('Number of Cases', fontname='Georgia', weight='bold', size=18, color="#070c23")
plt.show()

## train
A = sns.countplot(train['case_num'], color='gray', edgecolor='white', linewidth=1.5, saturation=1.5)

patch_h = []
for patch in A.patches:
    reading = patch.get_height()
    patch_h.append(reading)

idx_tallest = np.argmax(patch_h)
A.patches[idx_tallest].set_facecolor("red")

for p in A.patches:
    A.text(p.get_x() + p.get_width() / 2., p.get_height() + 3, p.get_height(), ha="center", size=10)

plt.ylabel('Count', weight='semibold', fontname='Georgia')
plt.xlabel('Cases', weight='semibold', fontname='Georgia')
plt.suptitle('Number of Cases', fontname='Georgia', weight='bold', size=18, color="#070c23")
plt.show()


# 통합 데이터 생성: train - patients_note - features 결합
total_data = pd.merge(train, patients_note[["pn_num", "case_num", "pn_history"]], how="left", on=["case_num", "pn_num"])
total_data = pd.merge(train, features[["feature_num", "case_num", "feature_text"]], how="left", on=["case_num", "feature_num"])

# annotation, location 컬럼 중 빈 값 처리
empty_cnt = len(total_data[total_data.location == '[]'])
empty_annotations = empty_cnt / len(total_data)
print(f'number of empty annotations in train: {empty_cnt} or {empty_annotations:.2%} from train dataset')
## number of empty annotations in train: 4399 or 30.76% from train dataset

# Word Cloud
cloud = WordCloud(background_color="white", max_words=50, stopwords=set(STOPWORDS), width=600, height=250)
f, axs = plt.subplots(5, 2, figsize=(20, 20))
f.suptitle('WordCloud by case', fontsize=28)
cnt = 0
for i in range(5):
    for j in range(2):
        _sample = total_data[total_data['case_num'] == cnt]
        txt = ''.join(_sample['feature_text'].tolist()).lower()
        txt = str(txt)
        cloud = cloud.generate(txt)
        axs[i, j].imshow(cloud, interpolation='bilinear')
        axs[i, j].axis('off')
        axs[i, j].set_title(f'Case {cnt}', fontsize=20)
        cnt += 1
f.show()

# 전처리
## 함수 선언
def process_feature_text(text):
    return text.replace("-OR-", ";-").replace("-", " ")

def loc_list_to_ints(loc_list):
    to_return = []
    for loc_str in loc_list:
        loc_strs = loc_str.split(";")
        for loc in loc_strs:
            start, end = loc.split()
            to_return.append((int(start), int(end)))
    return to_return

## 데이터 전처리
total_data["feature_text"] = [process_feature_text(x) for x in total_data["feature_text"]]

total_data["feature_text"] = total_data["feature_text"].apply(lambda x: x.lower())
total_data["pn_history"] = total_data["pn_history"].apply(lambda x: x.lower())

skf = StratifiedKFold(n_splits=5)
total_data["stratify_on"] = total_data["case_num"].astype(str) + total_data["feature_num"].astype(str)
total_data["fold"] = -1
for fold, (_, valid_idx) in enumerate(skf.split(total_data["id"], y=total_data["stratify_on"])):
    total_data.loc[valid_idx, "fold"] = fold



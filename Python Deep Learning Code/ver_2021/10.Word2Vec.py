import re
import urllib.request
import zipfile
import nltk
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# 사전작업
nltk.download('punkt')

# 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml",
                           filename="data/ted_en-20160408.xml")

# 데이터 로드
data = open('data/ted_en-20160408.xml', 'r', encoding='UTF8')
text = etree.parse(data)

# 원본 데이터 전처리
parse_text = '\n'.join(text.xpath('//content/text()'))
content = re.sub(r'\([^)]*\)', '', parse_text)

# Tokenizing
sent_text = sent_tokenize(content)

noralized_text = []
for string in sent_text:
    token = re.sub(r"[^a-z0-9]+", " ", string.lower())
    noralized_text.append(token)

result = [word_tokenize(sentence) for sentence in noralized_text]
print("총 샘플 개수: {}".format(len(result)))  # 273,424 개

# 샘플 출력
for line in result[:3]:
    print(line)
# ['here', 'are', 'two', 'reasons', 'companies', 'fail', 'they', 'only', 'do', 'more', 'of', 'the', 'same', 'or', 'they', 'only', 'do', 'what', 's', 'new']
# ['to', 'me', 'the', 'real', 'real', 'solution', 'to', 'quality', 'growth', 'is', 'figuring', 'out', 'the', 'balance', 'between', 'two', 'activities', 'exploration', 'and', 'exploitation']
# ['both', 'are', 'necessary', 'but', 'it', 'can', 'be', 'too', 'much', 'of', 'a', 'good', 'thing']

# Word2Vec 학습하기
# - sentence: 학습 시킬 문장
# - vector_size: 워드 벡터의 특징 값, 임베딩 된 벡터의 차원
# - window: 주변 단어 수
# - min_count: 단어 최소 빈도 수 제한(적은 빈도를 가진 단어는 학습하지 않음)
# - workers: 학습에 사용되는 프로세스 수
# - sg: Word2Vec 방식으로 0 = CBOW, 1 = Skip-gram
word2vec = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)
model_result = word2vec.wv.most_similar("man")
print(model_result)
# [('woman', 0.8517822623252869),
#  ('guy', 0.8084050416946411),
#  ('boy', 0.7709711790084839),
#  ('girl', 0.7593225240707397),
#  ('lady', 0.7527689933776855),
#  ('gentleman', 0.7414157390594482),
#  ('kid', 0.6947308778762817),
#  ('soldier', 0.69269859790802),
#  ('son', 0.6630584597587585),
#  ('surgeon', 0.6612566709518433)]

# 모델 저장 및 로드
## 모델 저장
word2vec.wv.save_word2vec_format('eng_w2v')

## 모델 로드
loaded_model = KeyedVectors.load_word2vec_format('eng_w2v')

## 로드 결과 확인
model_result = loaded_model.most_similar("man")
print(model_result)
# [('woman', 0.8517822623252869),
#  ('guy', 0.8084050416946411),
#  ('boy', 0.7709711790084839),
#  ('girl', 0.7593225240707397),
#  ('lady', 0.7527689933776855),
#  ('gentleman', 0.7414157390594482),
#  ('kid', 0.6947308778762817),
#  ('soldier', 0.69269859790802),
#  ('son', 0.6630584597587585),
#  ('surgeon', 0.6612566709518433)]



from collections import Counter

# KAKAO Coding Demo Test 1
def solution(v):
    answer = []

    x = [v[0][0], v[1][0], v[2][0]]
    y = [v[0][1], v[1][1], v[2][1]]

    xCount = Counter(x)
    yCount = Counter(y)

    for key in xCount:
        if xCount[key] == 1:
            answer.append(key)

    for key in yCount:
        if yCount[key] == 1:
            answer.append(key)

    return answer

solution([[1,4],[3,4],[3,10]])



# KAKAO Coding Demo Test 2
a, b = map(int, input().strip().split(' '))
for i in range(0, b):
    for j in range(0, a):
        print("*", sep="", end="")
    print("\n", sep="", end="")


# 2020 -1
s = "aabbaccc"
def solution(s):
    answer = 0

    splitLen = 1 # 파싱 갯수
    phrase = []  # 파싱된 문자열 저장 변수
    lenResult = []  # 결과 값 비교를 위한 변수
    prepSentence = ""  # 처리된 문자열

    if len(s) > 1000:
        print("문자열의 길이가 초과되었습니다")
        return -1

    else:
        while splitLen < len(s):
            phrase = [s[i:i+splitLen] for i in range(0, len(s), splitLen)]

            phraseCnt = {}
            sn = 0
            for i in range(0, len(phrase)-1):
                if i == 0:
                    phraseCnt[sn] = {}
                    phraseCnt[sn]["key"] = phrase[i]
                    phraseCnt[sn]["value"] = 1

                if phrase[i].__eq__(phrase[i+1]):
                    phraseCnt[sn]["value"] += 1

                else:
                    sn += 1
                    phraseCnt[sn] = {}
                    phraseCnt[sn]["key"] = phrase[i+1]
                    phraseCnt[sn]["value"] = 1

            for i in range(0, len(phraseCnt)):
                if phraseCnt[i]["value"] == 1:
                    prepSentence = prepSentence + phraseCnt[i]["key"]
                else:
                    prepSentence =  prepSentence + str(phraseCnt[i]["value"]) + phraseCnt[i]["key"]

            lenResult.append(len(prepSentence))
            splitLen += 1
            phrase = []
            prepSentence = ""
            phraseCnt = []

        answer = min(lenResult)

    return answer

solution(s)


# 2020 - 2
answer = []
build_frame= [[0,0,0,1],[2,0,0,1],[4,0,0,1],[0,1,1,1],[1,1,1,1],[2,1,1,1],[3,1,1,1],[2,0,0,0],[1,1,1,0],[2,2,0,1]]
for i in range(0, len(build_frame)):
    x, y, a, b = build_frame[i][0], build_frame[i][1], build_frame[i][2], build_frame[i][3]

    if a == 0:
        if b == 1:
            if i > 0 and ([x - 1, y, 1] not in answer) or ([x, y - 1, 0] not in answer):
                continue
            else:
                answer.append([x, y, a])

        elif b == 0:
            if ([x - 1, y + 1, 1] in answer) and ([x + 1, y + 1, 1] in answer):
                answer.remove([x, y, a])


# 2020 - 4
import re
words = ["frodo", "front", "frost", "frozen", "frame", "kakao"]
query = "fro??"

pattern = ""
if re.search('\?+', query).start() == 0:
    pattern = '$' + query[re.search('[a-z]+', query).start():re.search('[a-z]+', query).end()]
else:
    pattern = '^' + query[re.search('[a-z]+', query).start():re.search('[a-z]+', query).end()]


cnt = re.search('\?+', query).end() - re.search('\?+', query).start()

re.search('\?{'+str(cnt)+'}', query)

"?" in query
# re.search('\?+[a-z]+\?', query) is None
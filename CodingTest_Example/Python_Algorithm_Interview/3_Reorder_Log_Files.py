# 문제
# 로그를 재정렬하라.
# 정렬 기준은 다음과 같다.
# 1) 로그의 가장 앞부분은 식별자이다.
# 2) 문자로 구성된 로그가 숫자 로그보다 앞에 온다.
# 3) 식별자는 순서에 영향을 끼치지 않지만, 문자가 동일할 경우 식별자 순으로 한다.
# 4) 숫자로그는 입력 순서대로 한다.

# [입력] logs = ["dig1 8 1 5 1", "let1 art can", "dig2 3 6", "let2 own kit dig", "let3 art zero"]
# [출력] ["let1 art can", "let2 own kit dig", "let3 art zero", "dig1 8 1 5 1", "dig2 3 6"]

logs = ["dig1 8 1 5 1", "let1 art can", "dig2 3 6", "let2 own kit dig", "let3 art zero"]

def solution(logs: list):

    letters, digits = [], []  # 문자로그, 숫자로그 저장 변수

    # 문자 로그와 숫자 로그 분리
    for log in logs:
        if log.split(" ")[1].isdigit(): # 맨 앞은 구분자, 첫 로그가 숫자인지 확인
            digits.append(log)
        else:
            letters.append(log)

    # 문자로그에 한해 순서 재배치
    letters.sort(key=lambda x : (x.split(" ")[1:], x.split(" ")[0]))

    return letters + digits  # 문자 -> 숫자 순으로 로그 배치

print(solution(logs))


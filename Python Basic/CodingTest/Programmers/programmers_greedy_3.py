# 문제 설명
# 어떤 숫자에서 k개의 수를 제거했을 때 얻을 수 있는 가장 큰 숫자를 구하려 합니다.
# 예를 들어, 숫자 1924에서 수 두 개를 제거하면 [19, 12, 14, 92, 94, 24] 를 만들 수 있습니다.
# 이 중 가장 큰 숫자는 94 입니다.

# 문자열 형식으로 숫자 number와 제거할 수의 개수 k가 solution 함수의 매개변수로 주어집니다.
# number에서 k 개의 수를 제거했을 때 만들 수 있는 수 중 가장 큰 숫자를 문자열 형태로 return 하도록
# solution 함수를 완성하세요.

# 제한 조건
# number는 1자리 이상, 1,000,000자리 이하인 숫자입니다.
# k는 1 이상 number의 자릿수 미만인 자연수입니다.

# 입출력 예
# number	    k	return
# "1924"	    2	"94"
# "1231234"	    3	"3234"
# "4177252841"	4	"775841"

# 나의 풀이 (16.5점)
def solution(number, k):
    removeCnt = 0  # 제거 횟수, k 랑 같으면 종료
    result = ""

    # 제한 사항에서 벗어나는 경우
    if len(number) < 0 or len(number) > 1000000 or k < 1 or k >= len(number):
        exit(1)

    # 조건에 맞는 경우
    else:
        # 전부 같은 숫자로 구성된 경우
        if len(set([x for x in number])) == 1:
            result = number[k:]

        # 그 외
        else:
            while removeCnt != k:
                _list = [x for x in number[0:k + 1]]

                if removeCnt == 0:
                    _idx = _list.index(max(_list))
                    removeCnt += len(number[:_idx])
                    number = number[_idx:]
                else :
                    _idx = _list.index(min(_list))
                    number = number[0:_idx] + number[_idx + 1:]
                    removeCnt += 1
            result = number
    return result


# 정답
def solution(number, k):
    stack = [number[0]]
    for num in number[1:]:
        while len(stack) > 0 and stack[-1] < num and k > 0:
            k -= 1
            stack.pop()
        stack.append(num)
    if k != 0:
        stack = stack[:-k]
    return ''.join(stack)
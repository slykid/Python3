# 문제. 프로그래밍 대회 참가 팀을 추가모집함
# 이 때, 사전에 참가 신청한 팀은 teamIDs 에 존재하고, 추가 신청 팀ID 는 additional 에 존재함
# 사전 신청한 팀ID 로는 중복 신청불가 함
# 이 때, 추가신청한 팀만 출력하시오

# ex1. ["world","prog"], ["hello","world","code","hello","try","code"], ["hello", "code", "try"]
# ex2. ["abc","hq","xyz"], ["hq","abc","pp","xy","pp","hq"], ["pp", "xy"]

def solution(teamIDs, additional):
    answer = []

    for name in additional:
        if name not in teamIDs:
            answer.append(name)
            teamIDs.append(name)
    return answer
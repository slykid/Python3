import pandas as pd

## 팀별 자유투 확률을 계산하세요 (58.33 과 근사해야됨)
def avg_percentage(game_id, ft_number, team, result):
    """
    :param game_id: (list) The ID of the game.
    :param ft_number: (list) The number of the free throw.
    :param team: (list) Which team took the free throw.
    :param result: (list) The result of the free throw, which is either missed or made.
    :returns: (float) The mean value of the percentages (0.0-100.0) of free throws that
               each team scored in each game.
    """
    score = {}

    for i in range(len(game_id)):
        score[i] = {}
        score[i]["home_cnt"] = 0
        score[i]["home_score"] = 0
        score[i]["away_cnt"] = 0
        score[i]["away_score"] = 0

        for j in range(len(ft_number)):
            if team[j] == "home":
                score[i]["home_cnt"] += 1
                if result[j] == "made":
                    score[i]["home_score"] += 1

            elif team[j] == "away":
                score[i]["away_cnt"] += 1
                if result[j] == "made":
                    score[i]["away_score"] += 1

    home_avg = 0
    away_avg = 0
    for id in range(len(set(game_id))):
        home_avg += (score[id]["home_score"] / score[id]["home_cnt"]) * 100
        away_avg += (score[id]["away_score"] / score[id]["away_cnt"]) * 100

    return float((home_avg + away_avg) / 2)


# For example, with the parameters below, the function should return 58.33
print(avg_percentage(
    [1, 1, 1, 1, 2, 2],
    [1, 2, 3, 4, 1, 2],
    ['home', 'home', 'away', 'home', 'away', 'home'],
    ['made', 'missed', 'made', 'missed', 'missed', 'made']
))
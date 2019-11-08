import math
import time

k = int(input())
lambdas = list(map(int, input().split()))
alpha = int(input())
n = int(input())
cnts = [{} for _ in range(k)]
class_cnt = [0 for _ in range(k)]

for _ in range(n):
    line = input().split()
    c = int(line[0])
    words = set(line[2:])
    for word in words:
        cnts[c - 1].setdefault(word, 0)
        cnts[c - 1][word] += 1
    class_cnt[c - 1] += 1

prob = [{} for _ in range(k)]
for i in range(k):
    distinct_cnt = len(cnts[i])
    prob[i] = {word: ((cnt + alpha), (class_cnt[i] + distinct_cnt * alpha)) for word, cnt in cnts[i].items()}

m = int(input())

for _ in range(m):
    head, *tail = input().split()
    words = set(tail)
    scores = [None] * k
    ans = [0.0] * k
    for zz in range(k):
        if class_cnt[zz] != 0:
            scores[zz] = (math.log(lambdas[zz] * class_cnt[zz]), n)
            distinct_cnt = len(cnts[zz])
            for word in words:
                zero_prob = (alpha, (class_cnt[zz] + alpha * distinct_cnt))
                real_prob = prob[zz].get(word, zero_prob)
                a = scores[zz]
                b = tuple(map(lambda x: math.log(x), real_prob))
                scores[zz] = (a[0] + b[0], a[1] + b[1])
    cnt = 0
    sum_ln = 0
    for c in range(k):
        cur_score = scores[c]
        if cur_score is not None:
            cnt += 1
            sum_ln += cur_score[0] - cur_score[1]
    avg_ln = -sum_ln / cnt
    for c in range(k):
        cur_score = scores[c]
        if cur_score is not None:
            scores[c] = math.exp(avg_ln + cur_score[0] - cur_score[1])
    sum_scores = sum(filter(lambda x: x is not None, scores))
    for c in range(k):
        cur_score = scores[c]
        if cur_score is not None:
            ans[c] = cur_score / sum_scores
    print(*ans, sep=" ")

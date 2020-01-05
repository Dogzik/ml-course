import math


def avg(a):
    return sum(a) / len(a)


def dispersion(a):
    mean = avg(a)
    return sum([(e - mean) ** 2 for e in a])


def cov(a, b):
    mean_a = avg(a)
    mean_b = avg(b)
    return sum([(e_a - mean_a) * (e_b - mean_b) for e_a, e_b in zip(a, b)])


def close_to(a, b, eps=1e-6):
    return math.fabs(a - b) < eps


def pirson_coff(a, b):
    da = dispersion(a)
    db = dispersion(b)
    if close_to(da, 0) or close_to(db, 0):
        return 0
    else:
        return cov(a, b) / math.sqrt(da * db)


n = int(input())
x = []
y = []
for _ in range(n):
    x_i, y_i = map(int, input().split())
    x.append(x_i)
    y.append(y_i)
print(pirson_coff(x, y))

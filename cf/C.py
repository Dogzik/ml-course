from enum import Enum
import math


class DistanceType(Enum):
    MANHATTAN = 'manhattan'
    EUCLIDEAN = 'euclidean'
    CHEBYSHEV = 'chebyshev'


class KernelType(Enum):
    UNIFORM = 'uniform'
    TRIANGULAR = 'triangular'
    EPANECHNIKOV = 'epanechnikov'
    QUARTIC = 'quartic'
    TRIWEIGHT = 'triweight'
    TRICUBE = 'tricube'
    GAUSSIAN = 'gaussian'
    COSINE = 'cosine'
    LOGISTIC = 'logistic'
    SIGMOID = 'sigmoid'


class WindowType(Enum):
    FIXED = 'fixed'
    VARIABLE = 'variable'


def calc_dist(distance_type, a, b):
    if distance_type == DistanceType.MANHATTAN:
        dist = 0
        for x1, x2 in zip(a, b):
            dist += abs(x1 - x2)
        return dist
    elif distance_type == DistanceType.EUCLIDEAN:
        dist = 0
        for x1, x2 in zip(a, b):
            dist += (x1 - x2) ** 2
        return math.sqrt(dist)
    elif distance_type == DistanceType.CHEBYSHEV:
        dist = -1
        for x1, x2 in zip(a, b):
            dist = max(dist, abs(x1 - x2))
        return dist


def calc_kernel(kernel_type, cur_dist, p):
    u = cur_dist / p
    if kernel_type == KernelType.UNIFORM:
        if u < 1:
            return 1 / 2
        else:
            return 0
    elif kernel_type == KernelType.TRIANGULAR:
        if u < 1:
            return 1 - u
        else:
            return 0
    elif kernel_type == KernelType.EPANECHNIKOV:
        if u < 1:
            return (3 / 4) * (1 - (u ** 2))
        else:
            return 0
    elif kernel_type == KernelType.QUARTIC:
        if u < 1:
            return (15 / 16) * ((1 - (u ** 2)) ** 2)
        else:
            return 0
    elif kernel_type == KernelType.TRIWEIGHT:
        if u < 1:
            return (35 / 32) * ((1 - (u ** 2)) ** 3)
        else:
            return 0
    elif kernel_type == KernelType.TRICUBE:
        if u < 1:
            return (70 / 81) * ((1 - (u ** 3)) ** 3)
        else:
            return 0
    elif kernel_type == KernelType.GAUSSIAN:
        return (1 / (math.sqrt(2 * math.pi))) * math.exp((-1 / 2) * (u ** 2))
    elif kernel_type == KernelType.COSINE:
        if u < 1:
            return (math.pi / 4) * math.cos(math.pi * u / 2)
        else:
            return 0
    elif kernel_type == KernelType.LOGISTIC:
        return 1 / (math.exp(u) + 2 + math.exp(-u))
    elif kernel_type == KernelType.SIGMOID:
        return (2 / math.pi) / (math.exp(u) + math.exp(-u))


n, m = map(int, input().split())
points = []
values = []
for _ in range(n):
    line = list(map(int, input().split()))
    values.append(line.pop())
    points.append(line)
target_object = list(map(int, input().split()))
dist_type = DistanceType(input())
kernel_type = KernelType(input())
window_type = WindowType(input())
window_param = int(input())
sorted_points = sorted(
    list(
        map(
            lambda elem: (elem[0], elem[1], calc_dist(dist_type, elem[0], target_object)),
            zip(points, values)
        )
    ),
    key=lambda x: x[2]
)
if window_type == WindowType.FIXED:
    p = window_param
else:
    p = sorted_points[window_param][2]
if p == 0:
    if sorted_points[0][0] == target_object:
        neighbours = list(filter(lambda elem: elem[0] == target_object, sorted_points))
        res = sum([elem[1] for elem in neighbours]) / len(neighbours)
    else:
        res = sum([elem[1] for elem in sorted_points]) / n
else:
    x = map(lambda elem: elem[1] * calc_kernel(kernel_type, elem[2], p), sorted_points)
    y = map(lambda elem: calc_kernel(kernel_type, elem[2], p), sorted_points)
    try:
        res = sum(x) / sum(y)
    except ZeroDivisionError:
        res = sum([elem[1] for elem in sorted_points]) / n
print(res)

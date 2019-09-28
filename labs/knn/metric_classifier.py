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
            return 1
        else:
            return 0
    elif kernel_type == KernelType.TRIANGULAR:
        if u < 1:
            return 1 - u
        else:
            return 0
    elif kernel_type == KernelType.EPANECHNIKOV:
        if u < 1:
            return 1 - (u ** 2)
        else:
            return 0
    elif kernel_type == KernelType.QUARTIC:
        if u < 1:
            return (1 - (u ** 2)) ** 2
        else:
            return 0
    elif kernel_type == KernelType.TRIWEIGHT:
        if u < 1:
            return (1 - (u ** 2)) ** 3
        else:
            return 0
    elif kernel_type == KernelType.TRICUBE:
        if u < 1:
            return (1 - (u ** 3)) ** 3
        else:
            return 0
    elif kernel_type == KernelType.GAUSSIAN:
        return math.exp((-1 / 2) * (u ** 2))
    elif kernel_type == KernelType.COSINE:
        if u < 1:
            return math.cos(math.pi * u / 2)
        else:
            return 0
    elif kernel_type == KernelType.LOGISTIC:
        return 1 / (math.exp(u) + 2 + math.exp(-u))
    elif kernel_type == KernelType.SIGMOID:
        return 1 / (math.exp(u) + math.exp(-u))


def calc_class_weight(kernel_type, distance_type, window_type, window_param, points, classes, argument):
    sorted_points = sorted(
        list(
            map(
                lambda elem: (elem[0], elem[1], calc_dist(distance_type, elem[0], argument)),
                zip(points, classes)
            )
        ),
        key=lambda x: x[2]
    )
    if window_type == WindowType.FIXED:
        p = window_param
    else:
        p = sorted_points[window_param][2]
    ans = [0 for i in range(max(classes) + 1)]
    for point, point_class, dist in sorted_points:
        ans[point_class] += calc_kernel(kernel_type, dist, p)
    return ans

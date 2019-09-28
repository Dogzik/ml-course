n, m, k = map(int, input().split())
c = list(map(int, input().split()))
c_indexed = [(i, c[i]) for i in range(n)]
c_indexed = sorted(c_indexed, key=lambda e: e[1])
groups = [[] for i in range(k)]
for i in range(n):
    groups[i % k].append(c_indexed[i][0] + 1)
for group in groups:
    print(len(group), *group)



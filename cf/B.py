k = int(input())
cm = []
for i in range(k):
    cm.append(list(map(int, input().split())))


def get_class_cnt(idx):
    return sum(cm[idx])


def get_precision_and_recall(idx):
    tp = cm[idx][idx]
    tp_fp = sum([cm[j][idx] for j in range(k)])
    tp_fn = get_class_cnt(idx)
    precision = 0 if tp_fp == 0 else tp / tp_fp
    recall = 0 if tp_fn == 0 else tp / tp_fn
    return precision, recall


def calc_f_score(precision, recall):
    if precision + recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)


def get_f_score(idx):
    precision, recall = get_precision_and_recall(idx)
    return calc_f_score(precision, recall)


total_cnt = sum([sum(cm[i]) for i in range(k)])
macro_f_score = sum([get_f_score(i) * get_class_cnt(i) for i in range(k)]) / total_cnt
micro_precision = sum([get_precision_and_recall(i)[0] * get_class_cnt(i) for i in range(k)]) / total_cnt
micro_recall = sum([get_precision_and_recall(i)[1] * get_class_cnt(i) for i in range(k)]) / total_cnt
micro_f_score = calc_f_score(micro_precision, micro_recall)
print(micro_f_score)
print(macro_f_score)

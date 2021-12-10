# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2021/11/25 18:44
@Description: 
"""


def load_data(file):
    labels = []
    sentences = []
    with open(file) as f:
        for line in f.readlines():
            sample = line.strip().split('\t')
            labels.append(sample[0].split())
            sentences.append(sample[1])
    return labels, sentences


def load_label_set(file='data/types.txt'):
    labels = set()
    with open(file) as f:
        for line in f.readlines():
            labels.add(line.strip())
    return labels


def load_label_dict(file='data/types.txt'):
    label_dict = {}
    with open(file) as f:
        for i, line in enumerate(f.readlines()):
            label_dict[line.strip()] = i
    return label_dict


def macro_f1(true_and_prediction):
    # https://github.com/uwnlp/open_type/blob/master/eval_metric.py
    p, r = 0., 0.
    pred_example_count, gold_example_count = 0., 0.
    pred_label_count = 0.
    for true_labels, predicted_labels in true_and_prediction:
        if len(predicted_labels) > 0:
            pred_example_count += 1
            pred_label_count += len(predicted_labels)
            per_p = len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
            p += per_p
        if len(true_labels) > 0:
            gold_example_count += 1
            per_r = len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
            r += per_r
    precision, recall = 0, 0
    if pred_example_count > 0:
        precision = p / pred_example_count
    if gold_example_count > 0:
        recall = r / gold_example_count

    def calc_f1(p, r):
        if r == 0.:
            return 0.
        return 2 * p * r / float(p + r)

    return precision, recall, calc_f1(precision, recall)
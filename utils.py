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


def load_label_dict(file='data/ontology/types.txt'):
    label_dict = {}
    with open(file) as f:
        for i, line in enumerate(f.readlines()):
            label_dict[line.strip()] = i
    return label_dict

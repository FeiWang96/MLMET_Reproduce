# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2021/11/27 22:22
@Description: 
"""
from utils import load_data

labels, sentences = load_data('data/train_distant_300000.txt')

extend_labels = []
with open('data/distant.mlm') as f:
    for line in f.readlines():
        extend_labels.append(line.strip().split())

with open('data/train_mlm_300000.txt', 'w') as f:
    for label, extend_label, sentence in zip(labels, extend_labels, sentences):
        all_label = set(label) | set(extend_label)
        f.write(' '.join(all_label) + '\t' + sentence + '\n')

# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2021/11/24 18:25
@Description: 
"""
import os
from tqdm import tqdm
import inflect
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from utils import load_label_set, load_data, macro_f1

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
DEVICE = 'cuda:0'


def run(model_name="bert-base-cased", input_file='data/distant.prompt', output_file='data/distant.mlm',
        batch_size=64, max_seq_length=128, k=10):
    gold_labels, sentences = load_data(input_file)
    label_set = load_label_set()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()

    inflect_engine = inflect.engine()

    generated_labels = []
    skipped = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch_sentences = sentences[i:i + batch_size]
            batch_inputs = tokenizer(batch_sentences, max_length=max_seq_length, truncation=True, padding='max_length')
            batch_inputs = {key: torch.LongTensor(value).to(DEVICE) for key, value in batch_inputs.items()}

            batch_outputs = model(**batch_inputs)
            batch_logits = batch_outputs[0]

            for j, (logits, input_ids) in enumerate(zip(batch_logits, batch_inputs['input_ids'])):
                mask_index = (input_ids == 103).nonzero().reshape(-1)
                if mask_index.nelement() != 1:
                    generated_labels.append([])
                    skipped += 1
                    continue
                pred_token_ids = torch.topk(logits[mask_index[0]].reshape(-1), k)[1]  # TODO: other thresholds k
                pred_tokens = tokenizer.convert_ids_to_tokens(pred_token_ids)
                pred_labels = []
                for x in pred_tokens:
                    singular = inflect_engine.singular_noun(x)
                    if singular is not False:
                        pred_labels.append(singular)
                    else:
                        pred_labels.append(x)
                valid_labels = set(pred_labels) & label_set
                generated_labels.append(valid_labels)

    print(f'skip {skipped} sentences')

    if output_file:
        with open(output_file, 'w') as f:
            for labels in generated_labels:
                f.write(' '.join(labels) + '\n')

    true_and_prediction = []
    for gold, pred in zip(gold_labels, generated_labels):
        true_and_prediction.append((gold, pred))
    print('precision, recall, f1', macro_f1(true_and_prediction))


if __name__ == "__main__":
    run()
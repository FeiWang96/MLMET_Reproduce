# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2021/11/24 18:25
@Description: 
"""
import os
import inflect
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from utils import load_label_set, load_data

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = 'cuda:0'


def run(model_name="bert-base-cased", output_file='generated_labels.txt', batch_size=64, max_seq_length=128, k=20):
    labels, sentences = load_data('data/prompts.txt')
    label_set = load_label_set()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()

    inflect_engine = inflect.engine()

    generated_labels = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            batch_inputs = tokenizer(batch_sentences, max_length=max_seq_length, truncation=True)
            batch_inputs = {key: torch.LongTensor(value).to(DEVICE) for key, value in batch_inputs.items()}
            mask_indices = (batch_inputs['input_ids'] == 103).nonzero()
            batch_outputs = model(**batch_inputs)
            batch_logits = batch_outputs[0].cpu().numpy()

            for j, (logits, mask_index) in enumerate(zip(batch_logits, mask_indices)):
                assert j == mask_index[0]
                pred_token_ids = np.argsort(-logits[mask_index[1]].reshape(-1))[:k]  # TODO: other thresholds
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

    with open(output_file, 'w') as f:
        for labels in generated_labels:
            f.write(' '.join(labels) + '\n')


if __name__ == "__main__":
    run()

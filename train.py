# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2021/11/24 18:25
@Description: 
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments

from model import BertForUFET
from utils import load_label_dict, load_data, macro_f1

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


class UFDataset(Dataset):
    def __init__(self, labels, sentences, label_dict, tokenizer):
        self.labels = labels
        self.label_dict = label_dict
        self.inputs = self._process_input(sentences, tokenizer)

    def _process_label(self, label_list):
        label_flags = np.zeros(len(self.label_dict))
        for w in label_list:
            if w in self.label_dict:
                i = self.label_dict[w]
                label_flags[i] = 1
        return label_flags

    @staticmethod
    def _process_input(sentences, tokenizer):
        return tokenizer(sentences, max_length=128, truncation=True, padding=False)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {
            'labels': torch.FloatTensor(self._process_label(self.labels[idx])),
            'input_ids': torch.LongTensor(self.inputs['input_ids'][idx]),
            'attention_mask': torch.LongTensor(self.inputs['attention_mask'][idx]),
            'token_type_ids': torch.LongTensor(self.inputs['token_type_ids'][idx])
        }
        return sample


def run(model_name="bert-base-cased"):
    label_dict = load_label_dict()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    train_labels, train_sentences = load_data('data/train_mlm_300000.txt')
    train_dataset = UFDataset(train_labels, train_sentences, label_dict, tokenizer)
    dev_labels, dev_sentences = load_data('data/dev.txt')
    val_dataset = UFDataset(dev_labels, dev_sentences, label_dict, tokenizer)
    test_labels, test_sentences = load_data('data/test.txt')
    test_dataset = UFDataset(test_labels, test_sentences, label_dict, tokenizer)

    config = AutoConfig.from_pretrained(model_name, num_labels=len(label_dict))
    model = BertForUFET.from_pretrained(model_name, config=config)

    training_args = TrainingArguments(
        per_device_train_batch_size=48,
        per_device_eval_batch_size=256,
        learning_rate=2e-5,
        lr_scheduler_type='linear',
        warmup_steps=5000,  # 4000
        weight_decay=0.01,
        max_steps=100000,  # 40000
        logging_steps=1000,
        save_steps=1000,
        evaluation_strategy='steps',
        group_by_length=True,
        metric_for_best_model='f1',
        load_best_model_at_end=True,
        save_total_limit=1,
        output_dir='./results/mlm',  # output directory
    )

    def compute_metrics(p):
        gold_and_pred = []
        for logits, gold in zip(p.predictions, p.label_ids.astype(int)):
            pred = np.squeeze(np.argwhere(logits > 0), axis=1)  # TODO: threshold
            if len(pred) == 0:
                pred = [np.argmax(logits)]
            gold = np.squeeze(np.argwhere(gold > 0.5), axis=1)
            gold_and_pred.append((gold, pred))
        p, r, f1 = macro_f1(gold_and_pred)
        return {'f1': f1, 'precision': p, 'recall': r}

    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        tokenizer=tokenizer,
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        compute_metrics=compute_metrics
    )

    train_result = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    trainer.save_metrics("test", test_metrics)
    print(test_metrics)


if __name__ == "__main__":
    run()
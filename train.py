# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2021/11/24 18:25
@Description: 
"""
import os
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.models.bert.modeling_bert import BertForSequenceClassification

from utils import load_label_dict, load_data

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class UFDataset(Dataset):
    def __init__(self, labels, sentences, label_dict, tokenizer):
        self.labels = self._process_label(labels, label_dict)
        self.inputs = self._process_input(sentences, tokenizer)

    @staticmethod
    def _process_label(labels, label_dict):
        label_flags = np.zeros((len(labels), len(label_dict)))
        for i, label_list in enumerate(labels):
            for w in label_list:
                j = label_dict[w]
                label_flags[i][j] = 1
        return label_flags

    @staticmethod
    def _process_input(sentences, tokenizer):
        return tokenizer(sentences, max_length=128, truncation=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {
            'labels': torch.FloatTensor(self.labels[idx]),
            'input_ids': torch.LongTensor(self.inputs['input_ids'][idx]),
            'attention_mask': torch.LongTensor(self.inputs['attention_mask'][idx]),
            'token_type_ids': torch.LongTensor(self.inputs['token_type_ids'][idx])
        }
        return sample


def run(model_name="bert-base-uncased"):
    label_dict = load_label_dict()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    labels, sentences = load_data('data/train.txt')
    train_dataset = UFDataset(labels, sentences, label_dict, tokenizer)
    val_dataset = UFDataset(labels, sentences, label_dict, tokenizer)
    test_dataset = UFDataset(labels, sentences, label_dict, tokenizer)

    config = AutoConfig.from_pretrained(model_name, num_labels=len(label_dict),
                                        problem_type='multi_label_classification')
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
    )

    def compute_metrics(p):
        pred_labels = (p.predictions > 0).astype(int)  # TODO: threshold
        gold_labels = p.label_ids.astype(int)
        n_labels = pred_labels.shape[1]
        macro_f1 = 0
        for i in range(n_labels):
            macro_f1 += f1_score(gold_labels[:, i], pred_labels[:, i])
        macro_f1 /= n_labels
        return {'macro_f1': macro_f1}

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

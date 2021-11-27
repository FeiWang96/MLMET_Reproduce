# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2021/11/27 13:47
@Description: 
"""
import torch
from torch.nn import BCEWithLogitsLoss
from transformers.models.bert.modeling_bert import BertForSequenceClassification


class BertForUFET(BertForSequenceClassification):
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # TODO: weighted loss
            loss_fct = BCEWithLogitsLoss(reduce=False)
            general_mask = (torch.sum(labels[:, :9], dim=1) > 0).reshape(-1, 1)
            general_loss = loss_fct(logits[:, :9], labels[:, :9]) * general_mask
            fine_mask = (torch.sum(labels[:, 9:130], dim=1) > 0).reshape(-1, 1)
            fine_loss = loss_fct(logits[:, 9:130], labels[:, 9:130]) * fine_mask
            ultrafine_mask = (torch.sum(labels[:, 130:], dim=1) > 0).reshape(-1, 1)
            ultrafine_loss = loss_fct(logits[:, 130:], labels[:, 130:]) * ultrafine_mask
            loss = torch.mean(general_loss) + torch.mean(fine_loss) + torch.mean(ultrafine_loss)

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

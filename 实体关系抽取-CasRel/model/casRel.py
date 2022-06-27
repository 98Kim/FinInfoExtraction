import torch.nn as nn
import torch
from transformers import BertModel


class CasRel(nn.Module):
    def __init__(self, config):
        super(CasRel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_name)
        self.sub_heads_linear = nn.Linear(self.config.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(self.config.bert_dim, 1)
        self.obj_heads_linear = nn.Linear(self.config.bert_dim, self.config.num_relations)
        self.obj_tails_linear = nn.Linear(self.config.bert_dim, self.config.num_relations)

    def get_encoded_text(self, token_ids, mask):
        encoded_text = self.bert(token_ids, attention_mask=mask)[0]  # shape是(batch_size, sentence_lengh, 768)
        return encoded_text

    def get_subs(self, encoded_text):
        """给定句子，预测主体"""
        pred_sub_heads = torch.sigmoid(self.sub_heads_linear(encoded_text))
        pred_sub_tails = torch.sigmoid(self.sub_tails_linear(encoded_text))
        return pred_sub_heads, pred_sub_tails

    def get_objs_for_specific_sub(self, sub_head_mapping, sub_tail_mapping, encoded_text):
        """
            对给定的主体，得到他的客体
            - 主体的head和tail取平均保证维度，然后加到text中
        """
        # sub_head_mapping [batch, 1, seq] * encoded_text [batch, seq, dim]
        sub_head = torch.matmul(sub_head_mapping, encoded_text)
        sub_tail = torch.matmul(sub_tail_mapping, encoded_text)
        sub = (sub_head + sub_tail) / 2
        encoded_text = encoded_text + sub
        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))
        pred_obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))
        return pred_obj_heads, pred_obj_tails

    def forward(self, token_ids, mask, sub_head, sub_tail):
        """为什么用sub_head而不是sub_heads?这是随机抽取的一个主体"""
        encoded_text = self.get_encoded_text(token_ids, mask)  # 将每个词从id转换成bert向量
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text)
        sub_head_mapping = sub_head.unsqueeze(1)  # 原本是[batch_size, max_sentence_length]转变为[batch_size, 1, max_sentence_length]增加了一个维度
        sub_tail_mapping = sub_tail.unsqueeze(1)
        pred_obj_heads, pre_obj_tails = self.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, encoded_text)

        return {
            "sub_heads": pred_sub_heads,  # 预测的主体开头位置
            "sub_tails": pred_sub_tails,
            "obj_heads": pred_obj_heads,
            "obj_tails": pre_obj_tails,
        }
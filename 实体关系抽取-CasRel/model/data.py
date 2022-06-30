import json
from random import choice
from fastNLP import TorchLoaderIter, DataSet, Vocabulary, Sampler
from fastNLP.io import JsonLoader
import torch
import numpy as np
from transformers import BertTokenizer
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_rel = 18


def load_data(train_path, dev_path, test_path, rel_dict_path):
    paths = {'train': train_path, 'dev': dev_path, 'test': test_path}
    loader = JsonLoader({"text": "text", "spo_list": "spo_list"})
    data_bundle = loader.load(paths)
    id2rel = json.load(open(rel_dict_path, encoding='utf-8'))
    rel_vocab = Vocabulary(unknown=None, padding=None)
    rel_vocab.add_word_lst(list(id2rel.values()))  # rel_vocab是关系的数量
    return data_bundle, rel_vocab


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


class MyDataset(DataSet):
    def __init__(self, config, dataset, rel_vocab, is_test):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.rel_vocab = rel_vocab
        self.is_test = is_test
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_name)

    def __getitem__(self, item): # 对应每一个句子
        json_data = self.dataset[item]
        text = json_data['text']
        tokenized = self.tokenizer(text, max_length=self.config.max_len, truncation=True)  # 将文本转换成id
        tokens = tokenized['input_ids'] # 句子的token
        masks = tokenized['attention_mask'] # 是否是padding上来的
        text_len = len(tokens)

        token_ids = torch.tensor(tokens, dtype=torch.long) # 句子的token id
        masks = torch.tensor(masks, dtype=torch.bool) # 
        """主体和客体起始位置的记录"""
        sub_heads, sub_tails = torch.zeros(text_len), torch.zeros(text_len)
        sub_head, sub_tail = torch.zeros(text_len), torch.zeros(text_len)
        obj_heads = torch.zeros((text_len, self.config.num_relations)) # shape是文本长度x关系数量，每个关系都有
        obj_tails = torch.zeros((text_len, self.config.num_relations))

        if not self.is_test: # train
            s2ro_map = defaultdict(list)  # 创建一个dictionary，将键-值对更新为键-列表对，每个键可以调用list的属性
            for spo in json_data['spo_list']:
                triple = (self.tokenizer(spo['subject'], add_special_tokens=False)['input_ids'], 
                          self.rel_vocab.to_index(spo['predicate']),
                          self.tokenizer(spo['object'], add_special_tokens=False)['input_ids']) # 把文本转换成id然后记录三元组,同时避免加入[CLS][SEP]这些特殊符号
                """
                - ISSUE: 如果某一个词语多次出现则只能找到第一个位置，这里有问题
                - SOLUTION: 再加一个变量记录这个词语是否出现过，如果出现过就记录他的位置，然后从这个位置开始往后找
                - WHY: 其实不改也可以，因为一个主体在文本中的意思应该是一样的。但是因为BERT模型会考虑前后文信息所以最好还是改一下？不确定，看验证结果，不是很严重的问题。
                """
                sub_head_idx = find_head_idx(tokens, triple[0])
                obj_head_idx = find_head_idx(tokens, triple[2])
                """可以试一下assert判断+终止"""
                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1) # 主体位置
                    s2ro_map[sub].append(
                        (obj_head_idx, obj_head_idx + len(triple[2]) - 1, triple[1]))  # 用append解决一个主体对应多个客体的问题。客体位置+关系

            if s2ro_map:  # 可能没有记录
                for s in s2ro_map:  # 遍历主题的位置信息
                    sub_heads[s[0]] = 1  # 主体的头
                    sub_tails[s[1]] = 1
                sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys())) # 选择随机一个主体
                """
                这里每个句子随机选取一个主体然后用sub_head记录不知道是什么操作
                """
                sub_head[sub_head_idx] = 1
                sub_tail[sub_tail_idx] = 1
                for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                    obj_heads[ro[0]][ro[2]] = 1
                    obj_tails[ro[1]][ro[2]] = 1

        return token_ids, masks, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, json_data['spo_list']

    def __len__(self):
        return len(self.dataset)


def my_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch)) # 返回非空的batch
    token_ids, masks, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triples = zip(*batch)
    batch_token_ids = pad_sequence(token_ids, batch_first=True)
    batch_masks = pad_sequence(masks, batch_first=True)
    batch_sub_heads = pad_sequence(sub_heads, batch_first=True)
    batch_sub_tails = pad_sequence(sub_tails, batch_first=True)
    batch_sub_head = pad_sequence(sub_head, batch_first=True)
    batch_sub_tail = pad_sequence(sub_tail, batch_first=True)
    batch_obj_heads = pad_sequence(obj_heads, batch_first=True)
    batch_obj_tails = pad_sequence(obj_tails, batch_first=True)
    """
    return中第一个元素是用来训练的，放到forward去用
    第二个元素是正确的，用来计算loss
    """
    return {"token_ids": batch_token_ids.to(device),
            "mask": batch_masks.to(device),
            "sub_head": batch_sub_head.to(device), # 随机选取一个主体放到预测函数里，用来训练主体->客体映射
            "sub_tail": batch_sub_tail.to(device),
            "sub_heads": batch_sub_heads.to(device),
            }, \
           {"mask": batch_masks.to(device),
            "sub_heads": batch_sub_heads.to(device), # 真实的主体开头位置
            "sub_tails": batch_sub_tails.to(device),
            "obj_heads": batch_obj_heads.to(device),
            "obj_tails": batch_obj_tails.to(device),
            "triples": triples
            }


class MyRandomSampler(Sampler):
    def __call__(self, data_set):
        return np.random.permutation(len(data_set)).tolist()


def get_data_iterator(config, dataset, rel_vocab, is_test=False, collate_fn=my_collate_fn):
    dataset = MyDataset(config, dataset, rel_vocab, is_test)
    return TorchLoaderIter(dataset=dataset,
                           collate_fn=collate_fn,
                           batch_size=config.batch_size if not is_test else 1,
                           sampler=MyRandomSampler())




import json
import logging
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

logger = logging.getLogger()


CONLL03 = {0: 'person', 1: 'organisation', 2: 'location', 3: 'misc'}

re3d = {0: 'Person', 1: 'Organisation', 2: 'Location', 4: 'DocumentReference',
        5: 'MilitaryPlatform', 6: 'Money', 7: 'Country', 8: 'Quantity',
        9: 'Temporal', 10: 'Weapon'}

AI = {0: 'person', 1: 'organisation', 2: 'location', 3: 'misc',
      4: 'country', 5: "field", 6: "task", 7: "product", 8: "algorithm",
      9: "researcher", 10: "metrics", 11: "programlang", 12: "conference",
      13: "university"}

FEW_AI = {7: "product", 9: "researcher", 13: "university"}

LITERATURE = {0: 'person', 1: 'organisation', 2: 'location', 3: 'misc',
              4: 'country', 5: 'event', 6: "writer", 7: "book", 8: "award",
              9: "poem", 10: "magazine", 11: "literarygenre"}

FEW_LITERATURE = {6: "writer", 7: "book", 8: 'award'}

MUSIC = {0: 'person', 1: 'organisation', 2: 'location', 3: 'misc',
         4: 'country', 5: 'event', 6: 'musicalartist', 7: 'musicgenre',
         8: 'song', 9: 'band', 10: 'album', 11: 'musicalinstrument',
         12: 'award'}

FEW_MUSIC = {6: 'musicalartist', 8: 'song', 12: 'award'}

POLITICS = {0: 'person', 1: 'organisation', 2: 'location', 3: 'misc',
            4: 'country', 5: 'event', 6: 'politician', 7: 'election',
            8: 'politicalparty'}

FEW_POLITICS = {6: 'politician', 7: 'election', 8: 'politicalparty'}

SCIENCE = {0: 'person', 1: 'organisation', 2: 'location', 3: 'misc',
           4: 'country', 5: 'event', 6: 'scientist', 7: 'university',
           8: 'discipline', 9: 'enzyme', 10: 'protein', 11: 'chemicalelement',
           12: 'chemicalcompound', 13: 'astronomicalobject', 14: 'academicjournal',
           15: 'theory', 16: 'award'}

FEW_SCIENCE = {6: 'scientist', 7: 'university', 12: 'chemicalcompound', 13: 'astronomicalobject', 16: 'award'}

domain_entity_types_reverse = {
    'ai': AI,
    'literature': LITERATURE,
    'music': MUSIC,
    'politics': POLITICS,
    'science': SCIENCE,
    're3d': re3d,
    'conll03': CONLL03,
    'few_nerd_politics': FEW_POLITICS,
    'few_nerd_science': FEW_SCIENCE,
    'few_nerd_music': FEW_MUSIC,
    'few_nerd_literature': FEW_LITERATURE,
    'few_nerd_ai': FEW_AI,
}

domain_entity_types = {
    'ai': dict(zip(AI.values(), AI.keys())),
    'literature': dict(zip(LITERATURE.values(), LITERATURE.keys())),
    'music': dict(zip(MUSIC.values(), MUSIC.keys())),
    'politics': dict(zip(POLITICS.values(), POLITICS.keys())),
    'science': dict(zip(SCIENCE.values(), SCIENCE.keys())),
    're3d': dict(zip(re3d.values(), re3d.keys())),
    'conll03': dict(zip(CONLL03.values(), CONLL03.keys())),
    'few_nerd_politics': dict(zip(FEW_POLITICS.values(), FEW_POLITICS.keys())),
    'few_nerd_science': dict(zip(FEW_SCIENCE.values(), FEW_SCIENCE.keys())),
    'few_nerd_music': dict(zip(FEW_MUSIC.values(), FEW_MUSIC.keys())),
    'few_nerd_literature':  dict(zip(FEW_LITERATURE.values(), FEW_LITERATURE.keys())),
    'few_nerd_ai': dict(zip(FEW_AI.values(), FEW_AI.keys())),
}


def read_query(file_path, tokenizer: BertTokenizer):
    """
    read mrc query, the max length of query is 52
    :param file_path: mrc query place
    :param tokenizer: bert tokenizer
    :return:
    """
    with open(file_path, encoding='utf-8') as f:
        dat = json.load(f)
        for key, value in dat.items():
            dat[key] = tokenizer.tokenize(value)
    return dat


def read_bio(file_path, tokenizer: BertTokenizer):
    """
    read bio and split, the max length of data is 113 from conll03 English
    :param file_path: bio file
    :param tokenizer: bert tokenizer
    :return:
    """
    tokens = []
    labels = []
    with open(file_path, encoding='utf-8') as fr:
        token_list, label_list = [], []
        for i, line in enumerate(fr):
            line = line.strip()
            if line == "":
                # get a sentence
                tokens.append(token_list)
                labels.append(label_list)
                token_list, label_list = [], []
                continue

            splits = line.split("\t")
            if len(splits) == 1:
                splits = line.split(" ")
            token = splits[0]
            label = splits[-1]

            subs_ = tokenizer.tokenize(token)  # handle token，split unknow token to several know token

            if len(subs_) > 0:  # Prevent processing empty strings
                split_label = label.split('-')
                if split_label[0] == 'B':
                    label_list.extend([label] + ['I-' + split_label[1]] * (len(subs_) - 1))
                else:
                    label_list.extend(
                        [label] * (len(subs_))  # get label ID from target domain
                    )

                token_list.extend(subs_)

    return tokens, labels


class Example(object):
    def __init__(self, token, label, entitys, domain=None):
        self.token = token
        self.label = label
        self.entitys = entitys
        self.domain = domain


def get_examples(opt, domain, mode, tokenizer):
    """
    get examples
    """
    tokens, labels = read_bio(
        os.path.join(opt.domain_dir, domain+'/BIO/' + mode + '.txt'),
        tokenizer)

    examples = [] 
    for token_list, label_list in zip(tokens, labels):
        assert len(token_list) == len(label_list)
        entitys = []
        start_id = None
        entity_len = 0
        last_type = None
        for i in range(len(label_list)):
            if label_list[i].startswith('B'):
                start_id = i
                entity_len = 1
                last_type = label_list[i].split('-')[1]
            if label_list[i].startswith('I'):
                entity_len += 1

            if label_list[i] == 'O' and last_type:
                entitys.append(
                    {
                        'entity_type': last_type,
                        'start_id': start_id,
                        'entity_len': entity_len,
                        'entity': token_list[start_id: start_id+entity_len: 1]
                    }
                )

                entity_len = 0
                start_id = None
                last_type = None
        examples.append(
            Example(
                token=token_list,
                label=label_list,
                entitys=entitys,
                domain=domain
            )
        )
    return examples


class BaseFeature:
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids


class MRCFeature(BaseFeature):
    def __init__(self,
                 raw_token,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 contain_entity,
                 ent_type,
                 start_ids=None,
                 end_ids=None
                 ):
        super(MRCFeature, self).__init__(token_ids=token_ids,
                                         attention_masks=attention_masks,
                                         token_type_ids=token_type_ids)
        self.ent_type = ent_type
        self.start_ids = start_ids
        self.end_ids = end_ids
        self.raw_token = raw_token
        self.contain_entity = contain_entity


def convert_mrc_example(example: Example, tokenizer: BertTokenizer, max_seq_len, ent2query):
    tokens_b = example.token
    entities = example.entitys
    domain = example.domain
    features = []
    if len(tokens_b) == 0 or tokens_b is None:
        return None
    entity_dict = defaultdict(list)

    for ent in entities:
        entity_dict[ent['entity_type']].append(
            (ent['start_id'], ent['start_id'] + ent['entity_len'] - 1, ent['entity'])
        )  # 每一类实体分别存储，用（开始，结束，实体）表示一个实体

    # 每一类为一个 example，每一个样本都分解出 num of entity 个样本
    for _type in domain_entity_types[domain]:
        # 构建 start_ids 与 end_ids 的模板
        start_ids = [0] * len(tokens_b)
        end_ids = [0] * len(tokens_b)

        # 获取query
        tokens_a = ent2query[_type]

        # 查看是否有该类型数据，要是有则把对应位置标 1
        for _entity in entity_dict[_type]:
            start_ids[_entity[0]] = 1
            end_ids[_entity[1]] = 1

        contain_entity = 0
        if sum(start_ids) >= 1:
            contain_entity = 1

        if len(start_ids) > max_seq_len - len(tokens_a) - 3:
            start_ids = start_ids[:max_seq_len - len(tokens_a) - 3]
            end_ids = end_ids[:max_seq_len - len(tokens_a) - 3]
            print('产生了不该有的截断')
            continue
        raw_token = ['CLS'] + tokens_a + ['SEP'] + tokens_b + ['SEP']
        start_ids = [0] + [0] * len(tokens_a) + [0] + start_ids + [0]
        end_ids = [0] + [0] * len(tokens_a) + [0] + end_ids + [0]

        # pad
        if len(start_ids) < max_seq_len:
            pad_length = max_seq_len - len(start_ids)
            raw_token = raw_token + ['PAD'] * pad_length
            start_ids = start_ids + [0] * pad_length  # CLS SEP PAD label都为O
            end_ids = end_ids + [0] * pad_length

        assert len(start_ids) == max_seq_len
        assert len(end_ids) == max_seq_len

        encode_dict = tokenizer.encode_plus(text=tokens_a,
                                            text_pair=tokens_b,
                                            max_length=max_seq_len,
                                            pad_to_max_length=True,
                                            truncation_strategy='only_second',
                                            is_pretokenized=True,
                                            return_token_type_ids=True,
                                            return_attention_mask=True)

        token_ids = encode_dict['input_ids']
        attention_masks = encode_dict['attention_mask']
        token_type_ids = encode_dict['token_type_ids']

        feature = MRCFeature(
            raw_token=raw_token,
            token_ids=token_ids,
            attention_masks=attention_masks,
            token_type_ids=token_type_ids,
            ent_type=domain_entity_types[domain][_type],
            start_ids=start_ids,
            end_ids=end_ids,
            contain_entity=contain_entity
        )

        features.append(feature)

    return features


def convert_examples_to_features(examples, domain, max_seq_len, tokenizer: BertTokenizer, ent2query):
    logger.info(f'Convert {domain} domain`s {len(examples)} examples to features')
    features = []

    for i, example in enumerate(examples):
        feature = convert_mrc_example(
            example=example,
            max_seq_len=max_seq_len,
            ent2query=ent2query,
            tokenizer=tokenizer
        )

        if feature is not None:
            features.extend(feature)
    return features


class NERDataset(Dataset):
    def __init__(self, features):

        self.nums = len(features)
        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks).float() for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]
        self.start_ids = [torch.tensor(example.start_ids).long() for example in features]
        self.end_ids = [torch.tensor(example.end_ids).long() for example in features]
        self.raw_tokens = [example.raw_token for example in features]
        self.ent_type = [torch.tensor(example.ent_type).long() for example in features]
        self.contain_entity = [torch.tensor(example.contain_entity).long() for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {
            'token_ids': self.token_ids[index],
            'attention_masks': self.attention_masks[index],
            'token_type_ids': self.token_type_ids[index],
            'start_ids': self.start_ids[index],
            'end_ids': self.end_ids[index],
            'raw_tokens': self.raw_tokens[index],
            'ent_type': self.ent_type[index],
            'contain_entity': self.contain_entity[index]
        }
        return data


def get_dataset(opt, domain, mode, tokenizer: BertTokenizer, ent2query,
                up_sample=False, other_features=False, get_features=False):

    logger.info(f'get {domain} domain`s {mode} dataset')

    examples = get_examples(opt, domain, mode, tokenizer)

    logger.info(f'get {domain} domain`s {mode} mode`s {len(examples)} examples')

    features = convert_examples_to_features(examples, domain, opt.max_seq_len, tokenizer, ent2query)
    logger.info(f'get {domain} domain`s {len(features)} features')
    if get_features:
        return features
    if other_features:
        features = features + other_features
    length = len(features)

    if up_sample:
        features = features * (int(up_sample / (2*length)))
    logger.info(f'get {domain} domain`s {len(features)} features')

    dataset = NERDataset(features)

    return dataset, length





















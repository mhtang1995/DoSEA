import os
import torch
import torch.nn as nn
from transformers import BertModel
import math
import torch.nn.functional as F
import numpy as np
import dgl
from dgl.nn.pytorch import GraphConv


class BaseModel(nn.Module):
    """
    适合所有以 bert 输出作为输入的模型的骨架
    """

    def __init__(self,
                 bert_dir,
                 dropout_prob
                 ):
        super(BaseModel, self).__init__()
        config_path = os.path.join(bert_dir, 'config.json')

        assert os.path.exists(bert_dir) and os.path.exists(config_path), \
            'pretrained bert file does not exist'

        self.bert_module = BertModel.from_pretrained(bert_dir,
                                                     output_hidden_states=True,
                                                     hidden_dropout_prob=dropout_prob)

        self.bert_config = self.bert_module.config

    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)


class ConditionalLayerNorm(nn.Module):
    def __init__(self,
                 normalized_shape,
                 cond_shape,
                 eps=1e-12):
        super().__init__()

        self.eps = eps

        self.weight_dense = nn.Sequential(
            nn.Linear(cond_shape, normalized_shape),
            nn.Tanh(),
            nn.Dropout(0.1)
        )

    def forward(self, inputs, cond=None):
        assert cond is not None, 'Conditional tensor need to input when use conditional layer norm'
        cond = torch.unsqueeze(cond, 1)  # (b, 1, h*2)
        weight = self.weight_dense(cond)

        mean = torch.mean(inputs, dim=-1, keepdim=True)  # （b, s, 1）
        outputs = inputs - mean  # (b, s, h)
        variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)  # (b, s, 1)
        outputs = outputs / std  # (b, s, h)

        va = (outputs - weight) ** 2
        outputs = torch.sqrt(va + self.eps)
        return outputs


# class MRCModel(BaseModel):
#     def __init__(self,
#                  bert_dir,
#                  entity_types,
#                  type_embedding_dim=128,
#                  mid_linear_dims=128,
#                  max_len=200,
#                  dropout_prob=0.1):
#
#         super(MRCModel, self).__init__(bert_dir, dropout_prob=dropout_prob)
#
#         self.type_embedding_dim = type_embedding_dim
#         self.dropout = nn.Dropout(dropout_prob)  # dropout
#         self.max_len = max_len
#         # self.birnn = nn.LSTM(768, 384, batch_first=True, bidirectional=True,  num_layers=2, dropout=dropout_prob)
#
#         # class_mid linear
#         self.class_mid_linear = nn.Sequential(
#             nn.Linear(self.bert_config.hidden_size, mid_linear_dims),
#             nn.ReLU(),
#             self.dropout
#         )
#
#         # entity_mid linear
#         self.entity_mid_linear = nn.Sequential(
#             nn.Linear(self.bert_config.hidden_size, mid_linear_dims),
#             nn.ReLU(),
#             self.dropout
#         )
#
#         # type_embedding
#         self.type_embedding = nn.Embedding(entity_types, type_embedding_dim)
#
#         # classification
#         self.classification_linear = nn.Sequential(
#             nn.Linear(mid_linear_dims, type_embedding_dim),
#             nn.ReLU(),
#             self.dropout
#         )
#         self.contain_entity_fc = nn.Linear(type_embedding_dim, 2)
#
#         # entity extract
#         self.conditional_layer_norm = ConditionalLayerNorm(
#             mid_linear_dims,
#             type_embedding_dim,
#             eps=self.bert_config.layer_norm_eps
#         )
#
#         self.start_fc = nn.Linear(mid_linear_dims, 2)
#         self.end_fc = nn.Linear(mid_linear_dims, 2)
#
#         # 损失函数
#         # self.criterion = nn.CrossEntropyLoss()
#         self.criterion = nn.CrossEntropyLoss(reduction='none')
#
#         init_blocks = [
#             self.class_mid_linear,
#             self.entity_mid_linear,
#             self.type_embedding,
#             self.conditional_layer_norm,
#             self.classification_linear,
#             self.contain_entity_fc,
#             self.start_fc,
#             self.end_fc]
#         self._init_weights(init_blocks)
#
#     def forward(self, token_ids, attention_masks, token_type_ids, ent_type,
#                 contain_entity=None, start_ids=None, end_ids=None, cla=None, pseudo=None, rate=None):
#
#         bert_outputs = self.bert_module(
#             input_ids=token_ids,
#             attention_mask=attention_masks,
#             token_type_ids=token_type_ids
#         )
#         # bert 输出
#         seq_out = bert_outputs[0]
#
#         # seq_out, _ = self.birnn(seq_out)
#
#         # type embedding
#         ty = self.type_embedding(ent_type)
#
#         # classification
#         cla_out = self.class_mid_linear(seq_out)
#         cla_out = self.classification_linear(cla_out)
#         cla_out, _ = self.attention(
#             query=ty,
#             key=cla_out,
#             value=cla_out,
#             dropout=self.dropout,
#             mask=token_type_ids
#         )
#         contain_entity_logits = self.contain_entity_fc(cla_out)
#
#         # ner
#         ent_out = self.entity_mid_linear(seq_out)
#         ent_out = self.conditional_layer_norm(ent_out, ty)
#         start_logits = self.start_fc(ent_out)
#         end_logits = self.end_fc(ent_out)
#         if contain_entity is None and start_ids is None and end_ids is None:
#             # predicate
#             contain_entity_logits = torch.argmax(contain_entity_logits, dim=-1)
#             start_logits = torch.argmax(start_logits, dim=-1)
#             end_logits = torch.argmax(end_logits, dim=-1)
#             return start_logits, end_logits, contain_entity_logits
#         else:
#             # 辅助任务损失函数
#             ce_loss = self.criterion(contain_entity_logits, contain_entity)
#
#             # 对MRC任务进行处理
#             start_logits = start_logits.view(-1, 2)
#             end_logits = end_logits.view(-1, 2)
#             # 去掉 text_a 和 padding 部分的标签，计算真实 loss
#             active_loss = token_type_ids.view(-1) == 1
#             active_start_logits = start_logits[active_loss]
#             active_end_logits = end_logits[active_loss]
#
#             active_start_labels = start_ids.view(-1)[active_loss]
#             active_end_labels = end_ids.view(-1)[active_loss]
#
#             start_loss = self.criterion(active_start_logits, active_start_labels.view(-1))
#             end_loss = self.criterion(active_end_logits, active_end_labels.view(-1))
#
#             if cla is not None:
#                 return 0.1 * ce_loss.mean() + start_loss.mean() + end_loss.mean()
#             else:
#                 # 源领域训练策略
#                 # (batch,)
#                 # nums of pseudo data
#                 pseudo = torch.cat((pseudo.unsqueeze(-1),) * self.max_len, dim=-1).view(-1)
#                 active_pseudo = pseudo[active_loss]  # 1 代表有问题的，0代表没问题的
#
#                 start_loss = (rate * active_pseudo * start_loss).sum() + ((1 - active_pseudo) * start_loss).sum()
#                 end_loss = (rate * active_pseudo * end_loss).sum() + ((1 - active_pseudo) * end_loss).sum()
#
#                 return start_loss.mean() + end_loss.mean()
#
#     def attention(self, query, key, value, mask=None, dropout=None):
#         """
#         Compute 'Scaled Dot Product Attention'
#         :param query:
#         :param key:
#         :param value:
#         :param mask: 掩码张量
#         :param dropout: nn.Dropout 实例对象
#         :return:
#         """
#         """
#         自注意力机制中 query = key = value
#         假设 query 为 batch_size x seq_len x d_model 的 tensor
#         """
#         # 取 query 最后一维的大小， 即 512
#         d_k = query.size(-1)
#         # 计算分数：先将 Key 的最后两个维度转置，再除缩放系数
#         scores = torch.matmul(query.unsqueeze(1), key.transpose(-2, -1)) / math.sqrt(d_k)
#         mask = mask.unsqueeze(1)
#         if mask is not None:
#             # 将一些信息掩盖，即将注意力不集中在这里
#             # 将掩码张量和 scores 张量上每个位置一一比较，如果掩码张量等于0，则对应位置就用 -1e9 这个值填充
#             scores = scores.masked_fill(mask == 0, -1e9)
#         # 对 scores 最后一维做 softmax
#         p_attn = F.softmax(scores, dim=-1)
#         # 对softmax 结果做 dropout
#         if dropout is not None:
#             p_attn = dropout(p_attn)
#         # (48, 1, 200) matmul  (48, 200, 8)
#         out = torch.matmul(p_attn, value).squeeze(1)
#         # out = F.relu(out)
#         return out, p_attn


class MRCModel(BaseModel):
    def __init__(self,
                 bert_dir,
                 entity_types,
                 type_embedding_dim=128,
                 mid_linear_dims=128,
                 max_len=200,
                 dropout_prob=0.1):

        super(MRCModel, self).__init__(bert_dir, dropout_prob=dropout_prob)

        self.type_embedding_dim = type_embedding_dim
        self.dropout = nn.Dropout(dropout_prob)  # dropout
        self.max_len = max_len
        self.birnn = nn.LSTM(768, 384, batch_first=True, bidirectional=True, num_layers=2, dropout=dropout_prob)

        # class_mid linear
        self.class_mid_linear = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size, mid_linear_dims),
            nn.ReLU(),
            self.dropout
        )

        # entity_mid linear
        self.entity_mid_linear = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size, mid_linear_dims),
            nn.ReLU(),
            self.dropout
        )

        # type_embedding
        self.type_embedding = nn.Embedding(entity_types, type_embedding_dim)

        # classification
        self.classification_linear = nn.Sequential(
            nn.Linear(mid_linear_dims, type_embedding_dim),
            nn.ReLU(),
            self.dropout
        )
        self.contain_entity_fc = nn.Linear(type_embedding_dim, 1)

        # entity extract
        self.conditional_layer_norm = ConditionalLayerNorm(
            mid_linear_dims,
            type_embedding_dim,
            eps=self.bert_config.layer_norm_eps
        )

        self.start_fc = nn.Linear(mid_linear_dims, 1)
        self.end_fc = nn.Linear(mid_linear_dims, 1)

        # 损失函数
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

        init_blocks = [
            self.class_mid_linear,
            self.entity_mid_linear,
            self.type_embedding,
            self.conditional_layer_norm,
            self.classification_linear,
            self.contain_entity_fc,
            self.start_fc,
            self.end_fc]
        self._init_weights(init_blocks)

    def forward(self, token_ids, attention_masks, token_type_ids, ent_type,
                contain_entity=None, start_ids=None, end_ids=None, cla=None, task_rate=None, rate=None):

        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )
        # bert 输出
        seq_out = bert_outputs[0]

        seq_out, _ = self.birnn(seq_out)

        # type embedding
        ty = self.type_embedding(ent_type)

        # classification
        cla_out = self.class_mid_linear(seq_out)
        cla_out = self.classification_linear(cla_out)
        cla_out, _ = self.attention(
            query=ty,
            key=cla_out,
            value=cla_out,
            dropout=self.dropout,
            mask=token_type_ids
        )
        contain_entity_logits = self.contain_entity_fc(cla_out).squeeze(-1)

        # ner
        ent_out = self.entity_mid_linear(seq_out)
        ent_out = self.conditional_layer_norm(ent_out, ty)

        start_logits = self.start_fc(ent_out).squeeze(-1)
        end_logits = self.end_fc(ent_out).squeeze(-1)
        if contain_entity is None and start_ids is None and end_ids is None:
            # predicate
            return start_logits, end_logits, contain_entity_logits
        else:

            active_loss = token_type_ids.view(-1).float()
            start_logits = start_logits.view(-1)
            end_logits = end_logits.view(-1)

            start_ids = start_ids.view(-1)
            end_ids = end_ids.view(-1)

            start_loss = self.criterion(start_logits, start_ids.float())
            end_loss = self.criterion(end_logits, end_ids.float())

            if cla is not None:

                ce_loss = self.criterion(contain_entity_logits, contain_entity.float())
                start_loss = (start_loss * active_loss).sum() / active_loss.sum()
                end_loss = (end_loss * active_loss).sum() / active_loss.sum()
                return task_rate * ce_loss.mean() + start_loss + end_loss
            else:
                # 源领域训练策略
                # (batch,)
                # nums of pseudo data
                one = torch.ones_like(contain_entity_logits)
                zero = torch.zeros_like(contain_entity_logits)
                contain_entity_logits = torch.where(contain_entity_logits >= 0.5, one, zero)
                pseudo = contain_entity_logits.long().eq_(contain_entity)  # 相同的是1，不同的是0
                pseudo = torch.cat((pseudo.unsqueeze(-1),) * self.max_len, dim=-1).view(-1)
                start_loss = (pseudo * start_loss * active_loss).sum() / active_loss.sum()\
                             + (rate * (1 - pseudo) * start_loss * active_loss).sum() / active_loss.sum()
                end_loss = (pseudo * end_loss * active_loss).sum() / active_loss.sum()\
                           + (rate * (1 - pseudo) * end_loss * active_loss).sum() / active_loss.sum()

                return start_loss + end_loss

    def attention(self, query, key, value, mask=None, dropout=None):
        """
        Compute 'Scaled Dot Product Attention'
        :param query:
        :param key:
        :param value:
        :param mask: 掩码张量
        :param dropout: nn.Dropout 实例对象
        :return:
        """
        """
        自注意力机制中 query = key = value
        假设 query 为 batch_size x seq_len x d_model 的 tensor
        """
        # 取 query 最后一维的大小， 即 512
        d_k = query.size(-1)
        # 计算分数：先将 Key 的最后两个维度转置，再除缩放系数
        scores = torch.matmul(query.unsqueeze(1), key.transpose(-2, -1)) / math.sqrt(d_k)
        mask = mask.unsqueeze(1)
        if mask is not None:
            # 将一些信息掩盖，即将注意力不集中在这里
            # 将掩码张量和 scores 张量上每个位置一一比较，如果掩码张量等于0，则对应位置就用 -1e9 这个值填充
            scores = scores.masked_fill(mask == 0, -1e9)
        # 对 scores 最后一维做 softmax
        p_attn = F.softmax(scores, dim=-1)
        # 对softmax 结果做 dropout
        if dropout is not None:
            p_attn = dropout(p_attn)
        # (48, 1, 200) matmul  (48, 200, 8)
        out = torch.matmul(p_attn, value).squeeze(1)
        # out = F.relu(out)
        return out, p_attn
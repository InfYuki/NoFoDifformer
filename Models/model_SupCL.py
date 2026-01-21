import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
from torch.nn.parameter import Parameter


class SupConLoss(nn.Module):
    """监督对比学习损失函数: https://arxiv.org/pdf/2004.11362.pdf.
    也支持SimCLR中的无监督对比学习损失"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """计算模型的损失。如果'labels'和'mask'都为None，
        则退化为SimCLR无监督损失: https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: 隐藏向量，形状为 [bsz, n_views, ...].
            labels: 真实标签，形状为 [bsz].
            mask: 对比掩码，形状为 [bsz, bsz]，如果样本j与样本i属于同一类，
                  则mask_{i,j}=1。可以是非对称的。
        Returns:
            损失标量。
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features`需要是[bsz, n_views, ...],'
                             '至少需要3个维度')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('不能同时定义`labels`和`mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('标签数量与特征数量不匹配')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('未知模式: {}'.format(self.contrast_mode))

        # 计算logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # 为了数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 平铺掩码
        mask = mask.repeat(anchor_count, contrast_count)
        # 掩盖自对比情况
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 计算正样本对的对数似然平均值
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # 损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class SineEncoding(nn.Module):
    def __init__(self, k, hidden_dim=128):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_ws = nn.ModuleList([nn.Linear(hidden_dim + 1, 1) for i in range(k)])
        self.k = k

    def forward(self, e):
        # input:  [N]
        # output: [N, k]
        out_e = []
        ee = e.unsqueeze(1)
        for i in range(self.k):
            eeig = torch.full(ee.shape, torch.tensor(1.0)).to(e.device)
            ei = ee.pow(i + 1)
            div = torch.FloatTensor(np.arange(1, int(self.hidden_dim / 2) + 1)).to(e.device)
            pe = ei * div
            eeig = torch.cat((eeig, torch.sin(pe), torch.cos(pe)), dim=1)
            out_e.append(self.eig_ws[i](eeig))
        return out_e


class FeedForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout_rate):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.hidden_size = hidden_size

        self.linear_q = nn.Linear(hidden_size, num_heads * hidden_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * hidden_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * hidden_size)

        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * hidden_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        q = self.linear_q(q)  # (183,64)
        k = self.linear_k(k)
        v = self.linear_v(v)

        k = k.transpose(0, 1)  # (64,183)

        x = k.matmul(v)

        if attn_bias is not None:
            x = x + attn_bias
        x = self.att_dropout(x)
        x = torch.matmul(q, x)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class NoFoDifformer_SupCL(nn.Module):
    """使用监督对比学习的NoFoDifformer模型"""

    def __init__(self, nclass, nfeat, nlayer=1, hidden_dim=128, dim=32, nheads=1, k=10,
                 tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none',
                 proj_dim=128, temperature=0.07, use_diff_attn=False, cl_weight=0.1):
        super(NoFoDifformer_SupCL, self).__init__()

        self.norm = norm
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.temperature = temperature
        self.cl_weight = cl_weight  # 对比学习损失权重
        self.use_diff_attn = use_diff_attn  # 是否使用差分注意力

        # 特征编码器
        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 改为hidden_dim，用于投影
        )

        # 分类头
        self.classifier = nn.Linear(hidden_dim, nclass)

        # 投影头 (用于对比学习)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

        self.eig_encoder = SineEncoding(k, dim)

        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.prop_dropout = nn.Dropout(prop_dropout)

        self.k = k
        self.alpha = nn.Linear(self.k, 1, bias=False)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)

        if norm == 'none':
            self.mha_norm = nn.LayerNorm(hidden_dim)
            self.ffn_norm = nn.LayerNorm(hidden_dim)
            self.mha = MultiHeadAttention(hidden_dim, nheads, tran_dropout)
            self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)
        else:
            self.mha_norm = nn.LayerNorm(hidden_dim)
            self.ffn_norm = nn.LayerNorm(hidden_dim)
            self.mha = MultiHeadAttention(hidden_dim, nheads, tran_dropout)
            self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)

        # 对比学习损失
        self.supcon_loss = SupConLoss(temperature=temperature)

    def transformer_encoder(self, h, h_fur):
        mha_h = self.mha_norm(h)
        mha_h = self.mha(mha_h, mha_h, mha_h)
        mha_h_ = h + self.mha_dropout(mha_h) + h_fur

        ffn_h = self.ffn_norm(mha_h_)
        ffn_h = self.ffn(ffn_h)
        encoder_h = mha_h_ + self.ffn_dropout(ffn_h)
        return encoder_h

    def forward(self, e, u, x, y=None, train_idx=None, is_train=True):
        """
        前向传播函数
        Args:
            e: 特征值
            u: 特征向量
            x: 节点特征
            y: 标签，可选
            train_idx: 训练索引，可选
            is_train: 是否为训练模式
        Returns:
            logits: 分类logits
            total_loss: 总损失（如果是训练模式且提供了标签）
            cls_loss: 分类损失（如果是训练模式且提供了标签）
            sup_loss: 对比损失（如果是训练模式且提供了标签）
        """
        N = e.size(0)
        ut = u.permute(1, 0)

        # 特征编码
        h = self.feat_dp1(x)
        h = self.feat_encoder(h)

        # 特征传播
        eig = self.eig_encoder(e)
        new_e = torch.cat(eig, dim=1)
        new_e = self.alpha(new_e)

        # Transformer编码
        for conv in range(self.nlayer):
            utx = ut @ h
            h_encoder = u @ (new_e * utx)
            h_encoder = self.prop_dropout(h_encoder)
            h = self.transformer_encoder(h, h_encoder)

        h = self.feat_dp2(h)

        # 分类输出
        logits = self.classifier(h)

        # 如果不是训练模式或没有提供标签，只返回logits
        if not is_train or y is None or train_idx is None:
            return logits, None

        # 计算分类损失
        cls_loss = F.cross_entropy(logits[train_idx], y[train_idx])

        # 只对训练节点计算对比学习损失
        train_h = h[train_idx]
        train_y = y[train_idx]

        # 直接使用节点特征进行对比学习，不进行数据增强
        # 将特征投影到对比学习空间
        z = F.normalize(self.proj_head(train_h), dim=1)

        # 构造特征形状为 [N, 1, proj_dim]，以符合SupConLoss的输入要求
        features = z.unsqueeze(1)

        # 计算对比损失 - 使用标签信息构建正样本对
        sup_loss = self.supcon_loss(features, train_y)

        # 总损失 = 分类损失 + λ * 对比损失
        total_loss = cls_loss + self.cl_weight * sup_loss

        return logits, total_loss, cls_loss, sup_loss
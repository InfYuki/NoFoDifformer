import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
from torch.nn.parameter import Parameter


class ParamNonHarmonicEncoding(nn.Module):
    """
    非谐傅里叶级数编码器（严格参数化版本）
    - 频率严格递增，且满足最小间隔 delta_min。
    - Paley–Wiener: 频率限制在 [-Omega, Omega]。
    - Bessel-type: 分离性+L2正则。
    """

    def __init__(self, k, hidden_dim=128, num_freq=None, Omega=50.0, delta_min=0.25, weight_penalty=1e-4):
        """
        Args:
            k: 外层循环次数（与原始实现一致）
            hidden_dim: 输入维度 ≈ 1 + 2*num_freq
            num_freq: 使用的频率个数（默认 hidden_dim//2）
            Omega: 最大频率（带宽约束）
            delta_min: 最小间隔（保证频率分离，避免病态）
            weight_penalty: L2 正则项系数
        """
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim 必须为偶数"
        self.k = k
        self.Omega = Omega
        self.delta_min = delta_min
        self.weight_penalty = weight_penalty

        self.num_freq = num_freq if num_freq is not None else hidden_dim // 2

        # --- 参数化频率 ---
        # 我们用 softplus 保证间隔 ≥ delta_min
        # freq_deltas > 0
        self.freq_deltas = nn.Parameter(torch.rand(self.num_freq) * 0.5)

        # 一个偏置参数，决定频率起点（放在 [-Omega,Omega] 内）
        self.freq_bias = nn.Parameter(torch.tensor(0.0))

        # 线性层（输出）
        in_dim = 1 + 2 * self.num_freq
        self.readouts = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(k)])

        self.register_buffer("one_bias", torch.tensor(1.0))

    def _construct_freqs(self):
        """
        严格递增频率构造:
        λ_0 = bias
        λ_{i} = λ_{i-1} + delta_min + softplus(freq_deltas[i])
        然后映射到 [-Omega, Omega] （用 tanh 压缩）
        """
        deltas = F.softplus(self.freq_deltas) + self.delta_min  # 确保 ≥ delta_min
        freqs = torch.cumsum(deltas, dim=0) + self.freq_bias  # 递增
        # 压缩到 [-Omega, Omega]
        freqs = self.Omega * torch.tanh(freqs / self.Omega)
        return freqs

    def _weight_penalty(self):
        reg = 0.0
        for lin in self.readouts:
            reg = reg + (lin.weight ** 2).sum()
        return reg * self.weight_penalty

    def regularization_loss(self):
        return self._weight_penalty()

    def forward(self, e):
        e = e.view(-1)
        ee = e.unsqueeze(1)

        freqs = self._construct_freqs()  # [num_freq]
        norm_scale = (self.num_freq ** 0.5)

        outs = []
        for i in range(self.k):
            ei = ee.pow(i + 1)
            phase = ei * freqs.unsqueeze(0)

            feat = torch.cat([
                torch.ones_like(ee) * self.one_bias,  # 常数通道
                torch.sin(phase) / norm_scale,
                torch.cos(phase) / norm_scale,
            ], dim=1)

            outs.append(self.readouts[i](feat))
        return outs


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

        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        k = k.transpose(0, 1)

        x = k.matmul(v)

        if attn_bias is not None:
            x = x + attn_bias
        x = self.att_dropout(x)
        x = torch.matmul(q, x)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class NoFoDifformer_CL(nn.Module):

    def __init__(self, nclass, nfeat, nlayer=1, hidden_dim=128, dim=32, nheads=1, k=10,
                 tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none',
                 temperature=0.5, cl_weight=0.1):
        super(NoFoDifformer_CL, self).__init__()

        self.norm = norm
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.temperature = temperature
        self.cl_weight = cl_weight  # 对比学习权重系数

        # 特征编码器
        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nclass),
        )

        # 图傅里叶KAN编码器
        self.eig_encoder = ParamNonHarmonicEncoding(k, dim)

        # Transformer相关组件
        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.prop_dropout = nn.Dropout(prop_dropout)

        self.k = k
        self.alpha = nn.Linear(self.k, 1, bias=False)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)

        # 设置Transformer组件
        if norm == 'none':
            self.mha_norm = nn.LayerNorm(nclass)
            self.ffn_norm = nn.LayerNorm(nclass)
            self.mha = MultiHeadAttention(nclass, nheads, tran_dropout)
            self.ffn = FeedForwardNetwork(nclass, nclass, nclass)
        else:
            self.mha_norm = nn.LayerNorm(hidden_dim)
            self.ffn_norm = nn.LayerNorm(hidden_dim)
            self.mha = MultiHeadAttention(hidden_dim, nheads, tran_dropout)
            self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)
            self.classify = nn.Linear(hidden_dim, nclass)

        # 投影头 - 用于对比学习正则项
        proj_dim = nclass if norm == 'none' else hidden_dim
        self.proj_head_transformer = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim // 2)
        )

        self.proj_head_fourier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim // 2)
        )

        # 主要监督任务输出层
        self.main_output_layer = nn.Linear(2 * proj_dim, proj_dim)
        if norm != 'none':
            self.final_classifier = nn.Linear(proj_dim, nclass)

    def transformer_encoder(self, h, h_fur):
        mha_h = self.mha_norm(h)
        mha_h = self.mha(mha_h, mha_h, mha_h)
        mha_h_ = h + self.mha_dropout(mha_h) + h_fur

        ffn_h = self.ffn_norm(mha_h_)
        ffn_h = self.ffn(ffn_h)
        encoder_h = mha_h_ + self.ffn_dropout(ffn_h)
        return encoder_h

    def contrastive_loss(self, trans_feat, fourier_feat):
        # 归一化特征
        trans_feat = F.normalize(trans_feat, dim=1)
        fourier_feat = F.normalize(fourier_feat, dim=1)

        # 计算相似度矩阵
        logits = torch.matmul(trans_feat, fourier_feat.T) / self.temperature

        # 对角线元素是正样本对
        labels = torch.arange(logits.size(0), device=logits.device)

        # 计算对比损失（InfoNCE损失）
        loss_i = F.cross_entropy(logits, labels)
        loss_j = F.cross_entropy(logits.T, labels)

        return (loss_i + loss_j) / 2.0

    def forward(self, e, u, x, y=None, train_idx=None):
        """
        监督任务 + 对比正则项的前向传播
        Args:
            e: 特征值
            u: 特征向量
            x: 节点特征
            y: 标签（可选，用于计算监督损失）
            train_idx: 训练节点索引（可选，用于指定计算损失的节点）
        Returns:
            主任务输出和总损失（如果提供了标签）
        """
        N = e.size(0)
        ut = u.permute(1, 0)

        # 特征编码
        if self.norm == 'none':
            h = self.feat_dp1(x)
            h = self.feat_encoder(h)
            h = self.feat_dp2(h)
        else:
            h = self.feat_dp1(x)
            h = self.feat_encoder(h)

        # 图傅里叶KAN编码
        eig = self.eig_encoder(e)
        new_e = torch.cat(eig, dim=1)
        new_e = self.alpha(new_e)

        # Transformer分支
        h_transformer = h.clone()

        # 图傅里叶KAN分支
        h_fourier = h.clone()

        for conv in range(self.nlayer):
            # Transformer分支更新
            utx_t = ut @ h_transformer
            h_encoder_t = u @ (new_e * utx_t)
            h_encoder_t = self.prop_dropout(h_encoder_t)
            h_transformer = self.transformer_encoder(h_transformer, h_encoder_t)

            # 图傅里叶KAN分支更新
            utx_f = ut @ h_fourier
            h_fourier = u @ (new_e * utx_f)
            h_fourier = self.prop_dropout(h_fourier)

        # 投影用于对比学习
        z_transformer = self.proj_head_transformer(h_transformer)
        z_fourier = self.proj_head_fourier(h_fourier)

        # 计算对比损失作为正则项
        cl_loss = self.contrastive_loss(z_transformer, z_fourier)

        # 主要监督任务输出
        main_output = self.main_output_layer(torch.cat([h_transformer, h_fourier], dim=1))

        # 最终分类输出
        if self.norm == 'none':
            logits = main_output
        else:
            main_output = self.feat_dp2(main_output)
            logits = self.final_classifier(main_output)

        # 如果提供了标签和训练索引，计算监督损失和总损失
        if y is not None and train_idx is not None:
            # 只对训练节点计算监督损失
            sup_loss = F.cross_entropy(logits[train_idx], y)

            # 总损失 = 监督损失 + cl_weight * 对比损失
            total_loss = sup_loss + self.cl_weight * cl_loss

            return logits, total_loss, sup_loss, cl_loss
        else:
            # 如果没有标签或训练索引，只返回预测结果和对比损失
            return logits, cl_loss
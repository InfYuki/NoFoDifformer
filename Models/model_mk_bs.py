import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
from torch.nn.parameter import Parameter


# ========================= Fourier-KAN Layer =========================
class FourierKANLayer(nn.Module):
    """
    Fourier-KAN: 用傅里叶级数展开替代 KAN 的 spline
    - 每条边的函数 φ(λ) = bias + Σ (a_k cos(kλ) + b_k sin(kλ))
    - 节点只做加和
    """

    def __init__(self, num_basis=10, Omega=np.pi):
        super().__init__()
        self.num_basis = num_basis
        self.Omega = Omega

        # 可学习参数，相当于 spline 的控制点
        self.a = nn.Parameter(torch.randn(num_basis))  # cos 系数
        self.b = nn.Parameter(torch.randn(num_basis))  # sin 系数
        self.bias = nn.Parameter(torch.zeros(1))

    def edge_function(self, lambdas):
        """
        lambdas: [N] 图拉普拉斯特征值
        return: [N] 对应傅里叶-KAN边函数值
        """
        out = self.bias.expand_as(lambdas)
        for k in range(1, self.num_basis + 1):
            out = out + self.a[k - 1] * torch.cos(k * lambdas / self.Omega) \
                      + self.b[k - 1] * torch.sin(k * lambdas / self.Omega)
        return out

    def forward(self, lambdas):
        """
        lambdas: [N] 特征值 (已经在数据集中给出)
        return: [N,1] 经过傅里叶-KAN 变换的谱滤波系数
        """
        phi_lambda = self.edge_function(lambdas).unsqueeze(-1)
        return phi_lambda


# ========================= FeedForward =========================
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


# ========================= MultiHeadAttention (Diff Version) =========================
def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class Diff_MultiHeadAttention_Optimized(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout_rate):
        super(Diff_MultiHeadAttention_Optimized, self).__init__()

        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5
        self.hidden_size = hidden_size

        self.linear_q1 = nn.Linear(hidden_size, num_heads * hidden_size)
        self.linear_k1 = nn.Linear(hidden_size, num_heads * hidden_size)

        self.linear_q2 = nn.Linear(hidden_size, num_heads * hidden_size)
        self.linear_k2 = nn.Linear(hidden_size, num_heads * hidden_size)

        self.linear_v = nn.Linear(hidden_size, num_heads * hidden_size)

        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.layer_norm = nn.LayerNorm(num_heads * hidden_size)
        self.output_layer = nn.Linear(num_heads * hidden_size, hidden_size)

        self.lambda_init = lambda_init_fn(0)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.hidden_size).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.hidden_size).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.hidden_size).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.hidden_size).normal_(mean=0, std=0.1))

    def forward(self, q, k, v, layer, attn_bias=None):
        orig_q_size = q.size()

        q1 = self.linear_q1(q)
        k1 = self.linear_k1(k)
        q2 = self.linear_q2(q)
        k2 = self.linear_k2(k)
        v = self.linear_v(v)

        # 优化：先计算 KV
        k1v = k1.transpose(0, 1).matmul(v)
        k2v = k2.transpose(0, 1).matmul(v)

        k1v = self.att_dropout(k1v)
        k2v = self.att_dropout(k2v)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        self.lambda_init = lambda_init_fn(layer)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        x1 = torch.matmul(q1, k1v)
        x2 = torch.matmul(q2, k2v) * lambda_full
        x = x1 - x2

        x = self.layer_norm(x)
        x = x * (1 - self.lambda_init)
        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


# ========================= NoFoDifformer + FourierKAN =========================
class NoFoDifformer_FourierKAN(nn.Module):
    def __init__(self, nclass, nfeat, nlayer=1, hidden_dim=128, dim=32, nheads=1, k=10,
                 tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none'):
        super(NoFoDifformer_FourierKAN, self).__init__()

        self.norm = norm
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        self.dim = dim

        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nclass),
        )

        # 替换为 Fourier-KAN 编码器
        self.eig_encoder = FourierKANLayer(num_basis=k, Omega=np.pi)

        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.prop_dropout = nn.Dropout(prop_dropout)

        self.k = k
        self.alpha = nn.Linear(1, 1, bias=False)  # 简化：只缩放一维傅里叶滤波结果

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)

        # 确定特征维度
        feat_dim = nclass if norm == 'none' else hidden_dim

        if norm == 'none':
            self.mha_norm = nn.LayerNorm(nclass)
            self.ffn_norm = nn.LayerNorm(nclass)
            self.mha = Diff_MultiHeadAttention_Optimized(nclass, nheads, tran_dropout)
            self.ffn = FeedForwardNetwork(nclass, nclass, nclass)
        else:
            self.mha_norm = nn.LayerNorm(hidden_dim)
            self.ffn_norm = nn.LayerNorm(hidden_dim)
            self.mha = Diff_MultiHeadAttention_Optimized(hidden_dim, nheads, tran_dropout)
            self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)
            self.classify = nn.Linear(hidden_dim, nclass)

    def transformer_encoder(self, h, h_fur, layer):
        mha_h = self.mha_norm(h)
        mha_h = self.mha(mha_h, mha_h, mha_h, layer)
        mha_h_ = h + self.mha_dropout(mha_h) + h_fur

        ffn_h = self.ffn_norm(mha_h_)
        ffn_h = self.ffn(ffn_h)
        encoder_h = mha_h_ + self.ffn_dropout(ffn_h)
        return encoder_h

    def forward(self, e, u, x):
        """
        e: 边输入 (暂时保留不变，可不用)
        u: 传播矩阵
        x: 节点特征
        U: 图傅里叶基 [N,N]
        Lambda: 特征值 [N]
        """
        N = e.size(0)
        ut = u.permute(1, 0)

        if self.norm == 'none':
            h = self.feat_dp1(x)
            h = self.feat_encoder(h)
            h = self.feat_dp2(h)
        else:
            h = self.feat_dp1(x)
            h = self.linear_encoder(h)

        # Fourier-KAN 滤波
        eig = self.eig_encoder(e)  # 这里的 e 就是 Lambda
        new_e = self.alpha(eig)

        for conv in range(self.nlayer):
            utx = ut @ h
            h_encoder = u @ (new_e * utx)
            h_encoder = self.prop_dropout(h_encoder)
            h = self.transformer_encoder(h, h_encoder, conv)

        if self.norm == 'none':
            return h
        else:
            h = self.feat_dp2(h)
            h = self.classify(h)
            return h

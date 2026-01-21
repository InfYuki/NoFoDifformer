import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
from torch.nn.parameter import Parameter


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
        freqs = torch.cumsum(deltas, dim=0) + self.freq_bias    # 递增
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

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class Diff_MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout_rate):
        super(Diff_MultiHeadAttention, self).__init__()

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

        # 添加Layer Norm
        self.layer_norm = nn.LayerNorm(num_heads * hidden_size)

        self.output_layer = nn.Linear(num_heads * hidden_size, hidden_size)

        self.lambda_init = lambda_init_fn(0)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.hidden_size, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.hidden_size, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.hidden_size, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.hidden_size, dtype=torch.float32).normal_(mean=0, std=0.1))


    def forward(self, q, k, v, layer, attn_bias=None):
        orig_q_size = q.size()

        #print(q.size(), k.size(), v.size())
        #print(self.hidden_size)
        q1 = self.linear_q1(q)  # (183,64)
        k1 = self.linear_k1(k)


        q2 = self.linear_q2(q)  # (183,64)
        k2 = self.linear_k2(k)

        v = self.linear_v(v)
        #print(q1.size(), k1.size(), v.size())

        k1 = k1.transpose(0, 1)  # (64,183)
        k2 = k2.transpose(0, 1)  # (64,183)
        #print(q.size(), k.size(), v.size())


        attn_weights1 = torch.matmul(q1, k1)
        attn_weights2 = torch.matmul(q2, k2)
        attn_weights1 = self.att_dropout(attn_weights1)
        attn_weights2 = self.att_dropout(attn_weights2)
        #print(attn_weights1.size())

        attn_weights1 = torch.nan_to_num(attn_weights1)
        attn_weights2 = torch.nan_to_num(attn_weights2)
        attn_weights1 = F.softmax(attn_weights1, dim=-1, dtype=torch.float32).type_as(attn_weights1)
        attn_weights2 = F.softmax(attn_weights2, dim=-1, dtype=torch.float32).type_as(attn_weights2)
        #print(attn_weights1.size())

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        #print(self.lambda_q1.size(), self.lambda_k2.size())


        self.lambda_init=lambda_init_fn(layer)
        #print(self.lambda_init)

        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        diff_attn_weights = attn_weights1-attn_weights2 * lambda_full
        #print(diff_attn_weights.size())
        #print(lambda_full.size())
        attn = torch.matmul(diff_attn_weights, v)
        #print(attn.size())

        attn = self.layer_norm(attn)

        attn = attn * (1 - self.lambda_init)

        x = self.output_layer(attn)
        assert x.size() == orig_q_size
        #print(x.size())
        return x


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
        self.lambda_q1 = nn.Parameter(torch.zeros(self.hidden_size, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.hidden_size, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.hidden_size, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.hidden_size, dtype=torch.float32).normal_(mean=0, std=0.1))

    def forward(self, q, k, v, layer, attn_bias=None):
        orig_q_size = q.size()

        q1 = self.linear_q1(q)
        k1 = self.linear_k1(k)
        q2 = self.linear_q2(q)
        k2 = self.linear_k2(k)
        v = self.linear_v(v)

        # 优化计算顺序，先计算k与v的乘积
        k1v = k1.transpose(0, 1).matmul(v)
        k2v = k2.transpose(0, 1).matmul(v)

        # 应用dropout
        k1v = self.att_dropout(k1v)
        k2v = self.att_dropout(k2v)

        # 计算lambda系数
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        self.lambda_init = lambda_init_fn(layer)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # 计算最终结果
        x1 = torch.matmul(q1, k1v)
        x2 = torch.matmul(q2, k2v) * lambda_full
        x = x1 - x2

        x = self.layer_norm(x)
        x = x * (1 - self.lambda_init)
        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x





class NoFoDifformer_mk(nn.Module):

    def __init__(self, nclass, nfeat, nlayer=1, hidden_dim=128, dim=32, nheads=1, k=10,
                 tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none'):
        super(NoFoDifformer_mk, self).__init__()

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

        self.eig_encoder = ParamNonHarmonicEncoding(k, dim)

        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.prop_dropout = nn.Dropout(prop_dropout)

        self.k = k
        self.alpha = nn.Linear(self.k, 1, bias=False)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)

        # 确定特征维度
        feat_dim = nclass if norm == 'none' else hidden_dim

        # 添加门控机制
        self.gate_norm = nn.LayerNorm(feat_dim * 3)
        self.gate_linear = nn.Linear(feat_dim * 3, 3)  # 三个门控权重

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
        mha_h_dropout = self.mha_dropout(mha_h)
        #mha_h_ = h + self.mha_dropout(mha_h) + h_fur

        # 使用门控机制替代直接相加
        # 将三个特征拼接起来
        concat_features = torch.cat([h, mha_h_dropout, h_fur], dim=-1)
        # 计算门控权重
        gate_weights = self.gate_norm(concat_features)
        gate_weights = self.gate_linear(gate_weights)
        gate_weights = F.softmax(gate_weights, dim=-1).unsqueeze(-1)

        # 使用门控权重进行加权融合
        mha_h_ = h * gate_weights[:, 0] + mha_h_dropout * gate_weights[:, 1] + h_fur * gate_weights[:, 2]

        ffn_h = self.ffn_norm(mha_h_)
        ffn_h = self.ffn(ffn_h)
        encoder_h = mha_h_ + self.ffn_dropout(ffn_h)
        return encoder_h

    def forward(self, e, u, x):
        N = e.size(0)
        ut = u.permute(1, 0)

        if self.norm == 'none':
            h = self.feat_dp1(x)
            h = self.feat_encoder(h)
            h = self.feat_dp2(h)
        else:
            h = self.feat_dp1(x)
            h = self.linear_encoder(h)

        eig = self.eig_encoder(e)
        new_e = torch.cat(eig, dim=1)
        new_e = self.alpha(new_e)

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
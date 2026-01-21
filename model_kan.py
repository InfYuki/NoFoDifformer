import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
from torch.nn.parameter import Parameter
from kan.KANLayer import KANLayer


class NonHarmonicBaseFunction(nn.Module):
    """非谐波傅里叶基函数，用作KAN的基函数"""

    def __init__(self, num_freq=8, Omega=50.0, delta_min=0.25):
        super().__init__()
        self.Omega = Omega
        self.delta_min = delta_min
        self.num_freq = num_freq

        # 参数化频率
        self.freq_deltas = nn.Parameter(torch.rand(num_freq) * 0.5)
        self.freq_bias = nn.Parameter(torch.tensor(0.0))

        # 傅里叶系数
        self.a_coef = nn.Parameter(torch.randn(num_freq) * 0.1)  # cos系数
        self.b_coef = nn.Parameter(torch.randn(num_freq) * 0.1)  # sin系数
        self.bias = nn.Parameter(torch.zeros(1))

    def _construct_freqs(self):
        """构造递增频率"""
        deltas = F.softplus(self.freq_deltas) + self.delta_min
        freqs = torch.cumsum(deltas, dim=0) + self.freq_bias
        freqs = self.Omega * torch.tanh(freqs / self.Omega)
        return freqs

    def forward(self, x):
        """前向传播"""
        shape = x.shape
        x_flat = x.view(-1)
        freqs = self._construct_freqs()

        out = self.bias.expand_as(x_flat)
        for i in range(self.num_freq):
            out = out + self.a_coef[i] * torch.cos(freqs[i] * x_flat) + \
                  self.b_coef[i] * torch.sin(freqs[i] * x_flat)

        return out.view(shape)

# ========================= 简化版多层KAN编码器 =========================
class MultiLayerKANEncoding(nn.Module):
    """
    多层KAN编码器 - 保留model_mk中的结构风格，但使用KAN层
    - 每一层都是独立的KANLayer
    - 支持多层堆叠，每层可配置不同的参数
    - 输出仍然保持与ParamNonHarmonicEncoding兼容
    """

    def __init__(self, k, hidden_dim=128, num_layers=2, grid_num=5, spline_k=3,
                 noise_scale=0.3, grid_range=[-5, 5]):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 确保参数为整数
        grid_num = int(grid_num)
        spline_k = int(spline_k)

        # 创建k个独立的KAN编码器通道
        self.kan_channels = nn.ModuleList([
            self._create_kan_layers(num_layers, grid_num, spline_k, noise_scale, grid_range)
            for _ in range(k)
        ])

        # 最终的线性映射层
        self.readouts = nn.ModuleList([nn.Linear(1, 1) for _ in range(k)])

    def _create_kan_layers(self, num_layers, grid_num, spline_k, noise_scale, grid_range):
        """创建一个多层KAN序列"""
        layers = []
        for i in range(num_layers):
            # 第一层输入维度为1，中间层输入输出维度相同
            in_dim = 1 if i == 0 else 1
            out_dim = 1

            # 创建KAN层
            kan_layer = KANLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                num=grid_num,
                k=spline_k,
                base_fun=NonHarmonicBaseFunction(num_freq=8),  ###
                noise_scale=noise_scale,
                grid_range=grid_range
            )
            layers.append(kan_layer)

        return nn.ModuleList(layers)

    def forward(self, e):
        """
        输入: e [N] 特征值
        输出: 与ParamNonHarmonicEncoding兼容的输出列表 [N,1] * k
        """
        e = e.view(-1)  # 确保是一维向量
        ee = e.unsqueeze(1)  # [N,1]

        outs = []
        for i in range(self.k):
            # 通过当前通道的所有KAN层
            x = ee.clone()
            for layer in self.kan_channels[i]:
                x, _, _, _ = layer(x)  # 只取KAN的主要输出

            # 最终线性映射
            out = self.readouts[i](x)
            outs.append(out)

        return outs


# ========================= NoFoDifformer与多层KAN集成 =========================
class NoFoDifformer_MultiLayerKAN(nn.Module):
    """
    NoFoDifformer与多层KAN的结合，保留model_mk的大部分结构
    """

    def __init__(self, nclass, nfeat, nlayer=1, hidden_dim=128, dim=32, nheads=1, k=10,
                 kan_layers=2, kan_grid_num=5, kan_k=3, kan_noise_scale=0.3,
                 tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none'):
        super().__init__()

        # 确保整数参数
        kan_layers = int(kan_layers)
        kan_grid_num = int(kan_grid_num)
        kan_k = int(kan_k)

        self.norm = norm
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        self.dim = dim

        # 特征编码器
        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nclass),
        )

        # 使用多层KAN作为特征值编码器，替换原来的ParamNonHarmonicEncoding
        self.eig_encoder = MultiLayerKANEncoding(
            k=k,
            hidden_dim=dim,
            num_layers=kan_layers,
            grid_num=kan_grid_num,
            spline_k=kan_k,
            noise_scale=kan_noise_scale,
            grid_range=[-5, 5]
        )

        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.prop_dropout = nn.Dropout(prop_dropout)

        self.k = k
        self.alpha = nn.Linear(k, 1, bias=False)  # 与model_mk相同的输出映射

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)

        # 确定特征维度
        feat_dim = nclass if norm == 'none' else hidden_dim

        # 保留model_mk中的门控机制
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

        # 使用门控机制，与model_mk一致
        concat_features = torch.cat([h, mha_h_dropout, h_fur], dim=-1)
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
        """
        e: 特征值 [N]
        u: 传播矩阵 [N,N]
        x: 节点特征 [N,F]
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

        # KAN编码特征值 - 保持与model_mk的接口一致
        eig_outputs = self.eig_encoder(e)
        new_e = torch.cat(eig_outputs, dim=1)
        new_e = self.alpha(new_e)

        # 图传播与Transformer处理
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
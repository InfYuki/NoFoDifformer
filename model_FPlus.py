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


class LaplacianEncoding(nn.Module):
    def __init__(self, k, hidden_dim=128):
        super(LaplacianEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.eig_ws = nn.ModuleList([nn.Linear(hidden_dim + 1, 1) for i in range(k)])
        self.k = k
        # 拉普拉斯变换的衰减因子
        self.decay_factor = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, e):
        # input:  [N]
        # output: [N, k]
        out_e = []
        ee = e.unsqueeze(1)
        for i in range(self.k):
            # 常数项
            eeig = torch.full(ee.shape, torch.tensor(1.0)).to(e.device)
            ei = ee.pow(i + 1)
            div = torch.FloatTensor(np.arange(1, int(self.hidden_dim / 2) + 1)).to(e.device)
            pe = ei * div

            # 拉普拉斯变换的实部和虚部
            # s = σ + jω，使用可学习的衰减因子σ
            sigma = self.decay_factor * torch.ones_like(pe).to(e.device)
            real_part = torch.exp(-sigma * pe) * torch.cos(pe)  # e^(-σt)cos(ωt)
            imag_part = torch.exp(-sigma * pe) * torch.sin(pe)  # e^(-σt)sin(ωt)

            # 连接常数项和拉普拉斯基函数
            eeig = torch.cat((eeig, real_part, imag_part), dim=1)
            out_e.append(self.eig_ws[i](eeig))
        return out_e

class WaveletEncoding(nn.Module):
    def __init__(self, k, hidden_dim=128):
        super(WaveletEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.eig_ws = nn.ModuleList([nn.Linear(hidden_dim + 1, 1) for i in range(k)])
        self.k = k

    def forward(self, e):
        # input: [N]
        # output: [N, k]
        out_e = []
        ee = e.unsqueeze(1)
        for i in range(self.k):
            eeig = torch.full(ee.shape, torch.tensor(1.0)).to(e.device)
            ei = ee.pow(i + 1)
            div = torch.FloatTensor(np.arange(1, int(self.hidden_dim / 2) + 1)).to(e.device)
            pe = ei * div
            # 使用小波基函数替代正弦余弦
            morlet = torch.exp(-0.5 * (pe ** 2)) * torch.cos(5 * pe)
            mexican_hat = (1 - pe ** 2) * torch.exp(-0.5 * pe ** 2)
            eeig = torch.cat((eeig, morlet, mexican_hat), dim=1)
            out_e.append(self.eig_ws[i](eeig))
        return out_e

class LegendreEncoding(nn.Module):
    def __init__(self, k, hidden_dim=128):
        super(LegendreEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.eig_ws = nn.ModuleList([nn.Linear(hidden_dim + 1, 1) for i in range(k)])
        self.k = k

    def forward(self, e):
        # input:  [N]
        # output: [N, k]
        out_e = []
        ee = e.unsqueeze(1)
        for i in range(self.k):
            # 常数项
            eeig = torch.full(ee.shape, torch.tensor(1.0)).to(e.device)
            ei = ee.pow(i + 1)
            div = torch.FloatTensor(np.arange(1, int(self.hidden_dim / 2) + 1)).to(e.device)
            pe = ei * div

            # 归一化到[-1,1]区间，勒让德多项式定义在[-1,1]上
            x = torch.tanh(pe / 5.0)

            # 生成勒让德多项式
            # P_0(x) = 1, P_1(x) = x
            p0 = torch.ones_like(x)
            p1 = x

            # 存储所有阶数的勒让德多项式
            legendre_list = [p0, p1]

            # 使用递推公式生成高阶勒让德多项式
            # (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
            for n in range(1, self.hidden_dim // 2 - 1):
                p_next = ((2 * n + 1) * x * p1 - n * p0) / (n + 1)
                legendre_list.append(p_next)
                p0, p1 = p1, p_next

            # 连接所有勒让德多项式
            legendre_bases = torch.cat(legendre_list, dim=1)

            # 确保维度正确
            if legendre_bases.size(1) > self.hidden_dim:
                legendre_bases = legendre_bases[:, :self.hidden_dim]
            elif legendre_bases.size(1) < self.hidden_dim:
                padding = torch.zeros(legendre_bases.size(0), self.hidden_dim - legendre_bases.size(1)).to(e.device)
                legendre_bases = torch.cat([legendre_bases, padding], dim=1)

            # 连接常数项和勒让德多项式基函数
            eeig = torch.cat((eeig, legendre_bases), dim=1)

            # 通过线性层
            out_e.append(self.eig_ws[i](eeig))

        return out_e

'''
class LaguerreEncoding(nn.Module):
    def __init__(self, k, hidden_dim=128):
        super(LaguerreEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.eig_ws = nn.ModuleList([nn.Linear(hidden_dim + 1, 1) for i in range(k)])
        self.k = k
        # 拉盖尔多项式的衰减因子，可学习参数
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, e):
        # input:  [N]
        # output: [N, k]
        out_e = []
        ee = e.unsqueeze(1)
        for i in range(self.k):
            # 常数项
            eeig = torch.full(ee.shape, torch.tensor(1.0)).to(e.device)
            ei = ee.pow(i + 1)
            div = torch.FloatTensor(np.arange(1, int(self.hidden_dim / 2) + 1)).to(e.device)
            pe = ei * div

            # 确保输入为正值，拉盖尔多项式通常定义在[0,∞)上
            x = F.softplus(pe)

            # 生成拉盖尔多项式
            # L_0(x) = 1
            # L_1(x) = 1 - x
            l0 = torch.ones_like(x)
            l1 = 1 - x

            # 存储所有阶数的拉盖尔多项式
            laguerre_list = [l0, l1]

            # 使用递推公式生成高阶拉盖尔多项式
            # (n+1)L_{n+1}(x) = (2n+1-x)L_n(x) - nL_{n-1}(x)
            for n in range(1, self.hidden_dim // 2 - 1):
                l_next = ((2 * n + 1 - x) * l1 - n * l0) / (n + 1)
                laguerre_list.append(l_next)
                l0, l1 = l1, l_next

            # 连接所有拉盖尔多项式
            laguerre_bases = torch.cat(laguerre_list, dim=1)

            # 添加指数衰减权重 e^(-αx)
            alpha = self.alpha * torch.ones_like(x).to(e.device)
            exp_weight = torch.exp(-alpha * x)

            # 将每个多项式与衰减权重相乘 - 修复维度不匹配问题
            # 对每个拉盖尔多项式单独应用衰减权重
            weighted_bases_list = []
            for j in range(laguerre_bases.size(1)):
                weighted_bases_list.append(laguerre_bases[:, j:j + 1] * exp_weight)

            # 重新连接加权后的基函数
            weighted_bases = torch.cat(weighted_bases_list, dim=1)

            # 确保维度正确
            if weighted_bases.size(1) > self.hidden_dim:
                weighted_bases = weighted_bases[:, :self.hidden_dim]
            elif weighted_bases.size(1) < self.hidden_dim:
                padding = torch.zeros(weighted_bases.size(0), self.hidden_dim - weighted_bases.size(1)).to(e.device)
                weighted_bases = torch.cat([weighted_bases, padding], dim=1)

            # 连接常数项和拉盖尔多项式基函数
            eeig = torch.cat((eeig, weighted_bases), dim=1)

            # 通过线性层
            out_e.append(self.eig_ws[i](eeig))

        return out_e
'''

class LaguerreEncoding(nn.Module):
    def __init__(self, k, hidden_dim=128):
        super(LaguerreEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.eig_ws = nn.ModuleList([nn.Linear(hidden_dim + 1, 1) for i in range(k)])
        self.k = k
        # 拉盖尔多项式的衰减因子，可学习参数
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, e):
        # input:  [N]
        # output: [N, k]
        out_e = []
        ee = e.unsqueeze(1)
        for i in range(self.k):
            # 常数项
            eeig = torch.full(ee.shape, torch.tensor(1.0)).to(e.device)
            ei = ee.pow(i + 1)
            div = torch.FloatTensor(np.arange(1, int(self.hidden_dim / 2) + 1)).to(e.device)
            pe = ei * div

            # 确保输入为正值，拉盖尔多项式通常定义在[0,∞)上
            x = F.softplus(pe)

            # 生成拉盖尔多项式
            # L_0(x) = 1
            # L_1(x) = 1 - x
            l0 = torch.ones_like(x)
            l1 = 1 - x

            # 存储所有阶数的拉盖尔多项式
            laguerre_list = [l0, l1]

            # 使用递推公式生成高阶拉盖尔多项式
            # (n+1)L_{n+1}(x) = (2n+1-x)L_n(x) - nL_{n-1}(x)
            for n in range(1, self.hidden_dim // 2 - 1):
                l_next = ((2 * n + 1 - x) * l1 - n * l0) / (n + 1)
                laguerre_list.append(l_next)
                l0, l1 = l1, l_next

            # 连接所有拉盖尔多项式
            laguerre_bases = torch.cat(laguerre_list, dim=1)

            # 添加指数衰减权重 e^(-αx)
            alpha = self.alpha * torch.ones_like(x).to(e.device)
            exp_weight = torch.exp(-alpha * x)

            # 使用广播机制应用衰减权重
            # exp_weight形状为[N, hidden_dim/2]，需要调整为[N, 1]以便广播
            exp_weight_reshaped = torch.mean(exp_weight, dim=1, keepdim=True)
            weighted_bases = laguerre_bases * exp_weight_reshaped

            # 确保维度正确
            if weighted_bases.size(1) > self.hidden_dim:
                weighted_bases = weighted_bases[:, :self.hidden_dim]
            elif weighted_bases.size(1) < self.hidden_dim:
                padding = torch.zeros(weighted_bases.size(0), self.hidden_dim - weighted_bases.size(1)).to(e.device)
                weighted_bases = torch.cat([weighted_bases, padding], dim=1)

            # 连接常数项和拉盖尔多项式基函数
            eeig = torch.cat((eeig, weighted_bases), dim=1)

            # 通过线性层
            out_e.append(self.eig_ws[i](eeig))

        return out_e

class HermiteEncoding(nn.Module):
    def __init__(self, k, hidden_dim=128):
        super(HermiteEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.eig_ws = nn.ModuleList([nn.Linear(hidden_dim + 1, 1) for i in range(k)])
        self.k = k
        # 埃尔米特多项式的缩放因子，可学习参数
        self.scale_factor = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, e):
        # input:  [N]
        # output: [N, k]
        out_e = []
        ee = e.unsqueeze(1)
        for i in range(self.k):
            # 常数项
            eeig = torch.full(ee.shape, torch.tensor(1.0)).to(e.device)
            ei = ee.pow(i + 1)
            div = torch.FloatTensor(np.arange(1, int(self.hidden_dim / 2) + 1)).to(e.device)
            pe = ei * div

            # 缩放输入，埃尔米特多项式在实数域上定义
            # 使用可学习的缩放因子使输入适应模型
            scale = self.scale_factor * torch.ones_like(pe).to(e.device)
            x = scale * pe

            # 生成埃尔米特多项式
            # H_0(x) = 1
            # H_1(x) = 2x
            h0 = torch.ones_like(x)
            h1 = 2 * x

            # 存储所有阶数的埃尔米特多项式
            hermite_list = [h0, h1]

            # 使用递推公式生成高阶埃尔米特多项式
            # H_{n+1}(x) = 2x*H_n(x) - 2n*H_{n-1}(x)
            for n in range(1, self.hidden_dim // 2 - 1):
                h_next = 2 * x * h1 - 2 * n * h0
                hermite_list.append(h_next)
                h0, h1 = h1, h_next

            # 连接所有埃尔米特多项式
            hermite_bases = torch.cat(hermite_list, dim=1)

            # 添加高斯权重函数 e^(-x²/2)，这是埃尔米特多项式的自然权重
            gaussian_weight = torch.exp(-0.5 * x * x)

            # 使用广播机制应用高斯权重
            gaussian_weight_reshaped = torch.mean(gaussian_weight, dim=1, keepdim=True)
            weighted_bases = hermite_bases * gaussian_weight_reshaped

            # 确保维度正确
            if weighted_bases.size(1) > self.hidden_dim:
                weighted_bases = weighted_bases[:, :self.hidden_dim]
            elif weighted_bases.size(1) < self.hidden_dim:
                padding = torch.zeros(weighted_bases.size(0), self.hidden_dim - weighted_bases.size(1)).to(e.device)
                weighted_bases = torch.cat([weighted_bases, padding], dim=1)

            # 连接常数项和埃尔米特多项式基函数
            eeig = torch.cat((eeig, weighted_bases), dim=1)

            # 通过线性层
            out_e.append(self.eig_ws[i](eeig))

        return out_e

class ChebyshevEncoding(nn.Module):
    def __init__(self, k, hidden_dim=128):
        super(ChebyshevEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.eig_ws = nn.ModuleList([nn.Linear(hidden_dim + 1, 1) for i in range(k)])
        self.k = k
        # 切比雪夫多项式的缩放因子，可学习参数
        self.scale_factor = nn.Parameter(torch.ones(1) * 0.2)

    def forward(self, e):
        # input:  [N]
        # output: [N, k]
        out_e = []
        ee = e.unsqueeze(1)
        for i in range(self.k):
            # 常数项
            eeig = torch.full(ee.shape, torch.tensor(1.0)).to(e.device)
            ei = ee.pow(i + 1)
            div = torch.FloatTensor(np.arange(1, int(self.hidden_dim / 2) + 1)).to(e.device)
            pe = ei * div

            # 归一化到[-1,1]区间，切比雪夫多项式定义在[-1,1]上
            # 使用可学习的缩放因子调整归一化范围
            scale = self.scale_factor * torch.ones_like(pe).to(e.device)
            x = torch.tanh(scale * pe)

            # 生成切比雪夫多项式第一类
            # T_0(x) = 1
            # T_1(x) = x
            t0 = torch.ones_like(x)
            t1 = x

            # 存储所有阶数的切比雪夫多项式
            cheby_list = [t0, t1]

            # 使用递推公式生成高阶切比雪夫多项式
            # T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x)
            for n in range(1, self.hidden_dim // 2 - 1):
                t_next = 2 * x * t1 - t0
                cheby_list.append(t_next)
                t0, t1 = t1, t_next

            # 连接所有切比雪夫多项式
            cheby_bases = torch.cat(cheby_list, dim=1)

            # 添加切比雪夫权重函数 1/sqrt(1-x²)，这是切比雪夫多项式的自然权重
            # 为避免数值不稳定，使用平滑版本
            epsilon = 1e-6  # 小的平滑因子
            weight = 1.0 / torch.sqrt(1.0 - x * x + epsilon)

            # 使用广播机制应用权重
            weight_reshaped = torch.mean(weight, dim=1, keepdim=True)
            # 为了数值稳定性，对权重进行归一化
            weight_reshaped = torch.clamp(weight_reshaped, 0, 10)
            weighted_bases = cheby_bases * weight_reshaped

            # 确保维度正确
            if weighted_bases.size(1) > self.hidden_dim:
                weighted_bases = weighted_bases[:, :self.hidden_dim]
            elif weighted_bases.size(1) < self.hidden_dim:
                padding = torch.zeros(weighted_bases.size(0), self.hidden_dim - weighted_bases.size(1)).to(e.device)
                weighted_bases = torch.cat([weighted_bases, padding], dim=1)

            # 连接常数项和切比雪夫多项式基函数
            eeig = torch.cat((eeig, weighted_bases), dim=1)

            # 通过线性层
            out_e.append(self.eig_ws[i](eeig))

        return out_e

class DiscreteFourierEncoding(nn.Module):
    def __init__(self, k, hidden_dim=128):
        super(DiscreteFourierEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.eig_ws = nn.ModuleList([nn.Linear(hidden_dim + 1, 1) for i in range(k)])
        self.k = k
        # 离散傅里叶级数的缩放因子，可学习参数
        self.scale_factor = nn.Parameter(torch.ones(1) * 0.5)
        # 频率调整因子，可学习参数
        self.freq_factor = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, e):
        # input:  [N]
        # output: [N, k]
        out_e = []
        ee = e.unsqueeze(1)
        for i in range(self.k):
            # 常数项
            eeig = torch.full(ee.shape, torch.tensor(1.0)).to(e.device)
            ei = ee.pow(i + 1)
            div = torch.FloatTensor(np.arange(1, int(self.hidden_dim / 2) + 1)).to(e.device)
            pe = ei * div

            # 应用可学习的缩放因子
            scale = self.scale_factor * torch.ones_like(pe).to(e.device)
            freq = self.freq_factor * torch.ones_like(pe).to(e.device)
            x = scale * pe

            # 生成离散傅里叶级数的基函数
            # 对于每个频率，生成一对正弦和余弦函数
            fourier_bases = []

            # 直流分量 (DC component)
            dc_component = torch.ones_like(x[:, 0:1])
            fourier_bases.append(dc_component)

            # 生成不同频率的正弦和余弦对
            for j in range(1, self.hidden_dim // 2):
                # 2πj/N 中的N被freq_factor参数化
                freq_j = freq * j
                # 正弦项
                sin_term = torch.sin(freq_j * x)
                # 余弦项
                cos_term = torch.cos(freq_j * x)
                fourier_bases.append(sin_term)
                fourier_bases.append(cos_term)

            # 连接所有傅里叶基函数
            fourier_bases = torch.cat(fourier_bases, dim=1)

            # 确保维度正确
            if fourier_bases.size(1) > self.hidden_dim:
                fourier_bases = fourier_bases[:, :self.hidden_dim]
            elif fourier_bases.size(1) < self.hidden_dim:
                padding = torch.zeros(fourier_bases.size(0), self.hidden_dim - fourier_bases.size(1)).to(e.device)
                fourier_bases = torch.cat([fourier_bases, padding], dim=1)

            # 连接常数项和傅里叶基函数
            eeig = torch.cat((eeig, fourier_bases), dim=1)

            # 通过线性层
            out_e.append(self.eig_ws[i](eeig))

        return out_e

'''
class NonHarmonicEncoding(nn.Module):
    """
    非谐傅里叶级数版本（real form: sin/cos）
    - 可学习频率 λ_m，不要求为整数倍（非谐）。
    - Paley–Wiener: 用 tanh 将频率限制在 [-Omega, Omega]。
    - Bessel-type 上界:
        (i) 频率最小间隔分离正则，避免频率拥挤；
        (ii) 线性层权重 L2 正则 + 特征标准化，控制能量。
    兼容你的原接口：输入 e:[N]，输出为长度 k 的列表（每项形状 [N,1]）。
    """
    def __init__(self, k, hidden_dim=128, num_freq=None, Omega=50.0, delta_min=0.25, sep_penalty=1.0, weight_penalty=1e-4):
        """
        Args:
            k: 与原实现一致（外层循环次数）。
            hidden_dim: 线性层输入维度（= 1 + 2*num_freq）。建议为偶数。
            num_freq: 使用的非谐频率个数（正频），默认 hidden_dim//2
            Omega: 频率带宽上界（Paley–Wiener 约束）。
            delta_min: 最小频率间隔，促进分离（近似满足帧/基稳定性的必要条件之一）。
            sep_penalty: 频率间隔惩罚系数。
            weight_penalty: 线性层权重 L2 正则系数（Bessel型不等式的实践保障）。
        """
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim 应为偶数（由 1 + 2*num_freq 构成更方便）。"
        self.k = k
        self.Omega = float(Omega)
        self.delta_min = float(delta_min)
        self.sep_penalty = float(sep_penalty)
        self.weight_penalty = float(weight_penalty)

        # 使用的频率个数（正频）；最终用 ±λ_m 的 sin/cos 组合（等价地使用正频即可）
        self.num_freq = num_freq if num_freq is not None else hidden_dim // 2

        # 可学习原始参数，映射到 [-Omega, Omega]
        # 初始化为均匀分布在 [-Omega, Omega] 内的“近似等距”点，随后训练中变为“非谐”
        init_lin = torch.linspace(-0.9, 0.9, self.num_freq) * 0.8  # 在 (-1,1) 内较平滑的起点
        self.freq_raw = nn.Parameter(init_lin.clone())  # 将通过 tanh * Omega 映射

        # 线性层：每个 i 对应一个 readout（与原代码一致）
        in_dim = 1 + 2 * self.num_freq  # 常数项 + sin部分 + cos部分
        self.readouts = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(k)])

        # 预设一个常数偏置通道（与原实现的 eeig 常数1相同）
        self.register_buffer("one_bias", torch.tensor(1.0))

    def _bandlimited_freqs(self):
        # Paley–Wiener: 将原始参数压到 [-Omega, Omega]
        # 这里不强制排序，让“非谐”由训练自由演化；分离性用 penalty 约束
        return self.Omega * torch.tanh(self.freq_raw)

    def _separation_penalty(self, freqs):
        """
        对频率施加最小间隔约束：sum ReLU(delta_min - (f_{i+1} - f_i))^2
        说明：
        - 为了计算间隔，先排序，不改变 forward 表达（仅正则）。
        - 这鼓励频率“分开”，避免过密导致不稳定。
        """
        f_sorted, _ = torch.sort(freqs)
        diffs = f_sorted[1:] - f_sorted[:-1]
        gap_deficit = F.relu(self.delta_min - diffs)
        return (gap_deficit ** 2).sum()

    def _weight_penalty(self):
        """
        Bessel 上界的实践做法之一：限制线性读出层的能量（L2）。
        """
        reg = 0.0
        for lin in self.readouts:
            reg = reg + (lin.weight ** 2).sum()
        return reg * self.weight_penalty

    def regularization_loss(self):
        """
        将本模块建议的正则项（频率分离 + 权重L2）返回给外部，在总loss里加上即可。
        用法： total_loss = task_loss + enc.regularization_loss()
        """
        freqs = self._bandlimited_freqs()
        sep_reg = self.sep_penalty * self._separation_penalty(freqs)
        w_reg = self._weight_penalty()
        return sep_reg + w_reg

    def forward(self, e):
        """
        输入:
            e: [N] 张量
        输出:
            与原版一致：长度为 k 的列表，每个元素形状 [N, 1]
        """
        # [N, 1]
        e = e.view(-1)
        ee = e.unsqueeze(1)  # [N,1]

        # 取得带限后的非谐频率向量 [num_freq]
        freqs = self._bandlimited_freqs()  # ∈ [-Omega, Omega]

        # 归一化系数（Bessel友好）：让每个特征通道有可控幅度
        # 经验性处理：除以 sqrt(num_freq)
        norm_scale = (self.num_freq ** 0.5)

        outs = []
        # 与你的原结构对齐：对 i=1..k，使用 e^(i) 来“加权相位”，但频率是非谐 {λ_m}
        for i in range(self.k):
            ei = ee.pow(i + 1)  # [N,1]

            # 相位： [N, num_freq]
            phase = ei * freqs.unsqueeze(0)  # 广播

            # 特征： [N, 1 + 2*num_freq]
            # 常数通道 + sin/cos
            feat = torch.cat(
                [
                    torch.ones_like(ee) * self.one_bias,     # [N,1]
                    torch.sin(phase) / norm_scale,           # [N,num_freq]
                    torch.cos(phase) / norm_scale,           # [N,num_freq]
                ],
                dim=1,
            )

            outs.append(self.readouts[i](feat))  # [N,1]

        return outs
'''

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
        #print(q.size(), k.size(), v.size())

        x = k.matmul(v)

        if attn_bias is not None:
            x = x + attn_bias
        x = self.att_dropout(x)
        x = torch.matmul(q, x)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        #print(x.size())
        return x


class NoFoDifformer_FPlus(nn.Module):

    def __init__(self, nclass, nfeat, nlayer=1, hidden_dim=128, dim=32, nheads=1, k=10,
                 tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none'):
        super(NoFoDifformer_FPlus, self).__init__()

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


        ####################
        self.eig_encoder = ParamNonHarmonicEncoding(k, dim)

        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.prop_dropout = nn.Dropout(prop_dropout)

        self.k = k
        self.alpha = nn.Linear(self.k, 1, bias=False)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)

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

    def transformer_encoder(self, h, h_fur):
        mha_h = self.mha_norm(h)
        mha_h = self.mha(mha_h, mha_h, mha_h)
        mha_h_ = h + self.mha_dropout(mha_h) + h_fur

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
            h = self.transformer_encoder(h, h_encoder)

        if self.norm == 'none':
            return h
        else:
            h = self.feat_dp2(h)
            h = self.classify(h)
            return h
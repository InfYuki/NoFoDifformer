import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np
import pandas as pd
import scipy.sparse as sp

# ======================
# 1. 加载 Cora 数据集
# ======================
dataset = Planetoid(root='./', name='pubmed')
data = dataset[0]

# ======================
# 2. 构建无向邻接矩阵
# ======================
A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
A = A + sp.eye(A.shape[0])  # 加单位矩阵 (self-loop)
D = sp.diags(np.array(A.sum(1)).flatten())  # 度矩阵

# ======================
# 3. 计算归一化拉普拉斯矩阵 L = I - D^{-1/2} * A * D^{-1/2}
# ======================
D_inv_sqrt = sp.diags(np.power(np.array(A.sum(1)).flatten(), -0.5))
L = sp.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt

# ======================
# 4. 特征值分解（可根据节点数裁剪前N个）
# ======================
N = min(500, L.shape[0])  # 节点太多时取前500个
eigvals, eigvecs = np.linalg.eigh(L.toarray()[:N, :N])  # 对称矩阵 → eigh

# ======================
# 5. 计算谱能量 E(λ)
# ======================
# E = ∑ (Xᵀ * u)^2, 这里用节点特征矩阵 X
X = data.x.numpy()[:N, :]
E = np.sum((eigvecs.T @ X) ** 2, axis=1)

# ======================
# 6. 保存为 CSV 文件
# ======================
df = pd.DataFrame({
    'lambda': eigvals,
    'E': E
})
df.to_csv('Pubmed_spectrum.csv', index=False)

print("✅ 文件已生成: Pubmed_spectrum.csv")
print(df.head())

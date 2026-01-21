import argparse
import os
import torch
import pandas as pd
import numpy as np
from preprocess_node_data import load_data, eigen_decompositon

def load_eigs_from_cached(dataset):
    pt_path = os.path.join("../data", f"{dataset}.pt")
    if os.path.exists(pt_path):
        e, _, _, _ = torch.load(pt_path, map_location="cpu")
        return e.cpu().numpy()
    return None

def compute_eigs_from_raw(dataset):
    # 仅覆盖 cora/citeseer/pubmed 的简易路径，其他可按需要扩展
    if dataset in ["cora", "citeseer", "pubmed", "photo", "actor", "texas"]:
        adj, _, _ = load_data(dataset)
        adj = adj.todense()
        e, _ = eigen_decompositon(adj)
        return np.array(e, dtype=np.float32)
    raise ValueError(f"未实现数据集 {dataset} 的原始计算，请先生成 data/{dataset}.pt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="texas",
                        help="如 cora/citeseer/pubmed，对应 preprocess_node_data 生成的 data/{dataset}.pt")
    parser.add_argument("--output", type=str, default="./lambda/eigs.csv",
                        help="输出 CSV 路径（单列 eigenvalue）")
    args = parser.parse_args()

    eigs = load_eigs_from_cached(args.dataset)
    if eigs is None:
        eigs = compute_eigs_from_raw(args.dataset)

    df = pd.DataFrame({"eigenvalue": eigs})
    df.to_csv(args.output, index=False, float_format="%.8f")
    print(f"特征值已导出到 {args.output}，共 {len(df)} 行。")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import torch
from utils import seed_everything, init_params
from Models.model_DTGFKAN import NoFoDifformer

def extract_learnable_frequencies(model):
    """
    从模型中提取学习得到的频率
    需要确保模型中存在 eig_encoder.learnable_frequencies 张量
    """
    freq_list = []
    # 遍历模型层
    for name, module in model.named_modules():
        if hasattr(module, "learnable_frequencies"):
            freqs = module.learnable_frequencies.detach().cpu().numpy().flatten()
            freq_list.extend(freqs)
    return np.array(freq_list)


def generate_frequency_csv(args):
    SEEDS = np.arange(1, args.runs + 1)
    all_records = []

    for rp, seed in enumerate(SEEDS, start=1):
        print(f"▶ Running {rp}/{args.runs} with seed {seed}")
        seed_everything(seed)

        # 初始化模型
        device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
        model = NoFoDifformer(
            nclass=7,                 # 可根据你的数据集调整
            nfeat=1433,
            nlayer=args.nlayer,
            hidden_dim=args.hidden_dim,
            dim=args.dim,
            nheads=args.nheads,
            k=args.k,
            num_layers=args.num_layers,
            num_freq=args.num_freq,
            Omega=args.Omega,
            delta_min=args.delta_min,
            residual=True,
            weight_penalty=args.weight_penalty,
            tran_dropout=args.tran_dropout,
            feat_dropout=args.feat_dropout,
            prop_dropout=args.prop_dropout,
            norm=args.norm
        ).to(device)

        # 初始化参数（与原代码一致）
        model.apply(init_params)

        # ===== 提取频率 =====
        learned_freq = extract_learnable_frequencies(model)
        num_freq = len(learned_freq)
        fundamental = np.arange(1, num_freq + 1)
        diff = learned_freq - fundamental

        df = pd.DataFrame({
            "run": rp,
            "freq_index": np.arange(1, num_freq + 1),
            "fundamental_frequency": fundamental,
            "learnable_frequency": learned_freq,
            "diff": diff
        })
        all_records.append(df)

    # 合并所有 run 的结果
    df_all = pd.concat(all_records, ignore_index=True)
    df_all.to_csv("learned_frequencies.csv", index=False)
    print("✅ 已保存文件: learned_frequencies.csv")


# ============ 模拟 argparse 参数 ============
class Args:
    cuda = 0
    runs = 10
    nlayer = 2
    hidden_dim = 128
    dim = 16
    nheads = 2
    k = 3
    num_layers = 1
    num_freq = 16
    Omega = 50.0
    delta_min = 0.25
    weight_penalty = 1e-4
    tran_dropout = 0.4
    feat_dropout = 0.3
    prop_dropout = 0.0
    norm = "none"

if __name__ == "__main__":
    args = Args()
    generate_frequency_csv(args)

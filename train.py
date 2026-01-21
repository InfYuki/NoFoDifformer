import time
import yaml
import copy
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score, r2_score

from model import NoFoDifformer
from model_org import NoFoDifformer_org
from model_withDiffAttn import NoFoDifformer_Diff
from model_mk import NoFoDifformer_mk
from model_CL import NoFoDifformer_CL
from model_kan import NoFoDifformer_MultiLayerKAN
from model_DTGFKAN import NoFoDifformer
from model_temp import NoFoDifformer_temp

from utils import count_parameters, init_params, seed_everything, get_split
from tqdm import tqdm
import scipy.stats  # 添加这一行导入

def get_graph_fourier_basis_from_edges(e, num_nodes):
    """
    e: [2, E] 张量，边的索引
    num_nodes: 节点数 N
    return: U (傅里叶基), Lambda (特征值)
    """
    # 构建邻接矩阵 A
    A = torch.zeros((num_nodes, num_nodes))
    for i, j in e.t():
        A[i, j] = 1
        A[j, i] = 1  # 无向图

    D = torch.diag(A.sum(dim=1))
    # 对称归一化拉普拉斯
    D_inv_sqrt = torch.linalg.inv(torch.sqrt(D))
    L = torch.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt

    # 特征分解
    Lambda, U = torch.linalg.eigh(L)
    return U, Lambda



def main_worker(args):
    #print(args)
    seed_everything(args.seed)
    device = 'cuda:{}'.format(args.cuda)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.cuda)

    
    e, u, x, y = torch.load('data/{}.pt'.format(args.dataset))

    #print("u shape:", u.shape)
    #print("u min/max:", u.min().item(), u.max().item())
    #print("u diag:", torch.diag(u)[:10])
    #print("e shape:", e.shape)

    if len(y.size()) > 1:
        if y.size(1) > 1:   
            y = torch.argmax(y, dim=1)
        else:
            y = y.view(-1)
           
    else:
        y = y.view(-1)
    nclass = int(y.max().item()) + 1
            
    e, u, x, y = e.to(device), u.to(device), x.to(device), y.to(device)

    train, valid, test = get_split(args.dataset, y, nclass, args.seed) 
    train, valid, test = map(torch.LongTensor, (train, valid, test))
    train, valid, test = train.to(device), valid.to(device), test.to(device)

    nfeat = x.size(1)

    #net = NoFoDifformer_org(nclass, nfeat, args.nlayer, args.hidden_dim, args.dim, args.nheads, args.k,
    #                                    args.tran_dropout, args.feat_dropout, args.prop_dropout, args.norm).to(device)

    '''
    # 使用新的NoFoDifformer_MultiLayerKAN模型
    net = NoFoDifformer_MultiLayerKAN(
        nclass, nfeat,
        nlayer=args.nlayer,
        hidden_dim=args.hidden_dim,
        dim=args.dim,
        nheads=args.nheads,
        k=args.k,
        kan_layers=args.kan_layers,
        kan_grid_num=args.kan_grid_num,
        kan_k=args.kan_k,
        kan_noise_scale=args.kan_noise_scale,
        tran_dropout=args.tran_dropout,
        feat_dropout=args.feat_dropout,
        prop_dropout=args.prop_dropout,
        norm=args.norm
    ).to(device)
    '''
    net = NoFoDifformer(
        nclass, nfeat,
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


    # 初始化模型，添加温度和对比学习权重参数
    #net = NoFoDifformer_CL(nclass, nfeat, args.nlayer, args.hidden_dim, args.dim, args.nheads, args.k,
    #                    args.tran_dropout, args.feat_dropout, args.prop_dropout, args.norm,
    #                    temperature=args.temperature, cl_weight=args.cl_weight).to(device)
    net.apply(init_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    res = []
    min_loss = 100.0

    max_acc1 = 0
    max_acc2 = 0
    counter = 0
    best_val_acc = 0
    best_test_acc = 0
    
    time_run=[]
    for idx in range(args.epoch):
        t_st=time.time()
        net.train()
        optimizer.zero_grad()
        logits = net(e, u, x)
        loss = F.cross_entropy(logits[train], y[train])



        # 使用新的模型前向传播，传入标签进行监督学习
        #logits, total_loss, sup_loss, cl_loss = net(e, u, x, y=y[train], train_idx=train)

        # 注意：这里不需要再手动计算损失，因为模型已经返回了总损失
        #loss = total_loss

        loss.backward()
        optimizer.step()

        time_epoch = time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)

        net.eval()
        logits = net(e, u, x)
        # 评估时不需要计算对比损失
        #with torch.no_grad():
        #    logits, _ = net(e, u, x)


        evaluation = torchmetrics.Accuracy()
        val_loss = F.cross_entropy(logits[valid], y[valid]).item()
        val_acc = evaluation(logits[valid].cpu(), y[valid].cpu()).item()
        test_acc = evaluation(logits[test].cpu(), y[test].cpu()).item()
        res.append([val_loss, val_acc, test_acc])
        
        if val_loss < min_loss:
            min_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc
            counter = 0
                
        else:
            counter += 1

        # 每200个epoch输出频率序列
        #if idx % 200 == 0 :
            #print(f"\n{'=' * 50}")
            #print(f"Epoch {idx}: 输出频率序列")
            #print(f"{'=' * 50}")
            #if hasattr(net, 'eig_encoder') and hasattr(net.eig_encoder, 'get_and_print_frequencies'):
            #    net.eig_encoder.get_and_print_frequencies(idx)
            #print(f"{'=' * 50}\n")

        if counter == args.patience:
            max_acc1 = sorted(res, key=lambda x: x[0], reverse=False)[0][-1]
            max_acc2 = sorted(res, key=lambda x: x[1], reverse=True)[0][-1]
            #print(max_acc1, max_acc2)
            print(f"Run结束于第{idx}个epoch，最佳测试准确率: {max_acc1}, {max_acc2}")
            break

        
    return max_acc1,max_acc2, time_run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--runs', type=int, default=10) #5 for penn
    parser.add_argument('--dataset', default='pubmed')
    # cora  citeseer  pubmed
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--nheads', type=int, default=2)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--nlayer', type=int, default=3)
    parser.add_argument('--tran_dropout', type=float, default=0.4, help='dropout for neural networks.')
    parser.add_argument('--feat_dropout', type=float, default=0.3, help='dropout for neural networks.')
    parser.add_argument('--prop_dropout', type=float, default=0.0, help='dropout for neural networks.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay.')
    parser.add_argument('--norm', default='none')
    parser.add_argument('--patience', type=int, default=300, help='early stopping patience')
    # 添加对比学习相关参数
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature for contrastive learning')
    parser.add_argument('--cl_weight', type=float, default=0.1, help='weight for contrastive loss')

    # 在ArgumentParser部分添加新的参数（约在151-173行之间）

    # 在parser.add_argument('--patience', ...)之后添加以下代码
    #parser.add_argument('--kan_in_dim', type=int, default=1, help='KANLayer输入维度')
    #parser.add_argument('--kan_out_dim', type=int, default=1, help='KANLayer输出维度')
    #parser.add_argument('--kan_grid_num', type=int, default=5, help='KANLayer网格数量')
    #parser.add_argument('--kan_k', type=int, default=3, help='KANLayer样条阶数')
    #parser.add_argument('--fourier_basis_num', type=int, default=3, help='傅里叶基函数的数量')
    #parser.add_argument('--fourier_omega', type=float, default=3.14159, help='傅里叶基函数的频率范围')
    # 在ArgumentParser部分添加新的参数
    #parser.add_argument('--kan_layers', type=int, default=1, help='KAN层数')
    #parser.add_argument('--kan_grid_num', type=int, default=1, help='KAN网格数量')
    #parser.add_argument('--kan_k', type=int, default=3, help='KAN样条阶数')
    #parser.add_argument('--kan_noise_scale', type=float, default=0.1, help='KAN噪声缩放')

    # 在ArgumentParser部分添加新参数
    parser.add_argument('--num_layers', type=int, default=1,
                        help='非谐波傅里叶KAN的层数 (=1时等同于model_mk)')
    parser.add_argument('--num_freq', type=int, default=8,
                        help='每层使用的频率数量')
    parser.add_argument('--Omega', type=float, default=50.0,
                        help='最大频率约束')
    parser.add_argument('--delta_min', type=float, default=0.25,
                        help='频率最小间隔')
    #parser.add_argument('--no_residual', action='store_false', dest='residual',
    #                    help='不使用残差连接')
    parser.add_argument('--weight_penalty', type=float, default=1e-4,
                        help='')

    args = parser.parse_args()

    '''
    python train.py --dataset cora --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 2 --hidden_dim 128 --dim 16 --nheads 2 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.6
    python train.py --dataset citeseer --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 3 --hidden_dim 128 --dim 16 --nheads 1 --tran_dropout 0.5 --feat_dropout 0.5 --prop_dropout 0.3
    python train.py --dataset pubmed --lr 0.01 --weight_decay 5e-4 --nlayer 2 --k 2 --hidden_dim 128 --dim 16 --nheads 1 --tran_dropout 0.4 --feat_dropout 0.3 --prop_dropout 0.0
    python train.py --dataset photo --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 2 --hidden_dim 128 --dim 16 --nheads 2 --tran_dropout 0.4 --feat_dropout 0.4 --prop_dropout 0.4
    python train.py --dataset physics --lr 0.01 --weight_decay 5e-3 --nlayer 1 --k 2 --hidden_dim 128 --dim 16 --nheads 1 --tran_dropout 0.3 --feat_dropout 0.3 --prop_dropout 0.7
    python train.py --dataset wikics --lr 0.005 --weight_decay 5e-4 --dim 16 --hidden_dim 128 --k 2 --nlayer 1 --nheads 1 --tran_dropout 0.6 --feat_dropout 0.4 --prop_dropout 0.1
    python train.py --dataset penn --runs 5 --lr 0.01 --weight_decay 5e-5 --nlayer 1 --k 10 --hidden_dim 128 --dim 16 --nheads 1 --tran_dropout 0.5 --feat_dropout 0.2 --prop_dropout 0.0
    python train.py --dataset chameleon --lr 0.01 --weight_decay 5e-5 --nlayer 2 --k 5 --hidden_dim 128 --dim 32 --nheads 3 --tran_dropout 0.6 --feat_dropout 0.4 --prop_dropout 0.3
    python train.py --dataset squirrel --lr 0.01 --weight_decay 5e-5 --nlayer 1 --k 6 --hidden_dim 128 --dim 32 --nheads 1 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.0
    python train.py --dataset actor --lr 0.01 --weight_decay 5e-5 --nlayer 2 --k 2 --hidden_dim 128 --dim 32 --nheads 2 --tran_dropout 0.0 --feat_dropout 0.3 --prop_dropout 0.5
    python train.py --dataset texas --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 1 --hidden_dim 128 --dim 16 --nheads 3 --tran_dropout 0.6 --feat_dropout 0.6 --prop_dropout 0.8
    #python train.py --dataset texas --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 1 --hidden_dim 128 --dim 16 --nheads 2 --tran_dropout 0.5 --feat_dropout 0.8 --prop_dropout 0.8
    
    #model_withDiffAttn          Cora best         0.892
    python train.py --dataset cora --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 1 --hidden_dim 128 --dim 32 --nheads 1 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.6
    
    #model_withDiffAttn          citeseer best         0.802
    python train.py --dataset citeseer --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 2 --hidden_dim 128 --dim 32 --nheads 1 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.6
    
    #model_mk
    python train.py --dataset pubmed --lr 0.01 --weight_decay 5e-4 --nlayer 3 --k 3 --hidden_dim 128 --dim 16 --nheads 2 --tran_dropout 0.4 --feat_dropout 0.3 --prop_dropout 0.0
    
    '''

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
 
    results=[]
    time_results=[]
    SEEDS = np.arange(1, 11)
    for RP in tqdm(range(args.runs)):
        args.seed = SEEDS[RP]
        set_seed(args.seed)
        best_val_acc, best_test_acc, time_run = main_worker(args)
        results.append(best_test_acc)
        time_results.append(time_run)
        #print(f"Run {RP + 1}/{args.runs} 完成，使用了 {len(time_run)} 个epochs")

    run_sum=0
    epochsss=0
    for i in time_results:
        run_sum+=sum(i)
        epochsss+=len(i)
    print("each run avg_time:",run_sum/(args.runs),"s")
    print("each epoch avg_time:",1000*run_sum/epochsss,"ms")  
    #print(np.mean(results))

    mean_acc = np.mean(results)
    std_acc = np.std(results, ddof=1)  # 使用无偏估计
    n = len(results)
    # 计算95%置信区间
    confidence_level = 0.95
    degrees_freedom = n - 1
    t_critical = scipy.stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    margin_of_error = t_critical * (std_acc / np.sqrt(n))
    lower_bound = mean_acc - margin_of_error
    upper_bound = mean_acc + margin_of_error

    print(f"平均准确率: {mean_acc}")
    print(f"95%置信区间: [{lower_bound:.4f}, {upper_bound:.4f}]")
    print(f"置信区间宽度: {2*margin_of_error:.4f}")
    print(f"标准差: {std_acc:.4f}")

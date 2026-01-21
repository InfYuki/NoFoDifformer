# NoFoDifformer

### NoFoDifformer: A Graph Transformer Integrating Nonharmonic Fourier Filtering and Differential Attention

![image](M_2_4_5.jpg)

## Environment Settings
This implementation is based on Python3. To run the code, you need the following dependencies:

- torch==1.8.1+cu111
- torch-geometric==1.7.2
- scipy==1.13.1
- numpy==1.23.0
- tqdm==4.59.0
- seaborn==0.11.2
- scikit-learn==0.24.2
- dgl==0.6.1
- pandas==2.3.1
- googledrivedownloader==1.1.0

Detailed environment configuration is in the [environment.yml](environment.yml) file.

## Usage

### Run node classification experiment:

    python train.py --seed 42 --cuda 0 --runs 10 --dataset pubmed --epoch 2000 --k 3 --nheads 2 --dim 16 --hidden_dim 128 --nlayer 3 --tran_dropout 0.4 --feat_dropout 0.3 --prop_dropout 0.0 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 1 --num_freq 8 --Omega 50.0 --delta_min 0.25 --weight_penalty 1e-4


All instructions and parameters are in the [run.sh](run.sh) 

## Development - 开发（关于怎样开发的文档信息。（API 等。））

## Changelog - 更新日志（一个简短的历史记录（更改，替换或者其他）。）

## FAQ - 常见问题（常见问题。）

## Support - 支持

### Dos - 文档（更多文档。）

### Contact - 联系（其他联系信息（电子邮件地址，网站，公司名称，地址等）。提交bug，功能要求，提交补丁，加入邮件列表，得到通知，或加入用户或开发开发区群的介绍。）

## Authors and acknowledgment - 贡献者和感谢（作者列表和鸣谢。）

## License - 版权信息（版权和许可信息（或阅读许可证）、法律声明。）

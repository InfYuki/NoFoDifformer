python train.py --dataset cora --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 2 --hidden_dim 128 --dim 16 --nheads 2 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.6
python train.py --dataset citeseer --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 3 --hidden_dim 128 --dim 16 --nheads 1 --tran_dropout 0.5 --feat_dropout 0.5 --prop_dropout 0.3
python train_org.py --dataset pubmed --lr 0.01 --weight_decay 5e-4 --nlayer 2 --k 2 --hidden_dim 128 --dim 16 --nheads 1 --tran_dropout 0.4 --feat_dropout 0.3 --prop_dropout 0.0

#平均准确率: 0.949848186969757  标准差: 0.0054
python train_org.py --dataset photo --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 2 --hidden_dim 128 --dim 16 --nheads 2 --tran_dropout 0.4 --feat_dropout 0.4 --prop_dropout 0.4

python train_org.py --dataset physics --lr 0.01 --weight_decay 5e-3 --nlayer 1 --k 2 --hidden_dim 128 --dim 16 --nheads 1 --tran_dropout 0.3 --feat_dropout 0.3 --prop_dropout 0.7

python train.py --dataset wikics --lr 0.005 --weight_decay 5e-4 --dim 16 --hidden_dim 128 --k 2 --nlayer 1 --nheads 1 --tran_dropout 0.6 --feat_dropout 0.4 --prop_dropout 0.1
python train.py --dataset penn --runs 5 --lr 0.01 --weight_decay 5e-5 --nlayer 1 --k 10 --hidden_dim 128 --dim 16 --nheads 1 --tran_dropout 0.5 --feat_dropout 0.2 --prop_dropout 0.0
python train.py --dataset chameleon --lr 0.01 --weight_decay 5e-5 --nlayer 2 --k 5 --hidden_dim 128 --dim 32 --nheads 3 --tran_dropout 0.6 --feat_dropout 0.4 --prop_dropout 0.3
python train.py --dataset squirrel --lr 0.01 --weight_decay 5e-5 --nlayer 1 --k 6 --hidden_dim 128 --dim 32 --nheads 1 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.0

python train_org.py --dataset actor --lr 0.01 --weight_decay 5e-5 --nlayer 2 --k 2 --hidden_dim 128 --dim 32 --nheads 2 --tran_dropout 0.0 --feat_dropout 0.3 --prop_dropout 0.5

python train.py --dataset texas --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 1 --hidden_dim 128 --dim 16 --nheads 3 --tran_dropout 0.6 --feat_dropout 0.6 --prop_dropout 0.8

#python train.py --dataset texas --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 1 --hidden_dim 128 --dim 16 --nheads 2 --tran_dropout 0.5 --feat_dropout 0.8 --prop_dropout 0.8


#model_withDiffAttn          Cora best         0.892
python train.py --dataset cora --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 1 --hidden_dim 128 --dim 32 --nheads 1 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.6

python train_org.py --dataset cora --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 1 --hidden_dim 128 --dim 16 --nheads 1 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.6


#model_withDiffAttn          citeseer best         0.802
python train.py --dataset citeseer --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 2 --hidden_dim 128 --dim 32 --nheads 1 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.6

python train_org.py --dataset citeseer --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 3 --hidden_dim 128 --dim 16 --nheads 1 --tran_dropout 0.5 --feat_dropout 0.5 --prop_dropout 0.6


#new set
python train.py --seed 42 --cuda 0 --runs 10 --dataset pubmed --epoch 2000 --k 3 --nheads 2 --dim 16 --hidden_dim 128 --nlayer 3 --tran_dropout 0.4 --feat_dropout 0.3 --prop_dropout 0.0 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 1 --num_freq 8 --Omega 50.0 --delta_min 0.25 --weight_penalty 1e-4

python train.py --seed 42 --cuda 0 --runs 10 --dataset pubmed --epoch 2000 --k 3 --nheads 2 --dim 16 --hidden_dim 128 --nlayer 3 --tran_dropout 0.4 --feat_dropout 0.3 --prop_dropout 0.0 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 1 --num_freq 8 --Omega 50.0 --delta_min 0.25 --weight_penalty 1e-4

#0.8949
python train.py --seed 42 --cuda 0 --runs 10 --dataset cora --epoch 2000 --k 1 --nheads 1 --dim 32 --hidden_dim 128 --nlayer 1 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.6 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 2 --num_freq 16 --Omega 45.0 --delta_min 0.25 --weight_penalty 1e-4

python train.py --seed 42 --cuda 0 --runs 10 --dataset cora --epoch 2000 --k 1 --nheads 1 --dim 32 --hidden_dim 128 --nlayer 1 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.6 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 3 --num_freq 16 --Omega 50.0 --delta_min 0.25 --weight_penalty 1e-4

##############
python train.py --seed 42 --cuda 0 --runs 10 --dataset cora --epoch 2000 --k 1 --nheads 1 --dim 32 --hidden_dim 128 --nlayer 1 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.6 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 2 --num_freq 16 --Omega 16.0 --delta_min 0.25 --weight_penalty 1e-4



#python train.py --dataset cora --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 2 --hidden_dim 128 --dim 16 --nheads 2 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.6

#####   citeseer

python train.py --seed 42 --cuda 0 --runs 10 --dataset citeseer --epoch 2000 --k 2 --nheads 2 --dim 32 --hidden_dim 128 --nlayer 1 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.6 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 2 --num_freq 16 --Omega 50.0 --delta_min 0.25 --weight_penalty 1e-4

python train.py --seed 42 --cuda 0 --runs 10 --dataset citeseer --epoch 2000 --k 3 --nheads 1 --dim 16 --hidden_dim 128 --nlayer 1 --tran_dropout 0.5 --feat_dropout 0.5 --prop_dropout 0.3 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 2 --num_freq 16 --Omega 50.0 --delta_min 0.25 --weight_penalty 1e-4


python train.py --dataset citeseer --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 3 --hidden_dim 128 --dim 16 --nheads 1 --tran_dropout 0.5 --feat_dropout 0.5 --prop_dropout 0.3

#####   photo
python train_org.py --dataset photo --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 2 --hidden_dim 128 --dim 16 --nheads 2 --tran_dropout 0.4 --feat_dropout 0.4 --prop_dropout 0.4
#平均准确率: 0.9487348198890686   标准差: 0.0050
python train.py --seed 42 --cuda 0 --runs 10 --dataset photo --epoch 2000 --k 3 --nheads 2 --dim 16 --hidden_dim 128 --nlayer 3 --tran_dropout 0.4 --feat_dropout 0.3 --prop_dropout 0.0 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 1 --num_freq 8 --Omega 50.0 --delta_min 0.25 --weight_penalty 1e-4
#平均准确率: 0.9518218636512756  标准差: 0.0066
python train.py --seed 42 --cuda 0 --runs 10 --dataset photo --epoch 2000 --k 2 --nheads 2 --dim 16 --hidden_dim 128 --nlayer 1 --tran_dropout 0.4 --feat_dropout 0.4 --prop_dropout 0.4 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 1 --num_freq 16 --Omega 50.0 --delta_min 0.25 --weight_penalty 1e-4

python train.py --seed 42 --cuda 0 --runs 10 --dataset photo --epoch 2000 --k 2 --nheads 2 --dim 16 --hidden_dim 128 --nlayer 1 --tran_dropout 0.4 --feat_dropout 0.4 --prop_dropout 0.4 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 1 --num_freq 16 --Omega 50.0 --delta_min 0.25 --weight_penalty 1e-4


#####  actor
#平均准确率: 0.4217859387397766  标准差: 0.0167
python train.py --seed 42 --cuda 0 --runs 10 --dataset actor --epoch 2000 --k 2 --nheads 2 --dim 32 --hidden_dim 128 --nlayer 2 --tran_dropout 0.0 --feat_dropout 0.3 --prop_dropout 0.5 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 2 --num_freq 12 --Omega 40.0 --delta_min 0.25 --weight_penalty 1e-4

python train.py --seed 42 --cuda 0 --runs 10 --dataset actor --epoch 2000 --k 2 --nheads 2 --dim 16 --hidden_dim 128 --nlayer 2 --tran_dropout 0.0 --feat_dropout 0.3 --prop_dropout 0.5 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 1 --num_freq 12 --Omega 40.0 --delta_min 0.25 --weight_penalty 1e-4


python train_org.py --dataset actor --lr 0.01 --weight_decay 5e-5 --nlayer 2 --k 2 --hidden_dim 128 --dim 32 --nheads 2 --tran_dropout 0.0 --feat_dropout 0.3 --prop_dropout 0.5


#####  texas
#平均准确率: 0.9426229476928711   标准差: 0.0208
python train.py --seed 42 --cuda 0 --runs 10 --dataset texas --epoch 2000 --k 1 --nheads 2 --dim 32 --hidden_dim 128 --nlayer 1 --tran_dropout 0.5 --feat_dropout 0.8 --prop_dropout 0.8 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 2 --num_freq 12 --Omega 50.0 --delta_min 0.25 --weight_penalty 1e-4

python train.py --seed 42 --cuda 0 --runs 10 --dataset texas --epoch 2000 --k 1 --nheads 2 --dim 32 --hidden_dim 128 --nlayer 1 --tran_dropout 0.5 --feat_dropout 0.8 --prop_dropout 0.8 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 2 --num_freq 12 --Omega 50.0 --delta_min 0.25 --weight_penalty 1e-4


python train_org.py --dataset texas --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 1 --hidden_dim 128 --dim 16 --nheads 3 --tran_dropout 0.6 --feat_dropout 0.6 --prop_dropout 0.8

python train_org.py --dataset texas --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 1 --hidden_dim 128 --dim 16 --nheads 2 --tran_dropout 0.5 --feat_dropout 0.8 --prop_dropout 0.8
#平均准确率: 0.9360655725002289  标准差: 0.0313



##########        AS              ##################
python train_AS.py --seed 42 --cuda 0 --runs 10 --dataset cora --epoch 2000 --k 1 --nheads 1 --dim 32 --hidden_dim 128 --nlayer 1 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.6 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 2 --num_freq 16 --Omega 45.0 --delta_min 0.25 --weight_penalty 1e-4

python train_AS.py --seed 42 --cuda 0 --runs 10 --dataset pubmed --epoch 2000 --k 3 --nheads 2 --dim 16 --hidden_dim 128 --nlayer 3 --tran_dropout 0.4 --feat_dropout 0.3 --prop_dropout 0.0 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 1 --num_freq 8 --Omega 50.0 --delta_min 0.25 --weight_penalty 1e-4

python train_AS.py --seed 42 --cuda 0 --runs 10 --dataset pubmed --epoch 2000 --k 3 --nheads 2 --dim 16 --hidden_dim 128 --nlayer 1 --tran_dropout 0.4 --feat_dropout 0.3 --prop_dropout 0.0 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 1 --num_freq 8 --Omega 50.0 --delta_min 0.25 --weight_penalty 1e-4



python train_AS.py --seed 42 --cuda 0 --runs 10 --dataset texas --epoch 2000 --k 1 --nheads 2 --dim 32 --hidden_dim 128 --nlayer 1 --tran_dropout 0.5 --feat_dropout 0.8 --prop_dropout 0.8 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 2 --num_freq 12 --Omega 50.0 --delta_min 0.25 --weight_penalty 1e-4


###################  vis   ###########################
python train_vis.py --seed 42 --cuda 0 --runs 2 --dataset cora --epoch 2000 --k 1 --nheads 1 --dim 32 --hidden_dim 128 --nlayer 1 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.6 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 2 --num_freq 16 --Omega 45.0 --delta_min 0.25 --weight_penalty 1e-4

python train_vis.py --seed 42 --cuda 0 --runs 2 --dataset citeseer --epoch 2000 --k 2 --nheads 2 --dim 32 --hidden_dim 128 --nlayer 1 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.6 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 2 --num_freq 16 --Omega 50.0 --delta_min 0.25 --weight_penalty 1e-4

python train_vis.py --seed 42 --cuda 0 --runs 2 --dataset pubmed --epoch 2000 --k 3 --nheads 2 --dim 16 --hidden_dim 128 --nlayer 3 --tran_dropout 0.4 --feat_dropout 0.3 --prop_dropout 0.0 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 1 --num_freq 8 --Omega 50.0 --delta_min 0.25 --weight_penalty 1e-4

python train_vis.py --seed 42 --cuda 0 --runs 2 --dataset photo --epoch 2000 --k 3 --nheads 2 --dim 16 --hidden_dim 128 --nlayer 3 --tran_dropout 0.4 --feat_dropout 0.3 --prop_dropout 0.0 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 1 --num_freq 8 --Omega 50.0 --delta_min 0.25 --weight_penalty 1e-4

python train_vis.py --seed 42 --cuda 0 --runs 2 --dataset actor --epoch 2000 --k 2 --nheads 2 --dim 32 --hidden_dim 128 --nlayer 2 --tran_dropout 0.0 --feat_dropout 0.3 --prop_dropout 0.5 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 2 --num_freq 12 --Omega 40.0 --delta_min 0.25 --weight_penalty 1e-4

python train_vis.py --seed 42 --cuda 0 --runs 2 --dataset texas --epoch 2000 --k 1 --nheads 2 --dim 32 --hidden_dim 128 --nlayer 1 --tran_dropout 0.5 --feat_dropout 0.8 --prop_dropout 0.8 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 2 --num_freq 12 --Omega 50.0 --delta_min 0.25 --weight_penalty 1e-4









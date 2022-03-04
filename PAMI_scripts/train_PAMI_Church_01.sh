#!/bin/bash
#SBATCH -o ../PAMI_logs/220225_IDEAS_Church_N=2_lambdaREC=10.out ## 作业输出的日志文件
#SBATCH -J C2_10 ## 作业名
#SBATCH -N 1 ## 申请1个节点
#SBATCH --gres=gpu:1 ## 申请1个GPU
#SBATCH --ntasks-per-node=4 ## 每节点运行10个任务（每节点调用10核）
#SBATCH --nodelist=node07 ## 申请节点node06


nvidia-smi

cd ../

python train_PAMI.py \
--exp_name 'PAMI_Church_N=2_lambdaREC=10' \
--N 2 \
--lambda_REC 10 \
--dataset_path ../dataset/Church/church_outdoor_train_lmdb \
--dataset_type offical_lmdb \
--batch_size 1
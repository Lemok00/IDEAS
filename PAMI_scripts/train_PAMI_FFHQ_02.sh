#!/bin/bash
#SBATCH -o ../PAMI_logs/220305_PAMI02_FFHQ_N=1_lambdaREC=10.out ## 作业输出的日志文件
#SBATCH -J F1_10 ## 作业名
#SBATCH -N 1 ## 申请1个节点
#SBATCH --gres=gpu:1 ## 申请1个GPU
#SBATCH --ntasks-per-node=4 ## 每节点运行10个任务（每节点调用10核）
#SBATCH --nodelist=node06 ## 申请节点node06


nvidia-smi

cd ../

python train_PAMI_02.py \
--exp_name 'PAMI_02_FFHQ_N=1_lambdaREC=10' \
--N 1 \
--lambda_REC 10 \
--dataset_path ../dataset/FFHQ/prepared_train_256 \
--dataset_type resized_lmdb \
--batch_size 1
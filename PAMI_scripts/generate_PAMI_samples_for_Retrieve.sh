#!/bin/bash
#SBATCH -o ../PAMI_log_results/220225_generate_PAMI_samples_for_Retrieve.out ## 作业输出的日志文件
#SBATCH -J Retrieve ## 作业名
#SBATCH -N 1 ## 申请1个节点
#SBATCH --gres=gpu:1 ## 申请1个GPU
#SBATCH --ntasks-per-node=4 ## 每节点运行10个任务（每节点调用10核）
#SBATCH --nodelist=node06 ## 申请节点node06


nvidia-smi

cd ../

python generate_PAMI_samples_forRetrieve.py --exp_name 'PAMI_Bedroom_N=1_lambdaREC=10' --ckpt 900000 --sigma 1 --delta 0.5

python generate_PAMI_samples_forRetrieve.py --exp_name 'PAMI_Church_N=1_lambdaREC=10' --ckpt 900000 --sigma 1 --delta 0.5

python generate_PAMI_samples_forRetrieve.py --exp_name 'PAMI_FFHQ_N=1_lambdaREC=10' --ckpt 900000 --sigma 1 --delta 0.5

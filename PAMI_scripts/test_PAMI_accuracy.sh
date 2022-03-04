#!/bin/bash
#SBATCH -o ../PAMI_log_results/220224_test_PAMI_accuracy.out ## 作业输出的日志文件
#SBATCH -J ACC ## 作业名
#SBATCH -N 1 ## 申请1个节点
#SBATCH --gres=gpu:1 ## 申请1个GPU
#SBATCH --ntasks-per-node=4 ## 每节点运行10个任务（每节点调用10核）
#SBATCH --nodelist=node06 ## 申请节点node06


nvidia-smi

cd ../

python test_PAMI_accuracy.py --exp_name 'PAMI_Bedroom_N=1_lambdaREC=10' --ckpt 900000 --sigma 1 --delta 0
python test_PAMI_accuracy.py --exp_name 'PAMI_Bedroom_N=1_lambdaREC=10' --ckpt 900000 --sigma 1 --delta 0.25
python test_PAMI_accuracy.py --exp_name 'PAMI_Bedroom_N=1_lambdaREC=10' --ckpt 900000 --sigma 1 --delta 0.5

python test_PAMI_accuracy.py --exp_name 'PAMI_Bedroom_N=1_lambdaREC=10' --ckpt 900000 --sigma 2 --delta 0
python test_PAMI_accuracy.py --exp_name 'PAMI_Bedroom_N=1_lambdaREC=10' --ckpt 900000 --sigma 2 --delta 0.25
python test_PAMI_accuracy.py --exp_name 'PAMI_Bedroom_N=1_lambdaREC=10' --ckpt 900000 --sigma 2 --delta 0.5

python test_PAMI_accuracy.py --exp_name 'PAMI_Bedroom_N=1_lambdaREC=10' --ckpt 900000 --sigma 3 --delta 0
python test_PAMI_accuracy.py --exp_name 'PAMI_Bedroom_N=1_lambdaREC=10' --ckpt 900000 --sigma 3 --delta 0.25
python test_PAMI_accuracy.py --exp_name 'PAMI_Bedroom_N=1_lambdaREC=10' --ckpt 900000 --sigma 3 --delta 0.5

python test_PAMI_accuracy.py --exp_name 'PAMI_Church_N=1_lambdaREC=10' --ckpt 900000 --sigma 1 --delta 0
python test_PAMI_accuracy.py --exp_name 'PAMI_Church_N=1_lambdaREC=10' --ckpt 900000 --sigma 1 --delta 0.25
python test_PAMI_accuracy.py --exp_name 'PAMI_Church_N=1_lambdaREC=10' --ckpt 900000 --sigma 1 --delta 0.5

python test_PAMI_accuracy.py --exp_name 'PAMI_Church_N=1_lambdaREC=10' --ckpt 900000 --sigma 2 --delta 0
python test_PAMI_accuracy.py --exp_name 'PAMI_Church_N=1_lambdaREC=10' --ckpt 900000 --sigma 2 --delta 0.25
python test_PAMI_accuracy.py --exp_name 'PAMI_Church_N=1_lambdaREC=10' --ckpt 900000 --sigma 2 --delta 0.5

python test_PAMI_accuracy.py --exp_name 'PAMI_Church_N=1_lambdaREC=10' --ckpt 900000 --sigma 3 --delta 0
python test_PAMI_accuracy.py --exp_name 'PAMI_Church_N=1_lambdaREC=10' --ckpt 900000 --sigma 3 --delta 0.25
python test_PAMI_accuracy.py --exp_name 'PAMI_Church_N=1_lambdaREC=10' --ckpt 900000 --sigma 3 --delta 0.5

python test_PAMI_accuracy.py --exp_name 'PAMI_FFHQ_N=1_lambdaREC=10' --ckpt 900000 --sigma 1 --delta 0
python test_PAMI_accuracy.py --exp_name 'PAMI_FFHQ_N=1_lambdaREC=10' --ckpt 900000 --sigma 1 --delta 0.25
python test_PAMI_accuracy.py --exp_name 'PAMI_FFHQ_N=1_lambdaREC=10' --ckpt 900000 --sigma 1 --delta 0.5

python test_PAMI_accuracy.py --exp_name 'PAMI_FFHQ_N=1_lambdaREC=10' --ckpt 900000 --sigma 2 --delta 0
python test_PAMI_accuracy.py --exp_name 'PAMI_FFHQ_N=1_lambdaREC=10' --ckpt 900000 --sigma 2 --delta 0.25
python test_PAMI_accuracy.py --exp_name 'PAMI_FFHQ_N=1_lambdaREC=10' --ckpt 900000 --sigma 2 --delta 0.5

python test_PAMI_accuracy.py --exp_name 'PAMI_FFHQ_N=1_lambdaREC=10' --ckpt 900000 --sigma 3 --delta 0
python test_PAMI_accuracy.py --exp_name 'PAMI_FFHQ_N=1_lambdaREC=10' --ckpt 900000 --sigma 3 --delta 0.25
python test_PAMI_accuracy.py --exp_name 'PAMI_FFHQ_N=1_lambdaREC=10' --ckpt 900000 --sigma 3 --delta 0.5

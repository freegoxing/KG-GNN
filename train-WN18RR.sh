#!/bin/bash

# 本脚本用于为项目中的标准知识图谱数据集执行完整的评估流程：

set -e # 如果任何命令失败，则立即退出

# 要处理的标准数据集列表
dataset="WN18RR"
# 如果没有可用的 CUDA，请将 --use_cuda 标志移除或设置为空字符串 ""
USE_CUDA_FLAG="--use_cuda"
# 隐藏参数
HIDDEN_DIM=64

echo "============================================================"
echo ">>>>> [CLEAN] 清理 $dataset 的 checkpoints <<<<<"
echo "============================================================"

bash checkpoints/$dataset/clear.sh

echo "============================================================"
echo ">>>>> [TRAIN] 开始为数据集 '$dataset' 进行 RGCN 预训练 <<<<<"
echo "============================================================"

uv run train_rgcn.py \
    --dataset_type standard \
    --dataset_name "$dataset" \
    --epochs 4000 \
    --hidden_channels 32 \
    --out_channels $HIDDEN_DIM \
    --learning_rate 0.005 \
    --print_every 100 \
    $USE_CUDA_FLAG

echo "--- [TRAIN] RGCN 预训练完成: $dataset ---"
echo ""

echo "============================================================"
echo ">>>>> [TRAIN] 开始为数据集 '$dataset' 进行 RL 训练 <<<<<"
echo "============================================================"

uv run train_rl.py \
    --dataset_type standard \
    --dataset_name "$dataset" \
    --num_episodes 80000 \
    --gru_hidden_dim $HIDDEN_DIM \
    --save_every 1000 \
    --learning_rate 0.0003 \
    --gradient_accumulation_steps 32 \
    $USE_CUDA_FLAG

echo "--- [TRAIN] RL 训练完成: $dataset ---"
echo ""

echo "============================================================"
echo ">>>>> [EVAL] 开始为数据集 '$dataset' 进行模型评估 <<<<<"
echo "============================================================"

uv run evaluation.py \
    --dataset_type standard \
    --dataset_name "$dataset" \
    --save_plot \
    --gru_hidden_dim $HIDDEN_DIM \
    $USE_CUDA_FLAG

echo "--- [EVAL] 模型评估完成: $dataset ---"
echo ""
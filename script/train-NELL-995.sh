#!/bin/bash

# 本脚本用于为项目中的标准知识图谱数据集执行完整的评估流程：

set -e # 如果任何命令失败，则立即退出

# 要处理的标准数据集列表
dataset="NELL-995"
# 如果没有可用的 CUDA，请将 --use_cuda 标志移除或设置为空字符串 ""
USE_CUDA_FLAG="--use_cuda"
# 数据集类型
DATASET_TYPE="standard"
# 隐藏参数
GRU_HIDDEN_DIM=64
# 图表名称
FILE_NAME="evaluation_summary_v4"
# 设置种子
SEED=45

echo "============================================================"
echo ">>>>> [CLEAN] 清理 $dataset 的 checkpoints <<<<<"
echo "============================================================"

cd ../"checkpoints/$dataset"
bash clear.sh
cd ../..

echo "============================================================"
echo ">>>>> [TRAIN] 开始为数据集 '$dataset' 进行 RGCN 预训练 <<<<<"
echo "============================================================"

uv run train_rgcn.py \
    --dataset_type "$DATASET_TYPE" \
    --dataset_name "$dataset" \
    --epochs 10000 \
    --hidden_channels 64 \
    --out_channels 16 \
    --learning_rate 0.001869501264298604 \
	--weight_decay 2.6031324419300447e-06 \
	--neg_sample_ratio 3.407762661558161 \
    --print_every 100 \
    --seed $SEED \
    --force_deterministic \
    $USE_CUDA_FLAG

echo "--- [TRAIN] RGCN 预训练完成: $dataset ---"
echo ""

echo "============================================================"
echo ">>>>> [TRAIN] 开始为数据集 '$dataset' 进行 RL 训练 <<<<<"
echo "============================================================"

uv run train_rl.py \
    --dataset_type "$DATASET_TYPE" \
    --dataset_name "$dataset" \
    --num_episodes 80000 \
    --gru_hidden_dim $GRU_HIDDEN_DIM \
    --save_every 1000 \
    --learning_rate 0.00047364089705567757 \
    --discount_factor 0.9252209756704769 \
    --entropy_coeff 0.010139233644519272 \
    --gradient_accumulation_steps 32 \
    --action_pruning_k 6 \
    --reward_clipping_value 0.3 \
    --reward_ema_alpha 0.1 \
    --pagerank_exploration_steps 3 \
    --reward_alpha 0.09536431856765598 \
    --reward_eta 1.5910463016291545 \
    --use_advantage_moving_average \
    --advantage_ema_alpha 0.01 \
    --seed $SEED \
    $USE_CUDA_FLAG

echo "--- [TRAIN] RL 训练完成: $dataset ---"
echo ""

echo "============================================================"
echo ">>>>> [EVAL] 开始为数据集 '$dataset' 进行模型评估 <<<<<"
echo "============================================================"

uv run evaluation.py \
    --dataset_type "$DATASET_TYPE" \
    --dataset_name "$dataset" \
    --save_plot \
    --gru_hidden_dim $GRU_HIDDEN_DIM \
    --plot_filename_base "$FILE_NAME" \
    --seed $SEED \
    $USE_CUDA_FLAG

echo "--- [EVAL] 模型评估完成: $dataset ---"
echo ""
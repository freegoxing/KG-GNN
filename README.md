# Knowledge Graph Learning Path Planning with GNN and RL

本项目实现了一个结合关系图卷积网络 (RGCN) 编码器和 强化学习 (RL) 的知识图谱学习路径规划框架。

## 项目结构 (Project Structure)

```
KG-GNN/
├── script/               # 主要的训练和评估流程脚本
├── tuning/               # 使用 Optuna进行超参数调优的脚本
├── config/               # 存储模型的配置说明文件
├── data/                 # 数据集存放目录
├── rgcn_rl_planner/      # 项目核心代码 (模型、训练器、数据加载等)
├── checkpoints/          # 模型权重和训练状态保存目录
├── reports/              # 保存评估结果、指标和图表
├── train_rgcn.py         # RGCN 编码器训练脚本
├── train_rl.py           # RL 代理训练脚本
├── evaluation.py         # 模型评估脚本
├── pyproject.toml        # 项目依赖配置文件
└── README.md             # 本文档
```

## 平台要求 (Platform Requirements)

本项目依赖 NVIDIA RAPIDS 库（cuml, cugraph, cudf 等），推荐在 **Linux + CUDA GPU** 环境下运行。

- 推荐环境：
	- Ubuntu 24.04
	- NVIDIA GPU
	- CUDA 13.x
	- Python 3.12

> ⚠️ 注意：
> - 本项目默认使用 **CUDA 13**（cu13 系列）的 RAPIDS 和 PyTorch 包。
> - 若使用其他 CUDA 版本（例如 11.x 或 12.x），可以在 `project.toml` 中中修改对应依赖及索引 URL，然后重新执行 `uv sync`
>
> ```toml
> [project]
> name = "kg-gnn"
> requires-python = ">=3.12"
> dependencies = [
>     # 修改为对应cuda版本
>     "cudf-cu13==25.12.*",
>     "cuml-cu13==25.12.*",
>     "cugraph-cu13==25.12.*",
>     "nx-cugraph-cu13==25.12.*",
>     ...
> ]
> [[tool.uv.index]]
> name = "pytorch"
> url = "https://download.pytorch.org/whl/cu130"  # 修改为 cu118 / cu121 等对应版本
>
> [[tool.uv.index]]
> name = "nvidia"
> url = "https://pypi.nvidia.com"  # 可保留或根据需要更换
> ```
>
> - 修改后，重新运行 `uv add` 或 `uv sync` 即可安装与系统 CUDA 匹配的二进制包。

## 环境准备 (Environment Setup)

1. **克隆仓库**
   ```bash
   git clone https://github.com/freegoxing/KG-GNN.git
   cd KG-GNN
   ```

2. **安装依赖**
   本项目使用 Python 3.12。推荐使用 `uv` 作为包管理工具以获得最快的安装体验。

   ```bash
   # 安装或升级 uv
   pip install --upgrade uv

   # 使用 uv 同步依赖
   uv sync 
   ```

## 快速开始 (Quick Start)

项目提供了端到端的执行脚本，能够一键完成预训练、训练和评估的全过程。这是最推荐的使用方式。

1. **选择并修改脚本**
   进入 `script/` 目录，所有的执行脚本都在这里。以 `train-NELL-995.sh` 为例，你可以根据需要修改脚本开头的变量：
	- `dataset`: 要使用的数据集名称 (例如 "NELL-995")。
	- `USE_CUDA_FLAG`: 如果你的环境支持并希望使用 CUDA，设置为 `"--use_cuda"`；否则，设置为空字符串 `""`。
	- `SEED`: 随机种子。

2. **执行脚本**
   ```bash
   bash script/train-NELL-995.sh
   ```

   该脚本将会自动执行以下三个核心步骤：
	- **清理 Checkpoints**：清空旧的模型存档。
	- **RGCN 预训练**：使用优化后的超参数训练 RGCN 编码器。
	- **RL 训练**：加载预训练的编码器，并训练 RL 代理进行路径规划。
	- **模型评估**：在测试集上评估训练好的模型，并生成性能图表。

## 分步执行 (Step-by-Step Execution)

如果你想手动控制训练的每一个环节，也可以直接调用 Python 脚本。以下命令均提取自 `train-NELL-995.sh`，展示了其核心调用方式。

#### 1. RGCN 编码器预训练

```bash
uv run train_rgcn.py \
    --dataset_type "standard" \
    --dataset_name "NELL-995" \
    --epochs 10000 \
    --hidden_channels 64 \
    --out_channels 16 \
    --learning_rate 0.001869501264298604 \
    --weight_decay 2.6031324419300447e-06 \
    --neg_sample_ratio 3.407762661558161 \
    --print_every 100 \
    --seed 45 \
    --force_deterministic \
    --use_cuda
```

#### 2. 强化学习代理训练

```bash
uv run train_rl.py \
    --dataset_type "standard" \
    --dataset_name "NELL-995" \
    --num_episodes 80000 \
    --gru_hidden_dim 64 \
    --save_every 1000 \
    --learning_rate 0.00047364089705567757 \
    --discount_factor 0.9252209756704769 \
    --entropy_coeff 0.010139233644519272 \
    --gradient_accumulation_steps 32 \
    --use_cuda
```

#### 3. 模型评估

```bash
uv run evaluation.py \
    --dataset_type "standard" \
    --dataset_name "NELL-995" \
    --save_plot \
    --gru_hidden_dim 64 \
    --plot_filename_base "evaluation_summary_v4" \
    --seed 45 \
    --use_cuda
```

## 超参数调优 (Hyperparameter Tuning)

- 调优脚本位于 `tuning/` 目录，使用 Optuna。
- `script/` 内的训练参数已是调优后的最佳值。

## 结果 (Results)

当评估脚本运行时 (`evaluation.py` 或通过 `bash` 脚本间接运行)，性能指标（如 MRR, Hits@1/3/10）的摘要和图表会自动保存在
`reports/<dataset_name>/` 目录下。

## 数据集 (Datasets)

- [FB15k-237](https://huggingface.co/datasets/KGraph/FB15k-237)
- [WN18RR](https://huggingface.co/datasets/VLyb/WN18RR)
- [NELL-995](https://huggingface.co/datasets/CleverThis/nell-995)

## Notes

- 这个项目旨在用于研究和实验目的。
- 确保使用 GPU + CUDA 环境以获得最佳性能。**本项目对 NVIDIA RAPIDS 库有硬性依赖。** 如果缺少 CUDA 环境和对应的 RAPIDS 库，程序将无法正常运行。
- 该框架具有模块化特性：RGCN 编码器、强化学习代理以及评估功能均可独立运行。
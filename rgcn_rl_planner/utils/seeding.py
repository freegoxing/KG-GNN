# -*- coding: utf-8 -*-
# @Time    : 2024/7/24
# @Author  : free
# @File    : utils.py
# @Description :
import os
import random

import numpy as np
import torch


def set_seed(seed: int, force_deterministic: bool = False):
    """
    设置随机种子以确保代码可复现性。

    Args:
        seed (int): 随机种子。
        force_deterministic (bool): 是否强制使用确定性算法，这可能会牺牲性能。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        # 以下设置为确保 CUDA 操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if force_deterministic:
            # 强制 PyTorch 使用确定性算法
            torch.use_deterministic_algorithms(True)
            # 设置环境变量以确保 cublas 的确定性
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            print("--- 已强制使用确定性算法 (可能会影响性能) ---")

    print(f"--- 随机种子已设置为: {seed} ---")

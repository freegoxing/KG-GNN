# 项目
图神经网络加强化学习的对知识图谱的学习路径规划
知识图谱的数据都在 [知识图谱](./data/kg_data.json)
## 必须使用 Type Hinting（类型注解）

Gemini 生成的代码应满足：
所有函数参数、返回值必须写明类型
类属性建议标注类型
深度学习模型（如 nn.Module）内部方法需注明返回类型
示例：
```python
from torch import Tensor
from typing import Tuple


def preprocess(image: Tensor) -> Tensor:
    """对输入图像进行归一化处理。"""
    return (image - image.mean()) / image.std()
```

## 必须使用中文注释（必要时中英双语）

为增强可读性，Gemini 生成的代码要求：
关键函数顶部使用中文 docstring
复杂逻辑必须有中文行内注释
若涉及学术公式或行业名词，建议附英文注释
示例：
```python
class CNN(nn.Module):
    """
    一个简单的卷积神经网络示例
    A simple CNN model for demonstration
    """


    def forward(self, x: Tensor) -> Tensor:
        # 卷积 + 激活
        x = self.act(self.conv(x))
        return x
```
## 目录结构规范化（面向 Gemini 的项目结构）

代码时按照以下推荐组织：
```
project/
├──  checkpoints               # 模型保存文件夹
├──  data                      # 数据保存文件夹
├──  evaluation.py             # 模型测试代码
├──  reports                   # 测试图表，指标保存文件夹
├──  rgcn_rl_planner           # 模型定义文件夹
│  ├──  data_loader.py         # 数据加载文件
│  ├──  data_utils.py          # 数据处理文件
│  ├──  models.py              # 模型定义文件
│  └──  trainer.py             # 强化学习环境文件
├──  tests                     # 关于测试的工具函数文件夹
│  └──  visualization.py       # 图表可视化文件
├──  train_rgcn.py             # rgcn encoder训练文件
└──  train_rl.py               # 强化学习 训练文件
```
## 尽量使用类封装（Class-Based Structure）

Gemini 生成的训练代码需遵循：
使用 Trainer 类封装训练逻辑
使用 Config 类/字典存储超参数

## 日志输出规范（Logging，而非 print）

要求：
- 使用 Python logging 模块
- Gemini 生成的示例代码中必须包含 logging 配置

## 异常处理必须明确

Gemini 生成代码应：
捕获关键异常
在错误信息中提供明确提示

## 遵循 PEP 8，并适度空行/空格

要求：
- 函数间空 1 行
- 类之间空 2 行
- 变量命名清晰，不使用拼音

# 命令执行规范
关于包的安装`pip install xxx`我会自己安装，只需要修改代码文本内容就行

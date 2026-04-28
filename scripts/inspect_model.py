"""模型配置查看脚本

读取并打印模型的 AutoConfig，用于确认模型架构和参数。
运行：python scripts/inspect_model.py
"""

import sys
from pathlib import Path

# 将项目根目录加入 Python 搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniservellm.config import MODEL_NAME
from transformers import AutoConfig

# 加载模型配置（不下载权重，很快）
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(config)

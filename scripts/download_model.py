"""模型下载脚本

使用 HFLoader 从 HuggingFace Hub 下载 tokenizer 和模型权重到本地缓存。
运行：python scripts/download_model.py
"""

import sys
from pathlib import Path

# 将项目根目录加入 Python 搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniservellm.config import MODEL_NAME
from miniservellm.model_adapter.hf_loader import HFLoader

loader = HFLoader()

# 下载 tokenizer（较小，通常很快）
print(f"Downloading tokenizer: {MODEL_NAME}")
loader.load_tokenizer(MODEL_NAME, trust_remote_code=True)

# 下载模型权重（较大，可能需要较长时间）
# 下载时使用 CPU + float32 以避免设备/精度问题
print(f"Downloading model: {MODEL_NAME}")
loader.load_model(MODEL_NAME, device="cpu", dtype="float32", trust_remote_code=True)

print("Download done.")

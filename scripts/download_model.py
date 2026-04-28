import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniservellm.config import MODEL_NAME
from miniservellm.model_adapter.hf_loader import HFLoader

loader = HFLoader()

print(f"Downloading tokenizer: {MODEL_NAME}")
loader.load_tokenizer(MODEL_NAME, trust_remote_code=True)

print(f"Downloading model: {MODEL_NAME}")
loader.load_model(MODEL_NAME, device="cpu", dtype="float32", trust_remote_code=True)

print("Download done.")

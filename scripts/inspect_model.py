import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniservellm.config import MODEL_NAME
from transformers import AutoConfig

config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(config)

# importing libraries

from pathlib import Path

from models.backends.qwen import QwenBackend


backend = QwenBackend()

backend.load(
    Path("model_store/qwen2_vl")
)

print()

print(backend.get_model_info())

backend.unload()

print()

print("Unload Successful")
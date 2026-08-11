# importing libraries

from pathlib import Path

from utils.model_manager import ModelManager


def dummy_loader(model_path: Path):

    backend = {
        "backend": "dummy"
    }

    resources = {
        "path": model_path
    }

    return backend, resources


manager = ModelManager()

print("Current:", manager.current_model())

backend = manager.load(
    "qwen2_vl",
    loader=dummy_loader
)

print("Loaded:", manager.current_model())

print("Backend:", backend)

manager.unload()

print("After unload:", manager.current_model())
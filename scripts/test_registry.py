# importing libraries

from models.registry import ModelRegistry

print("Registered Models:")
print(ModelRegistry.list_models())

print()

print("Qwen registered?")
print(ModelRegistry.is_registered("qwen2_vl"))
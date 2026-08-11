# importing libraries

from models.backends.dummy import DummyBackend
from models.factory import ModelFactory
from models.registry import ModelRegistry

print()

print("Registered Models")

print("------------------")

print(ModelRegistry.list_models())

print()

backend = ModelFactory.create("dummy")

print("Backend Created")

print(type(backend).__name__)

backend.load()

print()

print(backend.get_model_info())

prediction = backend.predict(None)

print()

print(prediction)

backend.unload()

print()

print(backend.get_model_info())
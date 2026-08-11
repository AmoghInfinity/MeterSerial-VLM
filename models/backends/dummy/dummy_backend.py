# importing libraries

from __future__ import annotations

from pathlib import Path
from typing import Any

from models.base.base_model import BaseMeterModel
from models.registry import ModelRegistry


class DummyBackend(BaseMeterModel):
    """
    Dummy backend used to validate the framework.
    """

    def __init__(self) -> None:

        super().__init__("dummy")

    def load(self) -> None:

        self.is_loaded = True

    def unload(self) -> None:

        self.is_loaded = False

    def preprocess(
        self,
        image_path: Path,
    ) -> Any:

        return image_path

    def predict(
        self,
        processed_input: Any,
    ) -> Any:

        return {
            "serial_number": "123456789",
            "confidence": 1.0,
        }

    def postprocess(
        self,
        prediction: Any,
    ) -> dict:

        return prediction

    def get_model_info(self) -> dict:

        return {
            "name": self.model_name,
            "loaded": self.is_loaded,
        }


ModelRegistry.register(
    "dummy",
    DummyBackend,
)
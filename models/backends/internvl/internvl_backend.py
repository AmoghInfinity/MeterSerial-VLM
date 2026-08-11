# importing libraries

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from models.base.base_model import BaseMeterModel
from models.registry import ModelRegistry
from utils.transformers_compat import (
    ensure_transformers_v5_compatibility,
)


IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


class InternVLBackend(BaseMeterModel):
    """
    InternVL2.5-4B backend for electricity meter OCR.
    """

    def __init__(self) -> None:

        super().__init__("internvl")

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        if self.device.type == "cuda":
            self.dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            self.dtype = torch.float32

        self.model = None
        self.tokenizer = None

        self.input_size = 448
        self.max_num = 6

        self.tile_rows = 2
        self.tile_columns = 3
        self.tile_overlap = 0.20

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load(self, model_path: Path) -> None:
        """
        Load the locally stored InternVL2.5 model.
        """

        ensure_transformers_v5_compatibility()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            fix_mistral_regex=True,
        )

        self.model = AutoModel.from_pretrained(
            model_path,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        self.model = self.model.eval().to(
            self.device
        )

        if not hasattr(
            self.model,
            "all_tied_weights_keys",
        ):
            self.model.all_tied_weights_keys = {}

        self.is_loaded = True

    def unload(self) -> None:
        """
        Release model resources.
        """

        self.model = None
        self.tokenizer = None

        self.is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # InternVL image preprocessing
    # ------------------------------------------------------------------

    def _build_transform(self) -> T.Compose:
        """
        Build the image transformation used by InternVL.
        """

        return T.Compose(
            [
                T.Lambda(
                    lambda image: (
                        image.convert("RGB")
                        if image.mode != "RGB"
                        else image
                    )
                ),
                T.Resize(
                    (
                        self.input_size,
                        self.input_size,
                    ),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )

    def _find_closest_aspect_ratio(
        self,
        aspect_ratio: float,
        target_ratios: list[tuple[int, int]],
        width: int,
        height: int,
        image_size: int,
    ) -> tuple[int, int]:

        best_ratio_diff = float("inf")

        best_ratio = (
            1,
            1,
        )

        area = width * height

        for ratio in target_ratios:

            target_aspect_ratio = (
                ratio[0] / ratio[1]
            )

            ratio_diff = abs(
                aspect_ratio
                - target_aspect_ratio
            )

            if ratio_diff < best_ratio_diff:

                best_ratio_diff = ratio_diff
                best_ratio = ratio

            elif ratio_diff == best_ratio_diff:

                if area > (
                    0.5
                    * image_size
                    * image_size
                    * ratio[0]
                    * ratio[1]
                ):

                    best_ratio = ratio

        return best_ratio

    def _dynamic_preprocess(
        self,
        image: Image.Image,
        use_thumbnail: bool = True,
    ) -> list[Image.Image]:
        """
        Dynamically split an image according to its aspect ratio.

        This follows the official InternVL preprocessing approach.
        """

        orig_width, orig_height = image.size

        aspect_ratio = (
            orig_width / orig_height
        )

        target_ratios: list[tuple[int, int]] = []

        for number in range(
            1,
            self.max_num + 1,
        ):

            for width in range(
                1,
                number + 1,
            ):

                height = number // width

                if (
                    width * height <= self.max_num
                    and width * height >= 1
                ):

                    target_ratios.append(
                        (
                            width,
                            height,
                        )
                    )

        target_ratios = sorted(
            set(target_ratios),
            key=lambda value: (
                value[0] * value[1]
            ),
        )

        target_aspect_ratio = (
            self._find_closest_aspect_ratio(
                aspect_ratio,
                target_ratios,
                orig_width,
                orig_height,
                self.input_size,
            )
        )

        target_width = (
            self.input_size
            * target_aspect_ratio[0]
        )

        target_height = (
            self.input_size
            * target_aspect_ratio[1]
        )

        blocks = (
            target_aspect_ratio[0]
            * target_aspect_ratio[1]
        )

        resized_image = image.resize(
            (
                target_width,
                target_height,
            )
        )

        processed_images: list[Image.Image] = []

        for block_index in range(blocks):

            box = (
                (
                    block_index
                    % target_aspect_ratio[0]
                )
                * self.input_size,
                (
                    block_index
                    // target_aspect_ratio[0]
                )
                * self.input_size,
                (
                    (
                        block_index
                        % target_aspect_ratio[0]
                    )
                    + 1
                )
                * self.input_size,
                (
                    (
                        block_index
                        // target_aspect_ratio[0]
                    )
                    + 1
                )
                * self.input_size,
            )

            split_image = resized_image.crop(
                box
            )

            processed_images.append(
                split_image
            )

        if (
            use_thumbnail
            and len(processed_images) != 1
        ):

            thumbnail = image.resize(
                (
                    self.input_size,
                    self.input_size,
                )
            )

            processed_images.append(
                thumbnail
            )

        return processed_images

    def _load_image(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        """
        Convert an image into InternVL pixel values.
        """

        transform = self._build_transform()

        images = self._dynamic_preprocess(
            image,
            use_thumbnail=True,
        )

        pixel_values = [
            transform(image)
            for image in images
        ]

        pixel_values = torch.stack(
            pixel_values
        )

        return pixel_values

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def _build_prompt(self) -> str:

        return """
Read the electricity meter identification information.

Find the actual meter serial number.

Look for labels such as:
Serial Number, Serial No, S/N, SL NO, Meter Number,
Meter No, or Meter ID.

Return the value associated with the label.

Do not return:
- the label itself
- IMEI numbers unless clearly identified as the meter serial number
- meter readings
- voltage
- current
- frequency
- dates
- model numbers

If no serial number is visible, return:

NOT_FOUND

Return only the identifier value or NOT_FOUND.
""".strip()

    def _run_single_image(
        self,
        image: Image.Image,
        prompt: str,
    ) -> str:
        """
        Run InternVL using the official model.chat interface.
        """

        pixel_values = self._load_image(
            image
        )

        pixel_values = pixel_values.to(
            self.device,
            dtype=self.dtype,
        )

        generation_config = {
            "max_new_tokens": 64,
            "do_sample": False,
        }

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config,
        )

        return response.strip()

    # ------------------------------------------------------------------
    # Generic tiled processing
    # ------------------------------------------------------------------

    def _create_tiles(
        self,
        image: Image.Image,
    ) -> list[Image.Image]:

        width, height = image.size

        base_width = (
            width / self.tile_columns
        )

        base_height = (
            height / self.tile_rows
        )

        step_x = (
            base_width
            * (1.0 - self.tile_overlap)
        )

        step_y = (
            base_height
            * (1.0 - self.tile_overlap)
        )

        tile_width = min(
            width,
            int(
                base_width
                * (1.0 + self.tile_overlap)
            ),
        )

        tile_height = min(
            height,
            int(
                base_height
                * (1.0 + self.tile_overlap)
            ),
        )

        tiles: list[Image.Image] = []

        for row in range(
            self.tile_rows
        ):

            for column in range(
                self.tile_columns
            ):

                x = int(
                    column * step_x
                )

                y = int(
                    row * step_y
                )

                x = min(
                    x,
                    max(
                        0,
                        width - tile_width,
                    ),
                )

                y = min(
                    y,
                    max(
                        0,
                        height - tile_height,
                    ),
                )

                tiles.append(
                    image.crop(
                        (
                            x,
                            y,
                            x + tile_width,
                            y + tile_height,
                        )
                    )
                )

        return tiles

    # ------------------------------------------------------------------
    # Framework interface
    # ------------------------------------------------------------------

    def preprocess(
        self,
        image_path: Path,
        prompt: str | None = None,
    ) -> dict[str, Any]:

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        tiles = self._create_tiles(
            image
        )

        return {
            "image": image,
            "tiles": tiles,
            "prompt": (
                prompt
                or self._build_prompt()
            ),
        }

    def predict(
        self,
        processed_input: dict[str, Any],
    ) -> dict[str, Any]:

        if not self.is_loaded:
            raise RuntimeError(
                "InternVL2.5 model is not loaded."
            )

        image = processed_input["image"]
        tiles = processed_input["tiles"]
        prompt = processed_input["prompt"]

        candidates: list[
            dict[str, str]
        ] = []

        candidates.append(
            {
                "region": "full_image",
                "output": (
                    self._run_single_image(
                        image,
                        prompt,
                    )
                ),
            }
        )

        for index, tile in enumerate(
            tiles
        ):

            candidates.append(
                {
                    "region": (
                        f"tile_{index + 1}"
                    ),
                    "output": (
                        self._run_single_image(
                            tile,
                            prompt,
                        )
                    ),
                }
            )

        return {
            "candidates": candidates,
        }

    def postprocess(
        self,
        prediction: dict[str, Any],
    ) -> dict[str, Any]:

        return {
            "serial_number": "",
            "model": self.model_name,
            "candidates": prediction[
                "candidates"
            ],
        }

    def get_model_info(
        self,
    ) -> dict[str, Any]:

        return {
            "name": self.model_name,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "loaded": self.is_loaded,
            "input_size": self.input_size,
            "max_num": self.max_num,
            "tile_rows": self.tile_rows,
            "tile_columns": self.tile_columns,
            "tile_overlap": self.tile_overlap,
        }


ModelRegistry.register(
    "internvl",
    InternVLBackend,
)
# importing libraries

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from models.base.base_model import BaseMeterModel
from models.registry import ModelRegistry


class QwenBackend(BaseMeterModel):
    """
    Qwen2-VL backend for location-independent electricity meter
    serial number extraction.
    """

    def __init__(self) -> None:

        super().__init__("qwen2_vl")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if self.device.type == "cuda":
            self.dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            self.dtype = torch.float32

        self.processor = None
        self.model = None

        self.tile_rows = 2
        self.tile_columns = 3
        self.tile_overlap = 0.20

        self.min_pixels = 512 * 28 * 28
        self.max_pixels = 1024 * 28 * 28

    def load(self, model_path: Path) -> None:
        """
        Load the locally stored Qwen2-VL model.
        """

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map="auto",
        )

        self.model.eval()
        self.is_loaded = True

    def unload(self) -> None:
        """
        Release model resources.
        """

        self.model = None
        self.processor = None
        self.is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _create_tiles(
        self,
        image: Image.Image,
    ) -> list[Image.Image]:
        """
        Create a generic overlapping grid.
        """

        width, height = image.size

        rows = self.tile_rows
        columns = self.tile_columns

        base_tile_width = width / columns
        base_tile_height = height / rows

        step_x = base_tile_width * (1.0 - self.tile_overlap)
        step_y = base_tile_height * (1.0 - self.tile_overlap)

        tile_width = min(
            width,
            int(base_tile_width + base_tile_width * self.tile_overlap),
        )

        tile_height = min(
            height,
            int(base_tile_height + base_tile_height * self.tile_overlap),
        )

        tiles: list[Image.Image] = []

        for row in range(rows):

            for column in range(columns):

                x = int(column * step_x)
                y = int(row * step_y)

                x = min(x, max(0, width - tile_width))
                y = min(y, max(0, height - tile_height))

                tile = image.crop(
                    (
                        x,
                        y,
                        x + tile_width,
                        y + tile_height,
                    )
                )

                tiles.append(tile)

        return tiles

    def _build_prompt(self) -> str:
        """
        Build the OCR extraction instruction.
        """

        return """
You are performing OCR on an electricity meter photograph.

Find an actual electricity meter identifier in this image.

Priority order:

1. Serial Number
2. Serial No
3. S/N
4. SL NO
5. Meter Number
6. Meter No
7. Meter ID

A label such as "Serial Number", "Serial No", "SL NO",
"S/N", or "Meter No" is NOT the answer.

Return the VALUE associated with the label.

For example:

SL NO: U5028045

means the answer is:

U5028045

Do NOT return:
- the label itself
- IMEI numbers
- voltage
- current
- frequency
- dates
- meter readings
- model numbers
- manufacturer names

If this image region does not contain an identifiable serial number
or meter number, return exactly:

NOT_FOUND

Return only the identifier value or NOT_FOUND.
""".strip()

    def _run_single_image(
        self,
        image: Image.Image,
        prompt: str,
    ) -> str:
        """
        Run Qwen2-VL on one image region.
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.model.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )

        input_token_length = inputs["input_ids"].shape[1]

        generated_ids = generated_ids[
            :,
            input_token_length:,
        ]

        output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output[0].strip()

    def _clean_candidate(
        self,
        value: str,
    ) -> str:
        """
        Remove common formatting around an OCR candidate.
        """

        value = value.strip()

        prefixes = (
            "serial number:",
            "serial no:",
            "serial:",
            "sl no:",
            "s/n:",
            "meter number:",
            "meter no:",
            "meter id:",
        )

        lower_value = value.lower()

        for prefix in prefixes:

            if lower_value.startswith(prefix):

                value = value[len(prefix):].strip()

                break

        return value

    def _is_valid_candidate(
        self,
        value: str,
    ) -> bool:
        """
        Check whether an OCR output is a plausible identifier.
        """

        value = self._clean_candidate(value)

        if not value:
            return False

        if value.upper() == "NOT_FOUND":
            return False

        excluded_labels = {
            "serial",
            "serial number",
            "serial no",
            "serial number:",
            "meter",
            "meter number",
            "meter no",
            "meter id",
            "sl no",
            "s/n",
        }

        if value.lower() in excluded_labels:
            return False

        alphanumeric_count = sum(
            character.isalnum()
            for character in value
        )

        return alphanumeric_count >= 4

    def preprocess(
        self,
        image_path: Path,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Prepare the full image and generic tiles.
        """

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        tiles = self._create_tiles(image)

        return {
            "image": image,
            "tiles": tiles,
            "prompt": prompt or self._build_prompt(),
        }

    def predict(
        self,
        processed_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run full-image and tiled inference.
        """

        if not self.is_loaded:
            raise RuntimeError(
                "Qwen2-VL model is not loaded."
            )

        image = processed_input["image"]
        tiles = processed_input["tiles"]
        prompt = processed_input["prompt"]

        candidates: list[dict[str, Any]] = []

        full_output = self._run_single_image(
            image,
            prompt,
        )

        candidates.append(
            {
                "region": "full_image",
                "output": full_output,
            }
        )

        for index, tile in enumerate(tiles):

            output = self._run_single_image(
                tile,
                prompt,
            )

            candidates.append(
                {
                    "region": f"tile_{index + 1}",
                    "output": output,
                }
            )

        return {
            "image": image,
            "prompt": prompt,
            "candidates": candidates,
        }

    def _verify_candidates(
        self,
        image: Image.Image,
        candidates: list[str],
    ) -> str:
        """
        Use Qwen to select the candidate that actually corresponds
        to the serial-number field in the original image.
        """

        candidate_text = "\n".join(
            f"{index + 1}. {candidate}"
            for index, candidate in enumerate(candidates)
        )

        verification_prompt = f"""
You are verifying an OCR result from an electricity meter.

Look at the ORIGINAL meter image carefully.

Below are candidate identifiers extracted from different
regions of the image:

{candidate_text}

Determine which candidate is actually printed as the meter's
serial number, serial number value, SL NO, S/N, meter number,
or equivalent identifier.

Do not select an IMEI unless the image clearly identifies it
as the meter serial number.

Do not select a meter reading, date, voltage, current, model
number, or other unrelated number.

Compare the candidates directly against the visible text in
the image.

Return ONLY the exact candidate value.

If none of the candidates corresponds to the meter identifier,
return:

NOT_FOUND
""".strip()

        return self._run_single_image(
            image,
            verification_prompt,
        )

    def postprocess(
        self,
        prediction: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Clean candidates and use a verification pass to select
        the final serial number.
        """

        raw_candidates = prediction["candidates"]

        valid_candidates: list[str] = []

        for candidate in raw_candidates:

            value = self._clean_candidate(
                candidate["output"]
            )

            if self._is_valid_candidate(value):
                valid_candidates.append(value)

        unique_candidates: list[str] = []

        for candidate in valid_candidates:

            if candidate not in unique_candidates:
                unique_candidates.append(candidate)

        if not unique_candidates:

            serial_number = ""

        elif len(unique_candidates) == 1:

            serial_number = unique_candidates[0]

        else:

            verified = self._verify_candidates(
                prediction["image"],
                unique_candidates,
            )

            verified = self._clean_candidate(
                verified
            )

            if self._is_valid_candidate(verified):

                serial_number = verified

            else:

                serial_number = unique_candidates[0]

        return {
            "serial_number": serial_number,
            "model": self.model_name,
            "candidates": raw_candidates,
            "unique_candidates": unique_candidates,
        }

    def get_model_info(self) -> dict[str, Any]:
        """
        Return backend information.
        """

        return {
            "name": self.model_name,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "loaded": self.is_loaded,
            "tile_rows": self.tile_rows,
            "tile_columns": self.tile_columns,
            "tile_overlap": self.tile_overlap,
        }


ModelRegistry.register(
    "qwen2_vl",
    QwenBackend,
)
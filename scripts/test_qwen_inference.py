# importing libraries

import argparse
from pathlib import Path

from models.backends.qwen import QwenBackend


def load_prompt() -> str:
    """
    Load the optional external prompt.
    """

    prompt_path = Path(
        "prompts/meter_serial_prompt.txt"
    )

    if not prompt_path.exists():
        return ""

    return prompt_path.read_text(
        encoding="utf-8"
    ).strip()


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Run tiled Qwen2-VL meter OCR."
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to the meter image.",
    )

    args = parser.parse_args()

    backend = QwenBackend()

    try:

        print("Loading Qwen2-VL...")

        backend.load(
            Path("model_store/qwen2_vl")
        )

        print("Model loaded.")
        print()

        prompt = load_prompt()

        processed_input = backend.preprocess(
            args.image,
            prompt if prompt else None,
        )

        total_regions = (
            1 + len(processed_input["tiles"])
        )

        print(
            f"Running {total_regions} image regions..."
        )

        prediction = backend.predict(
            processed_input
        )

        result = backend.postprocess(
            prediction
        )

        print()
        print("=" * 60)
        print("Qwen2-VL Tiled OCR Result")
        print("=" * 60)

        print()
        print(
            f"Serial Number: "
            f"{result['serial_number']}"
        )

        print()
        print("Region Outputs:")
        print("-" * 60)

        for candidate in result["candidates"]:

            print(
                f"{candidate['region']:<15}"
                f"{candidate['output']}"
            )

    finally:

        backend.unload()

        print()
        print("Model unloaded.")


if __name__ == "__main__":
    main()
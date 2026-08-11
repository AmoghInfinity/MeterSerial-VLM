# importing libraries

import argparse
from pathlib import Path

from models.backends.lightonocr import LightOnOCRBackend


def load_prompt() -> str:
    """
    Load the shared meter serial-number prompt.
    """

    prompt_path = Path(
        "prompts/meter_serial_prompt.txt"
    )

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}"
        )

    return prompt_path.read_text(
        encoding="utf-8"
    ).strip()


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Run LightOnOCR meter serial-number extraction."
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to the meter image.",
    )

    args = parser.parse_args()

    backend = LightOnOCRBackend()

    try:

        print("Loading LightOnOCR-2-1B...")

        backend.load(
            Path("model_store/lightonocr")
        )

        print("Model loaded.")
        print()

        prompt = load_prompt()

        processed_input = backend.preprocess(
            args.image,
            prompt,
        )

        region_count = (
            1 + len(processed_input["tiles"])
        )

        print(
            f"Running {region_count} image regions..."
        )

        prediction = backend.predict(
            processed_input
        )

        result = backend.postprocess(
            prediction
        )

        print()
        print("=" * 70)
        print("LightOnOCR Serial Number Results")
        print("=" * 70)

        print()

        for region in result["regions"]:

            print(
                f"{region['region']:<15}"
                f"Serial: {region['serial_number']}"
            )

            print(
                f"{'':<15}"
                f"Raw:    {region['raw_output']}"
            )

            print()

    finally:

        backend.unload()

        print("Model unloaded.")


if __name__ == "__main__":
    main()
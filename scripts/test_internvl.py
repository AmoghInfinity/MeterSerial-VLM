# importing libraries

import argparse
from pathlib import Path

from models.backends.internvl import InternVLBackend


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Run InternVL on an electricity meter image."
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to the meter image.",
    )

    args = parser.parse_args()

    backend = InternVLBackend()

    try:

        print("Loading InternVL2.5-4B...")

        backend.load(
            Path("model_store/internvl")
        )

        print("Model loaded.")
        print()

        processed_input = backend.preprocess(
            args.image
        )

        print(
            f"Running "
            f"{1 + len(processed_input['tiles'])} "
            f"image regions..."
        )

        prediction = backend.predict(
            processed_input
        )

        print()
        print("=" * 60)
        print("InternVL Results")
        print("=" * 60)

        for candidate in prediction["candidates"]:

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
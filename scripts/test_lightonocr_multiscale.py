# importing libraries

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PIL import Image

from models.backends.lightonocr.lightonocr_backend import (
    LightOnOCRBackend,
)
from utils.lightonocr_extractor import (
    LightOnOCRExtractor,
)
from utils.lightonocr_consolidator import (
    LightOnOCRConsolidator,
)
from utils.multiscale_ocr import (
    MultiScaleOCR,
)


def print_separator(
    character: str = "=",
    width: int = 78,
) -> None:

    print(
        character * width
    )


def main() -> None:

    if len(sys.argv) < 2:

        print(
            'Usage: python -m scripts.test_lightonocr_multiscale '
            '"data/images/dm_1.png"'
        )

        raise SystemExit(1)

    image_path = Path(
        sys.argv[1]
    )

    if not image_path.exists():

        print(
            f"Image not found: {image_path}"
        )

        raise SystemExit(1)

    # --------------------------------------------------------------
    # Initialize components
    # --------------------------------------------------------------

    backend = LightOnOCRBackend()

    extractor = LightOnOCRExtractor()

    consolidator = LightOnOCRConsolidator()

    multiscale = MultiScaleOCR(
        upscale_factor=2.0,
        window_size=700,
        overlap=0.30,
    )

    all_extractions: dict[
        str,
        dict[str, Any],
    ] = {}

    try:

        print(
            "Loading LightOnOCR-2-1B..."
        )

        backend.load(
            Path("model_store/lightonocr")
        )

        print(
            "Model loaded."
        )

        # ----------------------------------------------------------
        # Load image
        # ----------------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        print()
        print(
            f"Image: {image_path}"
        )

        print(
            f"Original size: {image.size[0]} "
            f"x {image.size[1]}"
        )

        print()

        # ----------------------------------------------------------
        # Generate multi-scale regions
        # ----------------------------------------------------------

        regions = list(
            multiscale.generate(
                image
            )
        )

        print(
            f"Running {len(regions)} "
            "multi-scale OCR regions..."
        )

        # ----------------------------------------------------------
        # Run OCR
        # ----------------------------------------------------------

        for index, region in enumerate(
            regions,
            start=1,
        ):

            print()
            print_separator(
                "-"
            )

            print(
                f"[{index}/{len(regions)}] "
                f"{region.name}"
            )

            print(
                f"Scale: {region.scale}"
            )

            print(
                f"Size: "
                f"{region.image.size[0]} "
                f"x "
                f"{region.image.size[1]}"
            )

            print_separator(
                "-"
            )

            # ------------------------------------------------------
            # Use the backend's existing prompt.
            # ------------------------------------------------------

            prompt = backend._build_prompt()

            # ------------------------------------------------------
            # Run exactly one LightOnOCR inference.
            # ------------------------------------------------------

            raw_output = backend._run_single_image(
                region.image,
                prompt,
            )

            print()
            print(
                "RAW LIGHTONOCR OUTPUT"
            )

            print(
                raw_output
                if raw_output
                else "NOT_FOUND"
            )

            # ------------------------------------------------------
            # Python extraction
            # ------------------------------------------------------

            extraction = extractor.extract(
                raw_output
            )

            all_extractions[
                region.name
            ] = extraction

            print()
            print(
                "PYTHON EXTRACTION"
            )

            print(
                f"Serial Number : "
                f"{extraction.get('serial_number') or 'NOT_FOUND'}"
            )

            print(
                f"IMEI          : "
                f"{extraction.get('imei') or 'NOT_FOUND'}"
            )

            print(
                f"Dates         : "
                f"{extraction.get('dates') or 'NOT_FOUND'}"
            )

            # ------------------------------------------------------
            # Adaptive stopping
            # ------------------------------------------------------

            if _has_all_required_information(
                all_extractions
            ):

                print()
                print(
                    "All required fields have been found."
                )

                print(
                    "Stopping additional OCR regions."
                )

                break

        # ----------------------------------------------------------
        # Final consolidation
        # ----------------------------------------------------------

        final_result = consolidator.consolidate(
            all_extractions
        )

        print()
        print_separator()

        print(
            "                    FINAL RESULT"
        )

        print_separator()

        print()
        print(
            f"Serial Number : "
            f"{final_result.get('serial_number') or 'NOT_FOUND'}"
        )

        print(
            f"IMEI          : "
            f"{final_result.get('imei') or 'NOT_FOUND'}"
        )

        print()
        print(
            "Dates:"
        )

        dates = final_result.get(
            "dates",
            {},
        )

        for category in (
            "manufacturing",
            "dated",
            "installation",
            "commissioning",
        ):

            values = dates.get(
                category,
                [],
            )

            print(
                f"  {category.capitalize():14}: "
                f"{', '.join(values) if values else 'NOT_FOUND'}"
            )

        print()
        print_separator()

    finally:

        backend.unload()

        print()
        print(
            "Model unloaded."
        )


def _has_all_required_information(
    results: dict[
        str,
        dict[str, Any],
    ],
) -> bool:
    """
    Determine whether enough information has already
    been recovered to stop additional OCR.

    At minimum we require:
    - serial number
    - IMEI

    Dates are optional because some meters may not
    expose them clearly.
    """

    serial_found = False
    imei_found = False

    for result in results.values():

        if result.get(
            "serial_number",
            "",
        ):

            serial_found = True

        if result.get(
            "imei",
            "",
        ):

            imei_found = True

    return (
        serial_found
        and imei_found
    )


if __name__ == "__main__":
    main()
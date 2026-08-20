# importing libraries

from utils.lightonocr_consolidator import (
    LightOnOCRConsolidator,
)


def main() -> None:

    consolidator = LightOnOCRConsolidator()

    region_results = {
        "full_image": {
            "serial_number": "",
            "dates": {},
        },
        "tile_1": {
            "serial_number": "U5028359",
            "dates": {
                "manufacturing": ["02/2025"],
            },
        },
        "tile_2": {
            "serial_number": "U5028359",
            "dates": {
                "manufacturing": ["02/2025"],
                "dated": ["23.12.2024"],
            },
        },
        "tile_3": {
            "serial_number": "",
            "dates": {},
        },
        "tile_4": {
            "serial_number": "U5028359",
            "dates": {
                "installation": ["15/10/2025"],
            },
        },
        "tile_5": {
            "serial_number": "U50283",
            "dates": {
                "manufacturing": ["02/2025"],
            },
        },
        "tile_6": {
            "serial_number": "",
            "dates": {},
        },
    }

    result = consolidator.consolidate(
        region_results
    )

    print("=" * 70)
    print("CONSOLIDATION RESULT")
    print("=" * 70)
    print(result)

    expected_serial = "U5028359"

    assert (
        result["serial_number"]
        == expected_serial
    )

    assert (
        "02/2025"
        in result["dates"]["manufacturing"]
    )

    assert (
        "23.12.2024"
        in result["dates"]["dated"]
    )

    assert (
        "15/10/2025"
        in result["dates"]["installation"]
    )

    print()
    print("PASS")


if __name__ == "__main__":
    main()
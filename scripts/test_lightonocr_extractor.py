# importing libraries

from utils.lightonocr_extractor import (
    LightOnOCRExtractor,
)


def main() -> None:

    extractor = LightOnOCRExtractor()

    test_cases = [
        "SL NO U5028359",
        "SL NO. U5028359",
        "Serial Number: ABC12345",
        "Serial No ABC12345",
        "S/N: ZX987654",
        "Meter Number: MTR123456",
        "Meter No MTR998877",
        "Meter ID: ABCD1234",
    ]

    for text in test_cases:

        result = extractor.extract(
            text
        )

        print(
            f"{text:<35} -> {result}"
        )


if __name__ == "__main__":
    main()
# importing libraries

from __future__ import annotations

import re
from typing import Any


class LightOnOCRExtractor:
    """
    Extract structured meter information from LightOnOCR OCR text.

    Extracted information:
    - serial number
    - IMEI number
    - manufacturing date
    - dated
    - installation date
    - commissioning date
    """

    SERIAL_LABEL_PATTERNS = (
        r"serial\s*(?:number|no\.?)?",
        r"s\.?\s*/\s*n\.?",
        r"s\.?\s+no\.?",
        r"sl\.?\s*no\.?",
        r"meter\s*(?:number|no\.?|id)",
        r"device\s*(?:number|no\.?|id)",
    )

    IMEI_LABEL_PATTERNS = (
        r"imei",
        r"imei\s*(?:number|no\.?)",
    )

    DATE_LABEL_PATTERNS = {
        "manufacturing": (
            r"mfg\.?",
            r"manufactured",
            r"manufacturing\s*date",
            r"manufacture\s*date",
        ),
        "installation": (
            r"installation\s*date",
            r"installed\s*date",
            r"installation",
        ),
        "commissioning": (
            r"commissioning\s*date",
            r"commissioned",
            r"commissioning",
        ),
    }

    DATE_PATTERN = (
        r"\b"
        r"(?:"
        r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
        r"|"
        r"\d{4}[./-]\d{1,2}[./-]\d{1,2}"
        r"|"
        r"\d{1,2}[./-]\d{4}"
        r"|"
        r"\d{4}[./-]\d{1,2}"
        r"|"
        r"(?:"
        r"jan(?:uary)?|"
        r"feb(?:ruary)?|"
        r"mar(?:ch)?|"
        r"apr(?:il)?|"
        r"may|"
        r"jun(?:e)?|"
        r"jul(?:y)?|"
        r"aug(?:ust)?|"
        r"sep(?:tember)?|"
        r"oct(?:ober)?|"
        r"nov(?:ember)?|"
        r"dec(?:ember)?"
        r")"
        r"\s+"
        r"\d{1,2}"
        r"(?:st|nd|rd|th)?"
        r"(?:,)?"
        r"\s+"
        r"\d{4}"
        r")"
        r"\b"
    )

    def extract(
        self,
        ocr_text: str,
    ) -> dict[str, Any]:
        """
        Extract serial number, IMEI and dates.
        """

        if not ocr_text:
            return {
                "serial_number": "",
                "imei": "",
                "dates": {},
            }

        text = self._normalize_text(
            ocr_text
        )

        lines = text.splitlines()

        return {
            "serial_number": self._extract_serial_number(
                lines
            ),
            "imei": self._extract_imei(
                lines
            ),
            "dates": self._extract_dates(
                lines
            ),
        }

    def _extract_serial_number(
        self,
        lines: list[str],
    ) -> str:
        """
        Extract the first valid serial-number value.
        """

        for line in lines:

            result = self._extract_serial_from_line(
                line
            )

            if result:
                return result

        return ""

    def _extract_serial_from_line(
        self,
        line: str,
    ) -> str:
        """
        Extract an identifier following a serial-number label.
        """

        line = line.strip()

        if not line:
            return ""

        label_pattern = "|".join(
            self.SERIAL_LABEL_PATTERNS
        )

        pattern = re.compile(
            rf"""
            (?:{label_pattern})
            \s*
            [:;\-#]?
            \s*
            ([A-Za-z0-9][A-Za-z0-9./_-]{{3,}})
            """,
            re.IGNORECASE | re.VERBOSE,
        )

        match = pattern.search(line)

        if not match:
            return ""

        candidate = self._clean_candidate(
            match.group(1)
        )

        # Do not accept an IMEI as a serial number.
        if self._is_imei(candidate):
            return ""

        return candidate

    def _extract_imei(
        self,
        lines: list[str],
    ) -> str:
        """
        Extract the first valid IMEI associated with an IMEI label.

        IMEI values are expected to contain 15 digits.
        """

        label_pattern = "|".join(
            self.IMEI_LABEL_PATTERNS
        )

        pattern = re.compile(
            rf"""
            (?:{label_pattern})
            \s*
            [:;\-#]?
            \s*
            ([0-9][0-9\s]{{13,20}})
            """,
            re.IGNORECASE | re.VERBOSE,
        )

        for line in lines:

            match = pattern.search(line)

            if not match:
                continue

            candidate = re.sub(
                r"\s+",
                "",
                match.group(1),
            )

            if self._is_imei(candidate):
                return candidate

        return ""

    def _is_imei(
        self,
        value: str,
    ) -> bool:
        """
        Check whether a value has the expected IMEI format.
        """

        return bool(
            re.fullmatch(
                r"\d{15}",
                value,
            )
        )

    def _extract_dates(
        self,
        lines: list[str],
    ) -> dict[str, list[str]]:
        """
        Extract all relevant meter-associated dates.
        """

        dates: dict[str, list[str]] = {}

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # Extract specific date categories.
            for category in (
                "manufacturing",
                "installation",
                "commissioning",
            ):

                labels = self.DATE_LABEL_PATTERNS[
                    category
                ]

                label_pattern = "|".join(
                    labels
                )

                pattern = re.compile(
                    rf"""
                    (?<![A-Za-z])
                    (?:{label_pattern})
                    \s*
                    [:;\-]?
                    \s*
                    ({self.DATE_PATTERN})
                    """,
                    re.IGNORECASE | re.VERBOSE,
                )

                matches = pattern.findall(
                    line
                )

                for match in matches:

                    date_value = self._clean_date(
                        match
                    )

                    self._add_date(
                        dates,
                        category,
                        date_value,
                    )

            # Extract explicit "Dated".
            dated_pattern = re.compile(
                rf"""
                (?<![A-Za-z])
                dated
                \s*
                [:;\-]?
                \s*
                ({self.DATE_PATTERN})
                """,
                re.IGNORECASE | re.VERBOSE,
            )

            for match in dated_pattern.findall(line):

                date_value = self._clean_date(
                    match
                )

                self._add_date(
                    dates,
                    "dated",
                    date_value,
                )

            # Remove specific date labels before
            # searching for standalone "Date".
            generic_date_line = re.sub(
                r"""
                (?:
                    installation\s+date
                    |
                    installed\s+date
                    |
                    manufacturing\s+date
                    |
                    manufacture\s+date
                    |
                    commissioning\s+date
                )
                """,
                "",
                line,
                flags=re.IGNORECASE | re.VERBOSE,
            )

            standalone_date_pattern = re.compile(
                rf"""
                (?<![A-Za-z])
                date
                \s*
                [:;\-]?
                \s*
                ({self.DATE_PATTERN})
                """,
                re.IGNORECASE | re.VERBOSE,
            )

            for match in standalone_date_pattern.findall(
                generic_date_line
            ):

                date_value = self._clean_date(
                    match
                )

                self._add_date(
                    dates,
                    "dated",
                    date_value,
                )

        return dates

    def _add_date(
        self,
        dates: dict[str, list[str]],
        category: str,
        date_value: str,
    ) -> None:
        """
        Add a date without duplicates.
        """

        if not date_value:
            return

        dates.setdefault(
            category,
            []
        )

        if date_value not in dates[category]:

            dates[category].append(
                date_value
            )

    def _normalize_text(
        self,
        text: str,
    ) -> str:
        """
        Normalize OCR text while preserving lines.
        """

        text = (
            text
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

        text = re.sub(
            r"\bSL\s*\.\s*NO\b",
            "SL NO",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\bS\s*\.\s*NO\b",
            "S NO",
            text,
            flags=re.IGNORECASE,
        )

        return text.strip()

    def _clean_candidate(
        self,
        candidate: str,
    ) -> str:
        """
        Remove OCR punctuation.
        """

        return candidate.strip(
            ".,:;()[]{}<>\"'`"
        )

    def _clean_date(
        self,
        date_value: str,
    ) -> str:
        """
        Clean a detected date.
        """

        return date_value.strip(
            ".,:;()[]{}<>\"'`"
        )
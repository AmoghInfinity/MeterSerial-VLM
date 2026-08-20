# importing libraries

from __future__ import annotations

import re
from collections import Counter
from typing import Any


class LightOnOCRExtractor:
    """
    Deterministic extractor for electricity-meter OCR output.

    Extracts:
    - serial number
    - IMEI
    - manufacturing date
    - dated date
    - installation date
    - commissioning date
    """

    SERIAL_LABEL_PATTERNS = (
        r"\bserial\s*(?:number|no)\b",
        r"\bserial\s*no\.?\b",
        r"\bs\.?\s*/\s*n\.?\b",
        r"\bs\.?\s*no\.?\b",
        r"\bsl\.?\s*no\.?\b",
        r"\bmeter\s*(?:number|no|id)\b",
        r"\bdevice\s*(?:number|no|id)\b",
    )

    IMEI_LABEL_PATTERN = (
        r"\bimei\s*(?:no\.?|number)?\b"
    )

    DATE_LABEL_PATTERNS = {
        "manufacturing": (
            r"\bmfg\b",
            r"\bmfg\.",
            r"\bmanufactur(?:ed|ing)\b",
            r"\bmanufacture\s*date\b",
            r"\bmanufacturing\s*date\b",
        ),
        "dated": (
            r"\bdated\b",
        ),
        "installation": (
            r"\binstallation\s*date\b",
            r"\binstalled\b",
            r"\binstallation\b",
        ),
        "commissioning": (
            r"\bcommission(?:ed|ing)\s*date\b",
            r"\bcommission(?:ed|ing)\b",
        ),
    }

    DATE_PATTERN = (
        r"\b"
        r"(?:"
        r"\d{1,2}[\/.\-]\d{1,2}[\/.\-]\d{2,4}"
        r"|"
        r"\d{4}[\/.\-]\d{1,2}[\/.\-]\d{1,2}"
        r"|"
        r"\d{1,2}[\/.\-]\d{4}"
        r"|"
        r"\d{4}[\/.\-]\d{1,2}"
        r")"
        r"\b"
    )

    def __init__(self) -> None:
        pass

    def extract(
        self,
        text: str,
    ) -> dict[str, Any]:
        """
        Extract meter identification information from OCR text.
        """

        if not text:
            return {
                "serial_number": "",
                "imei": "",
                "dates": {},
            }

        normalized_text = self._normalize_text(
            text
        )

        lines = [
            line.strip()
            for line in normalized_text.splitlines()
            if line.strip()
        ]

        serial_number = self._extract_serial_number(
            lines
        )

        imei = self._extract_imei(
            lines
        )

        dates = self._extract_dates(
            lines
        )

        return {
            "serial_number": serial_number,
            "imei": imei,
            "dates": dates,
        }

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_text(
        self,
        text: str,
    ) -> str:
        """
        Normalize common OCR formatting variations.
        """

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        text = text.replace(
            "\\",
            "",
        )

        # Normalize common markdown emphasis.
        text = re.sub(
            r"[*_`]+",
            "",
            text,
        )

        # Normalize non-breaking spaces.
        text = text.replace(
            "\u00a0",
            " ",
        )

        # Normalize repeated whitespace while preserving lines.
        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        # Normalize SL NO variations.
        text = re.sub(
            r"\bS\s*L\s*\.\s*N\s*O\.?\b",
            "SL NO",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\bSL\s*\.\s*NO\.?\b",
            "SL NO",
            text,
            flags=re.IGNORECASE,
        )

        # Normalize S. NO variations.
        text = re.sub(
            r"\bS\s*\.\s*NO\.?\b",
            "S NO",
            text,
            flags=re.IGNORECASE,
        )

        # Normalize IMEI labels.
        text = re.sub(
            r"\bIMEI\s*(?:N[oO]|N[oO]\.|NUMBER)\b",
            "IMEI",
            text,
            flags=re.IGNORECASE,
        )

        return text.strip()

    # ------------------------------------------------------------------
    # Serial number
    # ------------------------------------------------------------------

    def _extract_serial_number(
        self,
        lines: list[str],
    ) -> str:
        """
        Extract a serial number from lines containing a serial label.
        """

        candidates: list[str] = []

        for line in lines:

            for pattern in self.SERIAL_LABEL_PATTERNS:

                match = re.search(
                    pattern,
                    line,
                    flags=re.IGNORECASE,
                )

                if not match:
                    continue

                remainder = line[
                    match.end():
                ].strip()

                remainder = re.sub(
                    r"^[\s:;,#=\-]+",
                    "",
                    remainder,
                )

                candidate = self._clean_identifier(
                    remainder
                )

                if self._is_valid_serial_candidate(
                    candidate
                ):
                    candidates.append(
                        candidate
                    )

                break

        if not candidates:
            return ""

        return self._select_best_identifier(
            candidates
        )

    def _clean_identifier(
        self,
        value: str,
    ) -> str:
        """
        Clean an OCR identifier candidate.
        """

        value = value.strip()

        # Remove trailing punctuation.
        value = re.sub(
            r"[\s:;,.\-]+$",
            "",
            value,
        )

        # Take the first token when OCR has
        # appended unrelated text.
        tokens = value.split()

        if not tokens:
            return ""

        first = tokens[0].strip(
            "[](){}<>:;,."
        )

        # Keep only common identifier characters.
        first = re.sub(
            r"[^A-Za-z0-9./_-]",
            "",
            first,
        )

        return first

    def _is_valid_serial_candidate(
        self,
        value: str,
    ) -> bool:
        """
        Validate a serial-number candidate.

        Serial numbers can be alphanumeric and may have
        different formats between manufacturers.
        """

        if not value:
            return False

        upper = value.upper()

        invalid_values = {
            "SERIAL",
            "NUMBER",
            "NO",
            "NOT_FOUND",
            "FOUND",
            "IMEI",
            "METER",
            "DEVICE",
        }

        if upper in invalid_values:
            return False

        # Require at least one digit.
        if not re.search(
            r"\d",
            value,
        ):
            return False

        # Reject obvious coordinate/measurement values.
        if re.fullmatch(
            r"\d+\.\d+",
            value,
        ):
            return False

        # Reject very short numeric noise.
        if value.isdigit() and len(value) < 4:
            return False

        # Reasonable identifier length.
        if len(value) < 4 or len(value) > 40:
            return False

        return True

    # ------------------------------------------------------------------
    # IMEI
    # ------------------------------------------------------------------

    def _extract_imei(
        self,
        lines: list[str],
    ) -> str:
        """
        Extract an IMEI from OCR lines.
        """

        candidates: list[str] = []

        for line in lines:

            match = re.search(
                self.IMEI_LABEL_PATTERN,
                line,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            remainder = line[
                match.end():
            ].strip()

            remainder = re.sub(
                r"^[\s:;,#=\-]+",
                "",
                remainder,
            )

            # OCR may insert spaces or hyphens inside an IMEI.
            digit_match = re.search(
                r"[\d][\d\s\-]{5,}",
                remainder,
            )

            if not digit_match:
                continue

            candidate = re.sub(
                r"\D",
                "",
                digit_match.group(0),
            )

            if self._is_valid_imei_candidate(
                candidate
            ):
                candidates.append(
                    candidate
                )

        if not candidates:
            return ""

        return self._select_best_imei(
            candidates
        )

    def _is_valid_imei_candidate(
        self,
        value: str,
    ) -> bool:
        """
        Validate an IMEI candidate.

        Standard IMEI values are normally 15 digits.
        We accept 14-16 digits to tolerate OCR errors,
        while rejecting obvious short numeric noise.
        """

        if not value.isdigit():
            return False

        return len(value) in {
            14,
            15,
            16,
        }

    def _select_best_imei(
        self,
        candidates: list[str],
    ) -> str:
        """
        Select the strongest IMEI candidate.
        """

        counts = Counter(
            candidates
        )

        # Prefer exact repeated candidates.
        most_common = counts.most_common()

        if most_common:
            return most_common[0][0]

        return ""

    # ------------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------------

    def _extract_dates(
        self,
        lines: list[str],
    ) -> dict[str, list[str]]:
        """
        Extract labeled dates.
        """

        dates: dict[str, list[str]] = {}

        for index, line in enumerate(lines):

            for category, labels in (
                self.DATE_LABEL_PATTERNS.items()
            ):

                label_found = False

                for label_pattern in labels:

                    if re.search(
                        label_pattern,
                        line,
                        flags=re.IGNORECASE,
                    ):

                        label_found = True
                        break

                if not label_found:
                    continue

                search_text = line

                # If the date is not on the same line,
                # inspect the next OCR line.
                if index + 1 < len(lines):

                    search_text = (
                        f"{line} "
                        f"{lines[index + 1]}"
                    )

                matches = re.findall(
                    self.DATE_PATTERN,
                    search_text,
                    flags=re.IGNORECASE,
                )

                for date_value in matches:

                    if date_value not in dates.get(
                        category,
                        [],
                    ):

                        dates.setdefault(
                            category,
                            [],
                        ).append(
                            date_value
                        )

        return dates

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _select_best_identifier(
        self,
        candidates: list[str],
    ) -> str:
        """
        Select the most reliable serial candidate.
        """

        counts = Counter(
            candidates
        )

        ranked = sorted(
            counts.items(),
            key=lambda item: (
                item[1],
                len(item[0]),
            ),
            reverse=True,
        )

        return ranked[0][0]
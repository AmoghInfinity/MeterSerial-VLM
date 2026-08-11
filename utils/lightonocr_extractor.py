# importing libraries

from __future__ import annotations

import re


class LightOnOCRExtractor:
    """
    Extract a meter serial number from LightOnOCR text.

    This class does not perform OCR or use another ML model.
    It only interprets the OCR text returned by LightOnOCR.
    """

    SERIAL_LABEL_PATTERNS = (
        r"serial\s*(?:number|no\.?)?",
        r"s\.?\s*/\s*n\.?",
        r"sl\s*no\.?",
        r"meter\s*(?:number|no\.?|id)",
    )

    def extract(
        self,
        ocr_text: str,
    ) -> str:
        """
        Extract the serial number from OCR text.
        """

        if not ocr_text:
            return ""

        text = self._normalize_text(
            ocr_text
        )

        lines = text.splitlines()

        for line in lines:

            result = self._extract_from_line(
                line
            )

            if result:
                return result

        return ""

    def _normalize_text(
        self,
        text: str,
    ) -> str:
        """
        Normalize OCR text while preserving line structure.
        """

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        return text.strip()

    def _extract_from_line(
        self,
        line: str,
    ) -> str:
        """
        Search one OCR line for a serial-number label
        followed by an identifier.
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
            (?:
                [:;\-#]?
            )
            \s*
            ([A-Za-z0-9][A-Za-z0-9./_-]{{3,}})
            """,
            re.IGNORECASE | re.VERBOSE,
        )

        match = pattern.search(line)

        if not match:
            return ""

        candidate = match.group(1)

        return self._clean_candidate(
            candidate
        )

    def _clean_candidate(
        self,
        candidate: str,
    ) -> str:
        """
        Clean punctuation introduced by OCR.
        """

        candidate = candidate.strip()

        candidate = candidate.strip(
            ".,:;()[]{}<>\"'`"
        )

        return candidate
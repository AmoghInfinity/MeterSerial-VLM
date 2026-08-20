# importing libraries

from __future__ import annotations

import re
from collections import Counter
from typing import Any


class LightOnOCRConsolidator:
    """
    Consolidate serial number, IMEI and date information
    obtained from multiple LightOnOCR regions/scales.
    """

    DATE_CATEGORIES = (
        "manufacturing",
        "dated",
        "installation",
        "commissioning",
    )

    def __init__(self) -> None:
        pass

    def consolidate(
        self,
        extracted_results: dict[
            str,
            dict[str, Any],
        ],
    ) -> dict[str, Any]:
        """
        Consolidate all extracted OCR results.
        """

        serial_candidates: list[str] = []
        imei_candidates: list[str] = []

        date_candidates: dict[
            str,
            list[str],
        ] = {
            category: []
            for category in self.DATE_CATEGORIES
        }

        for result in extracted_results.values():

            serial_number = str(
                result.get(
                    "serial_number",
                    "",
                )
                or ""
            ).strip()

            if serial_number:
                serial_candidates.append(
                    serial_number
                )

            imei = str(
                result.get(
                    "imei",
                    "",
                )
                or ""
            ).strip()

            if imei:
                imei_candidates.append(
                    imei
                )

            dates = result.get(
                "dates",
                {},
            )

            if not isinstance(
                dates,
                dict,
            ):
                continue

            for category in self.DATE_CATEGORIES:

                values = dates.get(
                    category,
                    [],
                )

                if isinstance(
                    values,
                    str,
                ):
                    values = [values]

                for value in values:

                    value = str(
                        value
                    ).strip()

                    if value:
                        date_candidates[
                            category
                        ].append(
                            value
                        )

        serial_number = self._select_identifier(
            serial_candidates,
            identifier_type="serial",
        )

        imei = self._select_identifier(
            imei_candidates,
            identifier_type="imei",
        )

        dates = self._consolidate_dates(
            date_candidates
        )

        return {
            "serial_number": serial_number,
            "imei": imei,
            "dates": dates,
        }

    # ------------------------------------------------------------------
    # Identifier consolidation
    # ------------------------------------------------------------------

    def _select_identifier(
        self,
        candidates: list[str],
        identifier_type: str,
    ) -> str:
        """
        Select the strongest identifier candidate.
        """

        if not candidates:
            return ""

        cleaned: list[str] = []

        for candidate in candidates:

            value = candidate.strip()

            if not value:
                continue

            if identifier_type == "imei":

                value = re.sub(
                    r"\D",
                    "",
                    value,
                )

                if len(value) not in {
                    14,
                    15,
                    16,
                }:
                    continue

            else:

                value = re.sub(
                    r"[^A-Za-z0-9./_-]",
                    "",
                    value,
                )

                if len(value) < 4:
                    continue

            cleaned.append(
                value
            )

        if not cleaned:
            return ""

        counts = Counter(
            cleaned
        )

        # --------------------------------------------------------------
        # Strongest rule:
        # repeated agreement across OCR regions.
        # --------------------------------------------------------------

        ranked = sorted(
            counts.items(),
            key=lambda item: (
                item[1],
                len(item[0]),
            ),
            reverse=True,
        )

        return ranked[0][0]

    # ------------------------------------------------------------------
    # Date consolidation
    # ------------------------------------------------------------------

    def _consolidate_dates(
        self,
        candidates: dict[
            str,
            list[str],
        ],
    ) -> dict[
        str,
        list[str],
    ]:
        """
        Remove duplicate dates while preserving
        their semantic category.
        """

        result: dict[
            str,
            list[str],
        ] = {}

        for category in self.DATE_CATEGORIES:

            values = candidates.get(
                category,
                [],
            )

            unique_values: list[str] = []

            for value in values:

                value = value.strip()

                if not value:
                    continue

                if value not in unique_values:

                    unique_values.append(
                        value
                    )

            if unique_values:

                result[
                    category
                ] = unique_values

        return result
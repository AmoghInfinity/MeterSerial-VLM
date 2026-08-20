# importing libraries

from __future__ import annotations

from collections import Counter
from typing import Any


class LightOnOCRConsolidator:
    """
    Consolidate structured results produced independently
    from LightOnOCR image regions.
    """

    REGION_PRIORITY = (
        "tile_4",
        "tile_5",
        "tile_1",
        "tile_2",
        "full_image",
        "tile_3",
        "tile_6",
    )

    def consolidate(
        self,
        region_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Consolidate serial number, IMEI and dates.
        """

        return {
            "serial_number": self._consolidate_serial_number(
                region_results
            ),
            "imei": self._consolidate_imei(
                region_results
            ),
            "dates": self._consolidate_dates(
                region_results
            ),
        }

    def _consolidate_serial_number(
        self,
        region_results: dict[str, dict[str, Any]],
    ) -> str:
        """
        Select the strongest serial-number candidate.
        """

        candidates: list[tuple[str, str]] = []

        for region_name, result in region_results.items():

            value = result.get(
                "serial_number",
                "",
            )

            if not value:
                continue

            normalized = self._normalize_serial(
                value
            )

            if normalized:
                candidates.append(
                    (
                        region_name,
                        normalized,
                    )
                )

        return self._select_best_candidate(
            candidates
        )

    def _consolidate_imei(
        self,
        region_results: dict[str, dict[str, Any]],
    ) -> str:
        """
        Select the strongest IMEI candidate.
        """

        candidates: list[tuple[str, str]] = []

        for region_name, result in region_results.items():

            value = result.get(
                "imei",
                "",
            )

            if not value:
                continue

            normalized = self._normalize_imei(
                value
            )

            if normalized:
                candidates.append(
                    (
                        region_name,
                        normalized,
                    )
                )

        return self._select_best_candidate(
            candidates
        )

    def _select_best_candidate(
        self,
        candidates: list[tuple[str, str]],
    ) -> str:
        """
        Select a candidate using frequency first
        and region priority as the tie-breaker.
        """

        if not candidates:
            return ""

        counts = Counter(
            value
            for _, value in candidates
        )

        max_count = max(
            counts.values()
        )

        best_candidates = {
            value
            for value, count in counts.items()
            if count == max_count
        }

        for region_name in self.REGION_PRIORITY:

            for candidate_region, value in candidates:

                if (
                    candidate_region == region_name
                    and value in best_candidates
                ):
                    return value

        return candidates[0][1]

    def _consolidate_dates(
        self,
        region_results: dict[str, dict[str, Any]],
    ) -> dict[str, list[str]]:
        """
        Merge and deduplicate dates across regions.
        """

        consolidated: dict[str, list[str]] = {}

        for result in region_results.values():

            dates = result.get(
                "dates",
                {},
            )

            for category, values in dates.items():

                if not values:
                    continue

                consolidated.setdefault(
                    category,
                    [],
                )

                for value in values:

                    normalized = self._normalize_date(
                        value
                    )

                    if (
                        normalized
                        and normalized
                        not in consolidated[category]
                    ):
                        consolidated[
                            category
                        ].append(
                            normalized
                        )

        return consolidated

    def _normalize_serial(
        self,
        value: str,
    ) -> str:
        """
        Normalize a serial number.
        """

        return "".join(
            value.upper().split()
        ).strip(
            ".,:;()[]{}<>\"'`"
        )

    def _normalize_imei(
        self,
        value: str,
    ) -> str:
        """
        Normalize an IMEI.
        """

        return "".join(
            value.split()
        )

    def _normalize_date(
        self,
        value: str,
    ) -> str:
        """
        Normalize a date.
        """

        return value.strip(
            ".,:;()[]{}<>\"'`"
        )
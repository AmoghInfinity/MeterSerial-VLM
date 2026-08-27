from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from models.backends.paddleocr.paddleocr_backend import (
    PaddleOCRBackend,
)


def main() -> None:

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            'python -m scripts.visualize_tiles '
            '"data/images/dm_29.png"'
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

    # ---------------------------------------------------------
    # Load image.
    # ---------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    width, height = image.size

    # ---------------------------------------------------------
    # Same geometry as PaddleOCR backend.
    # ---------------------------------------------------------

    backend = PaddleOCRBackend()

    backend.tile_rows = 2
    backend.tile_columns = 3
    backend.tile_overlap = 0.35

    tiles = backend._create_tiles(
        image
    )

    # ---------------------------------------------------------
    # Build visualization.
    #
    # We draw approximate tile boundaries based on the same
    # geometry used by the backend.
    # ---------------------------------------------------------

    import math

    rows = backend.tile_rows
    columns = backend.tile_columns
    overlap = backend.tile_overlap

    tile_width = math.ceil(
        width / (
            columns
            - overlap * (columns - 1)
        )
    )

    tile_height = math.ceil(
        height / (
            rows
            - overlap * (rows - 1)
        )
    )

    step_x = int(
        tile_width * (1.0 - overlap)
    )

    step_y = int(
        tile_height * (1.0 - overlap)
    )

    # ---------------------------------------------------------
    # Create canvas.
    # ---------------------------------------------------------

    canvas = image.copy()

    draw = ImageDraw.Draw(
        canvas
    )

    # ---------------------------------------------------------
    # Try to use a readable font.
    # ---------------------------------------------------------

    try:

        font = ImageFont.truetype(
            "arial.ttf",
            32,
        )

    except OSError:

        font = ImageFont.load_default()

    # ---------------------------------------------------------
    # Draw tile rectangles.
    # ---------------------------------------------------------

    tile_number = 1

    for row in range(rows):

        for column in range(columns):

            x1 = column * step_x
            y1 = row * step_y

            x2 = x1 + tile_width
            y2 = y1 + tile_height

            if column == columns - 1:

                x2 = width

                x1 = max(
                    0,
                    x2 - tile_width,
                )

            if row == rows - 1:

                y2 = height

                y1 = max(
                    0,
                    y2 - tile_height,
                )

            # -------------------------------------------------
            # Draw boundary.
            # -------------------------------------------------

            draw.rectangle(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                ),
                outline="red",
                width=5,
            )

            # -------------------------------------------------
            # Draw tile label.
            # -------------------------------------------------

            label = (
                f"Tile {tile_number}"
            )

            bbox = draw.textbbox(
                (0, 0),
                label,
                font=font,
            )

            label_width = (
                bbox[2] - bbox[0]
            )

            label_height = (
                bbox[3] - bbox[1]
            )

            label_x = (
                x1 + 10
            )

            label_y = (
                y1 + 10
            )

            draw.rectangle(
                (
                    label_x - 5,
                    label_y - 5,
                    label_x
                    + label_width
                    + 10,
                    label_y
                    + label_height
                    + 10,
                ),
                fill="white",
            )

            draw.text(
                (
                    label_x,
                    label_y,
                ),
                label,
                fill="black",
                font=font,
            )

            tile_number += 1

    # ---------------------------------------------------------
    # Save output.
    # ---------------------------------------------------------

    output_path = (
        image_path.parent
        / (
            image_path.stem
            + "_tiles.png"
        )
    )

    canvas.save(
        output_path
    )

    print()
    print(
        "=" * 70
    )

    print(
        "TILE VISUALIZATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Original size : "
        f"{width} x {height}"
    )

    print(
        f"Rows          : {rows}"
    )

    print(
        f"Columns       : {columns}"
    )

    print(
        f"Overlap       : "
        f"{overlap * 100:.0f}%"
    )

    print(
        f"Tile size     : "
        f"{tile_width} x {tile_height}"
    )

    print(
        f"Tile count    : "
        f"{len(tiles)}"
    )

    print()

    print(
        f"Saved to      : "
        f"{output_path}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
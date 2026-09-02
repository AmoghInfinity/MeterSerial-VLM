# MeterSerial-VLM

OCR pipeline for extracting **electricity-meter Serial Numbers and IMEI
numbers** from digital-meter photographs.

The project separates OCR from field extraction and consolidation so
that different OCR engines can be used with the same downstream logic.

## Architecture

``` text
Meter Image
    |
    +-------------------+
    |                   |
    v                   v
LightOnOCR          PaddleOCR
    |                   |
    +--------+----------+
             |
             v
      Universal Extractor
             |
             v
      Consensus Consolidator
             |
             v
       Serial Number + IMEI
```

### Current OCR strategy

The current pipeline runs **7 OCR regions per image**:

-   1 full image
-   6 overlapping tiles
-   2 x 3 tile arrangement
-   edge padding is supported

The tile strategy is currently kept simple so its behavior can be
evaluated before further geometry changes.

------------------------------------------------------------------------

## Repository layout

``` text
MeterSerial-VLM/
|
+-- data/
|   +-- images/                 # Meter images (not normally committed)
|
+-- model_store/
|   +-- lightonocr/             # Local LightOnOCR checkpoint
|
+-- models/
|   +-- base/
|   |   +-- base_model.py
|   +-- registry.py
|   +-- backends/
|       +-- lightonocr/
|       |   +-- lightonocr_backend.py
|       +-- paddleocr/
|           +-- paddleocr_backend.py
|
+-- utils/
|   +-- universal extractor     # Model-independent field extraction
|   +-- universal consolidator  # Candidate ranking/consensus
|
+-- scripts/
|   +-- test_lightonocr.py
|   +-- test_paddleocr.py
|   +-- visualize_tiles.py
|
+-- .venv/                       # LightOnOCR environment
+-- .paddlevenv/                 # PaddleOCR environment
+-- README.md
```

> Utility filenames can differ between revisions; keep the extractor and
> consolidator independent of the OCR backend.

------------------------------------------------------------------------

## Requirements

### Recommended hardware

A CUDA-capable NVIDIA GPU is strongly recommended. The development
system used an NVIDIA GeForce RTX 5050 with about 8 GB VRAM.

CPU execution may be possible for some components but is substantially
slower.

### Python

The development setup uses separate Python environments:

-   **Python 3.13** for LightOnOCR
-   **Python 3.11** for PaddleOCR

Using separate environments is important because PaddlePaddle/PaddleX
dependencies can conflict with the Transformers/LightOnOCR stack.

------------------------------------------------------------------------

# Installation

## 1. Clone the repository

``` powershell
git clone <repository-url>
cd MeterSerial-VLM
```

Keep the large image dataset and model weights outside Git unless Git
LFS or another large-file mechanism is being used.

## 2. LightOnOCR environment

``` powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the project's LightOnOCR dependencies. If the repository
provides a requirements file:

``` powershell
python -m pip install -r requirements.txt
```

Place the local LightOnOCR checkpoint under:

``` text
model_store/lightonocr/
```

The backend loads the checkpoint with the LightOnOCR processor/model
classes.

Test it with:

``` powershell
python -m scripts.test_lightonocr "data/images/dm_2.png"
```

## 3. PaddleOCR environment

Create the isolated environment:

``` powershell
py -3.11 -m venv .paddlevenv
.\.paddlevenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the appropriate **PaddlePaddle GPU build** and PaddleOCR version
for the target Python version, CUDA runtime, GPU and operating system.
Follow the official Paddle installation instructions for the target
machine rather than copying a GPU wheel from another CUDA configuration.

Verify Paddle:

``` powershell
python -c "import paddle; print('Paddle:', paddle.__version__); print('CUDA:', paddle.device.is_compiled_with_cuda()); print('Devices:', paddle.device.get_available_device())"
```

Run Paddle's self-check:

``` powershell
python -c "import paddle; paddle.utils.run_check()"
```

Verify PaddleOCR:

``` powershell
python -c "import paddleocr; print('PaddleOCR:', paddleocr.__version__)"
```

On the first OCR run, PaddleOCR/PaddleX may download and cache its model
files under a user directory similar to:

``` text
C:\Users\<USERNAME>\.paddlex\official_models\
```

The development setup used PP-OCRv6 detection and recognition models.

------------------------------------------------------------------------

# Running the pipeline

## PaddleOCR

Activate the PaddleOCR environment:

``` powershell
.\.paddlevenv\Scripts\Activate.ps1
```

Run an image:

``` powershell
python -m scripts.test_paddleocr "data/images/dm_2.png"
```

The output is intentionally divided into three stages:

1.  **PADDLEOCR RAW OUTPUT** --- exactly what the OCR stage read.
2.  **PYTHON EXTRACTION RESULTS** --- Serial Number/IMEI candidates
    found by the universal extractor.
3.  **FINAL CONSOLIDATED RESULT** --- candidates combined by the
    consolidator.

## LightOnOCR

Activate the LightOnOCR environment:

``` powershell
.\.venv\Scripts\Activate.ps1
```

Run:

``` powershell
python -m scripts.test_lightonocr "data/images/dm_2.png"
```

The same general stages are used:

``` text
image -> LightOnOCR -> raw text -> extractor -> consolidator -> final result
```

------------------------------------------------------------------------

# Identifier extraction

The downstream extraction layer is deliberately **model-independent**.

## Serial Number

Common labels handled by the extractor include variations of:

``` text
Serial Number
Serial No
S/N
S. No.
SL NO
SL. NO.
SL.
Meter Number
Meter No
Meter ID
Device ID
```

OCR can split a label/value across lines, for example:

``` text
SL.
NO.
U5020434
```

or:

``` text
SL.
U5020434
NO.
```

The extractor therefore searches surrounding OCR context instead of
assuming that the label and value must be on one line.

A generic `NO.` should not automatically become a Serial Number; context
and candidate validation are required to reduce false positives.

## IMEI

Common forms include:

``` text
IMEI
IMEI NO
IMEI NO:
IMEI NUMBER
IMEI:
```

The extractor validates numeric candidates rather than treating every
long number on a meter as an IMEI.

------------------------------------------------------------------------

# Consolidation

Different OCR regions can produce different versions of the same
identifier.

For example:

``` text
full_image -> U5020434
tile_1     -> U5020
tile_2     -> U5020434
tile_3     -> NOT_FOUND
```

The consolidator should prefer the complete, valid and repeatedly
supported candidate:

``` text
U5020434
```

The same principle applies to IMEI values. A truncated candidate should
not override a complete candidate when the complete candidate is
supported elsewhere.

The system should prefer `NOT_FOUND` over inventing an identifier.

------------------------------------------------------------------------

# Tile processing

The current implementation uses:

``` text
1 full image
+
6 overlapping tiles
=
7 OCR regions
```

The six tiles form a 2 x 3 layout:

``` text
+----------+----------+----------+
| Tile 1   | Tile 2   | Tile 3   |
+----------+----------+----------+
| Tile 4   | Tile 5   | Tile 6   |
+----------+----------+----------+
```

Tiles use overlap so text near a boundary can be captured by more than
one OCR region. Edge padding prevents important content from being
placed directly against the crop boundary.

Padding does not recover pixels that were never included in a crop;
overlap is what provides another opportunity to capture
boundary-adjacent text.

## Visualize tile geometry

``` powershell
python -m scripts.visualize_tiles "data/images/dm_29.png"
```

This generates a tile visualization such as:

``` text
data/images/dm_29_tiles.png
```

Use this when investigating whether a Serial Number or IMEI is close to
a tile boundary.

------------------------------------------------------------------------

# Recommended test images

Use a small representative set before processing the entire dataset:

``` text
dm_1.png
dm_2.png
dm_23.png
dm_29.png
```

Example:

``` powershell
python -m scripts.test_paddleocr "data/images/dm_1.png"
python -m scripts.test_paddleocr "data/images/dm_2.png"
python -m scripts.test_paddleocr "data/images/dm_23.png"
python -m scripts.test_paddleocr "data/images/dm_29.png"
```

Always inspect the raw OCR before deciding that extraction failed. An
OCR engine can read the correct value while the extractor or
consolidator selects the wrong candidate.

------------------------------------------------------------------------

# Troubleshooting

## PaddleOCR rejects PIL images

If PaddleOCR reports:

``` text
Not supported input data type!
Only `numpy.ndarray` and `str` are supported!
```

convert the PIL image before calling PaddleOCR:

``` python
import numpy as np

image_array = np.asarray(image)
```

Pass `image_array` to the OCR pipeline.

## Paddle oneDNN/PIR runtime errors

For errors such as:

``` text
ConvertPirAttribute2RuntimeAttribute not support
```

check:

``` powershell
python -c "import paddle; print(paddle.__version__)"
python -c "import paddle; print(paddle.device.is_compiled_with_cuda())"
python -c "import paddle; print(paddle.device.get_available_device())"
python -c "import paddle; paddle.utils.run_check()"
```

Make sure the PaddlePaddle build matches the target Python/CUDA/GPU
configuration. Keep PaddleOCR isolated in `.paddlevenv` while
troubleshooting.

## PaddleOCR is using CPU

Run:

``` powershell
python -c "import paddle; print('CUDA:', paddle.device.is_compiled_with_cuda()); print('Devices:', paddle.device.get_available_device())"
```

A working GPU build should report CUDA enabled and a GPU device such as
`gpu:0`.

## LightOnOCR loading warnings

Transformers may display a model-type warning when loading the
LightOnOCR checkpoint. Judge the installation by whether the intended
checkpoint loads and inference completes correctly, and verify the
actual OCR output.

------------------------------------------------------------------------

# Git hygiene

Do not normally commit:

``` text
.venv/
.paddlevenv/
model_store/
.paddlex/
data/images/
large model weights
```

Suggested `.gitignore` entries:

``` gitignore
__pycache__/
*.py[cod]
.venv/
.paddlevenv/
model_store/
.paddlex/
*.safetensors
*.bin
*.pt
*.pth
*.onnx
data/images/
*_tiles.png
.vscode/
.idea/
.DS_Store
Thumbs.db
```

Use Git LFS or external storage for large datasets/models when they need
to be distributed.

------------------------------------------------------------------------

# New-machine checklist

``` text
1. Clone repository
2. Install Python 3.13
3. Create .venv
4. Install LightOnOCR dependencies
5. Place/download LightOnOCR checkpoint
6. Install Python 3.11
7. Create .paddlevenv
8. Install compatible PaddlePaddle GPU build
9. Install PaddleOCR
10. Verify Paddle GPU
11. Run one PaddleOCR image test
12. Run one LightOnOCR image test
13. Run extractor/consolidator tests
```

Do not combine the two OCR environments unless dependency compatibility
has been explicitly verified.

------------------------------------------------------------------------

# Expected output

A successful run should ultimately produce something like:

``` text
======================================================================
                    FINAL CONSOLIDATED RESULT
======================================================================

Serial Number : U5020434
IMEI          : 862287074334863

======================================================================
```

If an identifier cannot be reliably determined:

``` text
Serial Number : NOT_FOUND
IMEI          : NOT_FOUND
```

The pipeline should prefer `NOT_FOUND` over guessing.

------------------------------------------------------------------------

# Design principles

-   **Model independence:** OCR backends produce text; extraction and
    consolidation are separate.
-   **Manufacturer independence:** Serial Number and IMEI locations are
    not assumed to be fixed.
-   **Multiple observations:** Full-image and tiled OCR provide multiple
    opportunities to recognize small text.
-   **Conservative identification:** Do not invent or guess identifiers.
-   **Environment isolation:** LightOnOCR and PaddleOCR use separate
    virtual environments.
-   **Incremental improvement:** Diagnose OCR, extraction and
    consolidation independently before changing all three at once.

------------------------------------------------------------------------

# Future improvements

Possible future work includes:

-   improved multi-scale OCR
-   improved tile geometry and overlap
-   stronger edge handling
-   text-detection-based high-resolution crops
-   OCR confidence scoring
-   stronger candidate ranking
-   automated regression testing over the full dataset
-   batch inference
-   meter-specific OCR fine-tuning
-   benchmarking additional OCR backends

These should be evaluated against the current pipeline before being
introduced.

------------------------------------------------------------------------

## Summary

MeterSerial-VLM is a reusable electricity-meter identification pipeline:

``` text
Electricity Meter Image
          |
          +----------------------+
          |                      |
          v                      v
     LightOnOCR              PaddleOCR
          |                      |
          +----------+-----------+
                     |
                     v
                Raw OCR Text
                     |
                     v
             Universal Extractor
                  /       \
                 /         \
        Serial Number      IMEI
                 \         /
                  \       /
                   v     v
                 Consolidator
                     |
                     v
              Final Serial + IMEI
```

The architecture is intentionally designed so another OCR model can be
added later without rewriting the core identifier extraction and
consolidation logic.

# MeterSerial-VLM

A modular, model-independent OCR pipeline for extracting information from digital electricity-meter images.

The primary objective is to reliably extract:

- **Meter Serial Number**
- **IMEI**
- Other text fields that can be added later
- Meter **ON/OFF** status based on the visible green backlight/display

The project is designed so that different OCR/VLM models can be added or replaced without rewriting the extraction and consolidation logic.

---

## 1. Project Goals

MeterSerial-VLM is designed around the following principles:

1. Support multiple OCR models through a common interface.
2. Keep model-specific code isolated inside backend modules.
3. Use a **universal extractor** independent of the OCR model.
4. Use a **universal consolidator** to combine OCR results from multiple image regions.
5. Avoid hardcoding meter manufacturer, model, or location.
6. Return `NOT_FOUND` instead of guessing an identifier.
7. Load models lazily so unused models do not consume GPU memory.
8. Make it easy to add future OCR/VLM backends.

---

## 2. OCR Processing Strategy

For every input meter image, the current pipeline processes:

- **1 full image**
- **6 overlapping tiles**
- **7 OCR regions in total**

The tiles follow a **2 × 3 overlapping grid**.

Edge padding is applied to the tiles to reduce the possibility of cutting characters at tile boundaries.

This seven-region strategy is the current baseline and should not be changed unless intentionally evaluated against the existing approach.

---

## 3. Architecture

```text
                  Meter Image
                       |
                       v
          +-------------------------+
          | Image / Tile Generation |
          | 1 Full + 6 Tiles        |
          +-------------------------+
                       |
                       v
          +-------------------------+
          |      OCR Backend        |
          +-------------------------+
             /        |                     /         |                     v          v           v
     LightOnOCR    PaddleOCR    Future Models
           \          |           /
            \         |          /
             +--------+----------+
                      |
                      v
              Raw OCR Results
                      |
                      v
          +-----------------------+
          | Universal Extractor   |
          +-----------------------+
                      |
                      v
          +-----------------------+
          | Universal Consolidator|
          +-----------------------+
                      |
                      v
             Final Structured Data
```

The key principle is:

> **OCR models can change; extraction and consolidation should not have to.**

---

## 4. Repository Structure

The project is organized approximately as follows:

```text
MeterSerial-VLM/
│
├── data/
│   └── images/
│       └── *.png
│
├── models/
│   ├── base/
│   │   └── base_model.py
│   │
│   ├── backends/
│   │   ├── lightonocr/
│   │   │   └── lightonocr_backend.py
│   │   │
│   │   └── paddleocr/
│   │       └── paddleocr_backend.py
│   │
│   └── registry.py
│
├── scripts/
│   ├── test_lightonocr.py
│   ├── test_paddleocr.py
│   └── visualize_tiles.py
│
├── model_store/
│   └── lightonocr/
│
├── requirements.txt
├── requirements-paddle.txt
└── README.md
```

Additional utility modules may be present depending on the current revision of the repository.

---

# 5. Requirements

## Hardware

GPU execution is recommended, especially for VLM-based OCR models.

The development system uses an NVIDIA GPU with approximately **8 GB VRAM**.

The amount of available VRAM can affect which models can be loaded and whether multiple models can run simultaneously.

CPU execution may be possible for some components, but inference performance will be significantly lower.

---

# 6. Environment Structure

The project intentionally maintains separate Python environments for the OCR systems.

```text
MeterSerial-VLM/
│
├── .venv/
│     └── LightOnOCR environment
│
└── .paddlevenv/
      └── PaddleOCR environment
```

This separation is important because the two OCR stacks have different dependencies.

---

# 7. LightOnOCR Environment

LightOnOCR is installed in the main `.venv` environment.

## Create Environment

From the project root:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

```powershell
python -m pip install -U huggingface_hub
pip install qwen-vl-utils

hf download lightonai/LightOnOCR-2-1B --local-dir model_store/lightonocr

hf download Qwen/Qwen2-VL-2B-Instruct --local-dir model_store/qwen2-vl-2b

hf download OpenGVLab/InternVL2_5-4B --local-dir model_store/internvl2_5-4b
```

## Model

The LightOnOCR model used by the project is:

```text
lightonai/LightOnOCR-2-1B
```

The model can be stored locally under:

```text
model_store/lightonocr/
```

The LightOnOCR backend handles:

- Model loading
- Processor initialization
- Input preparation
- OCR inference
- Conversion of model output into the common OCR representation

Model-specific logic should remain inside the backend.

---

# 8. PaddleOCR Environment

PaddleOCR is maintained in a **separate environment** from LightOnOCR.

## Python Version

The PaddleOCR environment uses:

```text
Python 3.11
```

Create it using:

```powershell
py -3.11 -m venv .paddlevenv
```

Activate it:

```powershell
.\.paddlevenv\Scripts\Activate.ps1
```

Verify the Python version:

```powershell
python --version
```

Expected:

```text
Python 3.11.x
```

## PaddleOCR Versions

The working PaddleOCR setup uses:

```text
PaddlePaddle  3.3.1
PaddleOCR     3.7.0
PaddleX       3.7.2
```

The direct project dependencies are:

```text
paddlepaddle==3.3.1
paddleocr==3.7.0
paddlex==3.7.2
```

Recommended `requirements-paddle.txt`:

```text
paddlepaddle==3.3.1
paddleocr==3.7.0
paddlex==3.7.2
```

Install using:

```powershell
python -m pip install -r requirements-paddle.txt
```

### Supporting Dependencies

The PaddleOCR/PaddleX installation also installs supporting packages, including:

```text
numpy
opencv-contrib-python
PyYAML
pydantic
pypdfium2
pyclipper
python-bidi
pycryptodome
shapely
requests
aiohttp
modelscope
aistudio-sdk
prettytable
ujson
ruamel.yaml
py-cpuinfo
colorlog
```

along with their required transitive dependencies.

These do not need to be manually listed if they are installed automatically by the pinned direct dependencies.

## Verify PaddlePaddle

```powershell
python -c "import paddle; print('Paddle:', paddle.__version__); print('CUDA:', paddle.device.is_compiled_with_cuda()); print('Devices:', paddle.device.get_available_device())"
```

Verify PaddleOCR:

```powershell
python -c "import paddleocr; print('PaddleOCR:', paddleocr.__version__)"
```

Run PaddlePaddle's built-in check:

```powershell
python -c "import paddle; paddle.utils.run_check()"
```

A correctly configured GPU environment should report CUDA support and an available GPU device.

## Important CUDA / OS Note

The exact PaddlePaddle GPU installation depends on the target machine's:

- Operating system
- Python version
- NVIDIA driver
- CUDA compatibility
- GPU architecture

Do **not** blindly copy a GPU installation command from another operating system.

The versions documented above represent the project's working PaddleOCR environment. When deploying to another machine, use a PaddlePaddle build compatible with that machine while maintaining the separate PaddleOCR environment.

---

# 9. Complete Installation

Clone the repository:

```powershell
git clone <repository-url>
cd MeterSerial-VLM
```

## LightOnOCR

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## PaddleOCR

Create the environment specifically with Python 3.11:

```powershell
py -3.11 -m venv .paddlevenv
```

Activate:

```powershell
.\.paddlevenv\Scripts\Activate.ps1
```

Install:

```powershell
python -m pip install -r requirements-paddle.txt
```

Verify:

```powershell
python --version
```

It should report:

```text
Python 3.11.x
```

---

# 10. Running PaddleOCR

Activate the PaddleOCR environment:

```powershell
.\.paddlevenv\Scripts\Activate.ps1
```

Run the PaddleOCR test:

```powershell
python scripts/test_paddleocr.py
```

The script runs OCR against the configured meter image.

The current pipeline processes the full image and six overlapping tiles.

Raw OCR results can then be inspected before extraction and consolidation.

---

# 11. Running LightOnOCR

Activate the LightOnOCR environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run:

```powershell
python scripts/test_lightonocr.py
```

The LightOnOCR backend is responsible for model-specific loading and inference.

---

# 12. Visualizing the OCR Tiles

To inspect the current seven-region processing strategy:

```powershell
python scripts/visualize_tiles.py
```

This allows you to verify:

- Full-image processing
- Six tile regions
- Overlap between tiles
- Edge padding
- Location of important meter text

Generated visualization files may be saved under:

```text
data/images/
```

---

# 13. OCR Backend Design

Each OCR model is implemented as a separate backend.

Conceptually:

```text
Base Model
    |
    +---- LightOnOCR Backend
    |
    +---- PaddleOCR Backend
    |
    +---- Future OCR/VLM Backend
```

A backend is responsible for:

1. Loading its model.
2. Preparing model-specific inputs.
3. Running inference.
4. Returning OCR results in the common format.

The backend should **not** contain meter-specific extraction rules.

For example, avoid logic such as:

```python
if manufacturer == "XYZ":
    ...
```

or:

```python
if model == "ABC":
    ...
```

This keeps the system independent of any particular meter manufacturer or model.

---

# 14. Universal Extractor

The extractor is intentionally independent of the OCR backend.

It receives OCR results and identifies fields such as:

```text
Serial Number
IMEI
Date
Other future fields
```

Different OCR models may produce different textual formats.

For example:

```text
U5028045
```

```text
SL. NO. U5028045
```

```text
Serial No: U5028045
```

The extractor should recognize the underlying value without requiring model-specific rules.

---

# 15. Universal Consolidator

Because every image is processed through seven OCR regions, multiple observations of the same value can be produced.

For example:

```text
Full Image:
    U5028045

Tile 1:
    U5028045

Tile 2:
    U5028045

Tile 3:
    U50280
```

The consolidator combines these observations and selects the strongest consistent candidate.

It should:

- Combine results from multiple regions.
- Handle duplicate values.
- Handle partial OCR results.
- Prefer consistent candidates.
- Reject unrelated candidates.
- Avoid selecting a value simply because it appeared first.
- Return `NOT_FOUND` when the evidence is insufficient.

---

# 16. Identifier Extraction Safety

Meter images contain many unrelated numbers.

Examples include:

- Voltage readings
- Current readings
- Energy values
- Dates
- Model numbers
- Codes
- Serial numbers
- IMEI numbers

Therefore, generic patterns such as searching for any number following `NO` can result in incorrect extraction.

The extractor should use contextual evidence and candidate validation.

If the available OCR evidence is insufficient:

```text
NOT_FOUND
```

should be returned instead of guessing.

This is particularly important for meter serial numbers and IMEI values, where a plausible but incorrect value is worse than a missing value.

---

# 17. Example Final Output

A successful extraction may look like:

```json
{
    "serial_number": "U5028045",
    "imei": "860738079449140"
}
```

If a value cannot be confidently identified:

```json
{
    "serial_number": "NOT_FOUND",
    "imei": "860738079449140"
}
```

The exact output structure should follow the current implementation of the extractor/consolidator.

---

# 18. Meter ON / OFF Classification

The project can also determine whether a meter is ON or OFF based on its visible display/backlight.

Current definition:

```text
ON  = green backlight/display is visibly illuminated
OFF = green backlight/display is not visibly illuminated
```

This classification is separate from OCR.

```text
OCR
  |
  +---- Serial Number
  +---- IMEI
  +---- Other text

Image Classification
  |
  +---- ON / OFF
```

The ON/OFF component can be replaced or extended independently in the future.

---

# 19. Adding a New OCR/VLM Model

To add a new model:

### Step 1

Create a backend directory:

```text
models/backends/<new_model>/
```

### Step 2

Implement the common model interface.

### Step 3

Add model-specific:

- Loading
- Preprocessing
- Inference
- Output conversion

### Step 4

Register the backend with the model registry/factory if required.

### Step 5

Add a test script.

For example:

```text
models/
└── backends/
    └── new_model/
        └── new_model_backend.py
```

The universal extractor and consolidator should not need to be rewritten simply because a new OCR model was added.

---

# 20. Lazy Model Loading

Models should be loaded only when requested.

Conceptually:

```text
Application starts
       |
       v
No OCR model loaded
       |
       +----------------------+
       |                      |
       v                      v
 Request PaddleOCR       Request LightOnOCR
       |                      |
       v                      v
 Load PaddleOCR          Load LightOnOCR
```

This is important because VLM/OCR models can consume significant GPU memory.

The model registry/factory should therefore avoid loading every model during application startup.

---

# 21. Dataset

Input meter images are stored under:

```text
data/images/
```

Example:

```text
data/images/
├── dm_1.png
├── dm_2.png
├── dm_3.png
├── ...
└── dm_1999.png
```

The current dataset contains approximately **2,000 meter images**.

The same image-processing pipeline should be used when comparing different OCR models so that model comparisons remain meaningful.

---

# 22. Model Files and Git

Large model weights generally should not be committed directly to the Git repository.

Depending on the project distribution strategy, folders such as:

```text
.venv/
.paddlevenv/
__pycache__/
*.pyc
model_store/
```

may be added to `.gitignore`.

If model weights are required for deployment, they can instead be distributed through an appropriate model/artifact storage system.

---

# 23. Troubleshooting

## PowerShell Activation Error

If PowerShell blocks the virtual-environment activation script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

---

## PaddleOCR Is Using the Wrong Python

Check:

```powershell
python --version
```

The PaddleOCR environment must use:

```text
Python 3.11.x
```

If the environment was accidentally created using another Python version, recreate it:

```powershell
deactivate
Remove-Item -Recurse -Force .paddlevenv
```

Create it again:

```powershell
py -3.11 -m venv .paddlevenv
.\.paddlevenv\Scripts\Activate.ps1
```

---

## Paddle Cannot Detect the GPU

Run:

```powershell
python -c "import paddle; print(paddle.device.is_compiled_with_cuda()); print(paddle.device.get_available_device())"
```

If CUDA is unavailable, check:

1. NVIDIA driver installation.
2. Active Python environment.
3. Installed PaddlePaddle build.
4. Python version.
5. CUDA compatibility.
6. GPU compatibility.

Make sure the command is being executed inside:

```text
.paddlevenv
```

and not:

```text
.venv
```

---

## PaddleOCR Downloads Models

PaddleOCR/PaddleX may download model files the first time a particular OCR model is used.

These models are normally cached locally by PaddleX.

---

## OCR Produces Incomplete Text

The seven-region strategy is intended to improve coverage.

Inspect:

```text
Full Image OCR
Tile 1 OCR
Tile 2 OCR
Tile 3 OCR
Tile 4 OCR
Tile 5 OCR
Tile 6 OCR
```

Also inspect the tile visualization.

Before changing the tile strategy, determine whether the missing characters are visible in another region.

---

# 24. Development Guidelines

## Keep Model-Specific Code Inside Backends

Use:

```text
models/backends/paddleocr/
models/backends/lightonocr/
```

for model-specific code.

Avoid putting PaddleOCR-specific or LightOnOCR-specific logic into the universal extractor.

## Keep Extraction Model-Independent

The extractor should be able to work with:

```text
PaddleOCR
LightOnOCR
Future OCR models
Future VLMs
```

## Prefer `NOT_FOUND` Over Guessing

For identifiers such as Serial Number and IMEI:

```text
Correct value > NOT_FOUND > guessed value
```

A wrong identifier can be more harmful than a missing identifier.

## Preserve the Current OCR Baseline

The current baseline is:

```text
1 full image + 6 overlapping tiles
```

Changes to preprocessing should be evaluated rather than made solely to accommodate one OCR model.

---

# 25. Quick Start

## PaddleOCR

```powershell
git clone <repository-url>
cd MeterSerial-VLM

py -3.11 -m venv .paddlevenv
.\.paddlevenv\Scripts\Activate.ps1

python -m pip install -r requirements-paddle.txt

python scripts/test_paddleocr.py
```

## LightOnOCR

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt

python scripts/test_lightonocr.py
```

---

# 26. Environment Summary

| Component | Environment | Python | Version |
|---|---|---:|---:|
| LightOnOCR | `.venv` | Project environment | See `requirements.txt` |
| PaddleOCR | `.paddlevenv` | **3.11** | **3.7.0** |
| PaddlePaddle | `.paddlevenv` | **3.11** | **3.3.1** |
| PaddleX | `.paddlevenv` | **3.11** | **3.7.2** |

The two OCR environments are intentionally isolated:

```text
.venv
    └── LightOnOCR

.paddlevenv
    └── PaddleOCR
        ├── Python 3.11
        ├── PaddlePaddle 3.3.1
        ├── PaddleOCR 3.7.0
        └── PaddleX 3.7.2
```

---

# 27. Final Pipeline

```text
                         Meter Image
                              |
                              v
              +-----------------------------+
              | 1 Full Image + 6 Tiles      |
              | 7 OCR Regions               |
              +-----------------------------+
                              |
                              v
                     OCR Backend Layer
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
        LightOnOCR         PaddleOCR       Future VLMs
             |                |                |
             +----------------+----------------+
                              |
                              v
                       Raw OCR Results
                              |
                              v
                   Universal Extractor
                              |
                              v
                  Universal Consolidator
                              |
                              v
             +-------------------------------+
             | Serial Number                  |
             | IMEI                           |
             | Other Extracted Fields         |
             +-------------------------------+
                              |
                              v
                    Image Classification
                              |
                              v
                           ON / OFF
```

---

# 28. Core Design Principle

> **The OCR model is replaceable; the extraction and consolidation pipeline should remain model-independent.**

This allows the same meter-image processing pipeline to evaluate different OCR/VLM models while maintaining consistent downstream extraction behavior.

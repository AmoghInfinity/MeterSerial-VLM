# MeterSerial-VLM

A lightweight vision-language system for automatic electricity meter serial-number recognition.

Electricity meter serial numbers can appear in different locations across manufacturers, meter models, and physical designs. **MeterSerial-VLM** addresses this problem using full-image and generic tiled inference with multiple locally hosted OCR/VLM models.

## Models

The current implementation supports:

- **Qwen2-VL-2B**
- **LightOnOCR-2-1B**
- **InternVL2.5-4B**

Each model is evaluated and executed independently.

## Architecture

```text
                         Meter Image
                              |
                              v
                 Full Image + Generic Tiles
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
         Qwen2-VL        LightOnOCR         InternVL
             |                |                |
             v                v                v
        Serial Number     OCR Output       Serial Number
                              |
                              v
                    Deterministic OCR
                       Extraction
                              |
                              v
                        Serial Number
```

The system processes the complete image together with six generic overlapping image regions, resulting in seven inference regions per image.

The same task prompt is used across the three models.

LightOnOCR uses a deterministic text-processing layer because it can return surrounding OCR text instead of consistently returning only the requested identifier.

No model receives the output of another model.

---

## Key Features

- Location-independent serial-number detection
- Full-image + tiled inference
- Multiple interchangeable OCR/VLM backends
- Local model execution
- Lazy model loading
- Model registry
- GPU-aware inference
- Common serial-number recognition prompt
- Deterministic LightOnOCR OCR extraction
- No hardcoded meter layouts
- No model-to-model inference
- Modular model backend architecture

---

## Current Inference Strategy

Each input meter image is processed using:

```text
1 × Full Image
+
6 × Generic Overlapping Tiles
=
7 Image Regions
```

The approach does not assume that the serial number is located at a predefined coordinate.

```text
                  Full Meter Image
                         |
          +--------------+--------------+
          |              |              |
       Tile 1         Tile 2         Tile 3
          |              |              |
          +--------------+--------------+
          |              |              |
       Tile 4         Tile 5         Tile 6
```

This allows the system to detect serial numbers placed in different areas of different meter designs.

---

## Project Structure

```text
MeterSerial-VLM/
│
├── data/
│   ├── images/
│   └── README.md
│
├── model_store/
│   └── README.md
│
├── models/
│   ├── backends/
│   │   ├── qwen2_vl/
│   │   ├── lightonocr/
│   │   └── internvl/
│   │
│   ├── base/
│   └── registry.py
│
├── prompts/
│   └── meter_serial_prompt.txt
│
├── scripts/
│   ├── test_qwen_inference.py
│   ├── test_lightonocr.py
│   ├── test_lightonocr_extractor.py
│   ├── test_internvl.py
│   └── ...
│
├── utils/
│   ├── model_manager.py
│   ├── lightonocr_extractor.py
│   ├── transformers_compat.py
│   └── ...
│
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── requirements-full.txt
```

---

## Environment

Current development environment:

- Windows 11
- Python 3.13
- NVIDIA RTX 5050 Laptop GPU
- CUDA-enabled PyTorch
- Transformers 5.x
- Hugging Face Hub

### Python Version

The project uses Python 3.13.

Create the virtual environment:

```powershell
py -3.13 -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Verify Python:

```powershell
python --version
```

---

## Installation

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the project dependencies:

```powershell
python -m pip install -r requirements-full.txt
```

The repository includes `requirements-full.txt` as the environment snapshot used during development.

---

## Model Storage

Model weights are downloaded locally and are intentionally excluded from Git.

Expected local structure:

```text
model_store/
├── qwen2_vl/
├── lightonocr/
└── internvl/
```

The model manager handles local model paths and lazy loading.

Models are loaded into GPU memory only when the corresponding model is selected.

This prevents all three models from occupying GPU memory simultaneously.

---

## Model Management

The project uses a model registry and model manager.

```text
Model Registry
      |
      +---- Qwen2-VL
      |
      +---- LightOnOCR
      |
      +---- InternVL
              |
              v
        Model Manager
              |
              v
       Local Model Store
```

Models can therefore be loaded and unloaded independently.

---

## Running Inference

### Qwen2-VL

```powershell
python -m scripts.test_qwen_inference "data/images/dm_1.png"
```

### LightOnOCR

```powershell
python -m scripts.test_lightonocr "data/images/dm_1.png"
```

LightOnOCR produces OCR text from each region. A deterministic OCR-specific extractor identifies values associated with serial-number labels.

### LightOnOCR Extractor Test

```powershell
python -m scripts.test_lightonocr_extractor
```

### InternVL2.5

```powershell
python -m scripts.test_internvl "data/images/dm_1.png"
```

InternVL uses its native image preprocessing and `model.chat()` inference interface.

---

## Common Prompt

The three models use the same task-level prompt.

The prompt is designed to:

- search the entire meter image
- avoid assuming a fixed serial-number location
- recognize common serial-number labels
- distinguish serial numbers from IMEI, model numbers, meter readings, and other identifiers
- return the complete identifier
- return `NOT_FOUND` when the serial number cannot be identified

The prompt is stored at:

```text
prompts/meter_serial_prompt.txt
```

---

## Model Independence

Each model operates independently.

```text
                  Meter Image
                       |
       +---------------+---------------+
       |               |               |
       v               v               v
    Qwen2-VL      LightOnOCR       InternVL
       |               |               |
       v               v               v
    Result          OCR Text        Result
                       |
                       v
                OCR Extraction
                       |
                       v
                    Result
```

There is no model-to-model inference or cross-model extraction.

---

## Development Status

### Completed

- [x] Python 3.13 environment
- [x] CUDA-enabled PyTorch
- [x] RTX 5050 GPU validation
- [x] Local model storage
- [x] Lazy model loading
- [x] Model registry
- [x] Model manager
- [x] Qwen2-VL backend
- [x] LightOnOCR backend
- [x] InternVL2.5 backend
- [x] Full-image inference
- [x] Generic tiled inference
- [x] Common serial-number prompt
- [x] LightOnOCR deterministic extractor
- [x] Local model loading and unloading
- [x] Initial meter-image validation

### In Progress

- [ ] Independent model benchmarking
- [ ] Accuracy evaluation on larger meter-image dataset
- [ ] Latency comparison
- [ ] GPU memory comparison
- [ ] Production inference interface
- [ ] vLLM inference path
- [ ] Application UI

---

## Design Principles

### Generic over hardcoded

The system does not assume that a serial number appears at a fixed location.

### Model independence

Each model must solve the recognition problem independently.

### Local inference

Models are downloaded and executed locally rather than relying on external inference APIs.

### Lazy loading

Only the selected model is loaded into GPU memory.

### Modular architecture

Each model has its own backend while exposing a common application-level interface.

### Reproducibility

Environment dependencies and model-management logic are maintained within the repository.

---

## Data

Meter images are intentionally excluded from the repository.

Place local test images under:

```text
data/images/
```

Example:

```text
data/images/
├── dm_1.png
├── dm_2.png
└── ...
```

Do not commit private, sensitive, or production meter images to the repository.

---

## Models and Licenses

The models used by this project are third-party models.

Before redistribution or commercial deployment, review the respective model licenses and usage conditions:

- Qwen2-VL
- LightOnOCR
- InternVL2.5

This repository does not redistribute model weights.

---

## Future Direction

```text
Independent Model Testing
          |
          v
Benchmarking
          |
          v
Latency / VRAM Evaluation
          |
          v
vLLM Inference
          |
          v
Production Inference Pipeline
          |
          v
Application UI
```

The project is currently focused on establishing reliable independent serial-number recognition before moving to production deployment.

---

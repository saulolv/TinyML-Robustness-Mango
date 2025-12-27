# A Comparative Analysis of Lightweight CNNs for Robust and Efficient Mango Leaf Disease Classification

This repository contains the code, scripts, notebooks, and results for the research paper **"A Comparative Analysis of Lightweight CNNs for Robust and Efficient Mango Leaf Disease Classification"**.

## Research Summary

We compare three lightweight CNNs—**LCNN**, **MobileNetV3-Small**, and **EfficientNet-B0**—for mango leaf disease classification, focusing on:

- **Accuracy** on clean data
- **Robustness** to common image corruptions (mCE) using **MangoLeafDB-C**
- **Efficiency** for edge deployment (model size + **on-device** latency/throughput)

To enable a rigorous robustness evaluation, we introduce **MangoLeafDB-C**, built from the original [MangoLeafDB dataset](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset/data) by applying **19 corruption types** across **5 severity levels** (ImageNet-C protocol).

For performance evaluation on real hardware, inference latency and throughput are measured **on-device** using TensorFlow Lite (CPU) on **Android smartphones** and a **Raspberry Pi Zero 2 W**.

## Repository Structure

- **`App/RobustnessMango/`**: Android benchmark application (TFLite) used for on-device evaluation (clean accuracy + latency/throughput + corruption evaluation) and the paper sources (`artigo.tex`, `bib.bib`).
- **`Raspberry/`**: Raspberry Pi benchmark (Python) with `evaluate.py`, plus `models/` (`.tflite`) and a dataset copy under `dataset/`.
- **`mobile_analyses/`**: On-device metrics exported from smartphones/Raspberry (JSON) and a notebook (`metrics_analysis.ipynb`) used to analyze and visualize mobile/edge results.
- **`final_results/`**: Consolidated, paper-ready artifacts (figures in PDF, accuracy/mCE/latency JSONs, and exported model files/sizes).
- **`scripts/v2/`**: Previous/legacy codebase containing the notebooks, plots, tables, and intermediate artifacts used in the experiments (training, corruption evaluation, mCE computation, desktop analysis).
- **`external/`**: Adapted ImageNet-C code used to generate the corrupted benchmark (MangoLeafDB-C).
- **`mangoleaf/`**: A clean dataset copy (8 classes, 4000 images) used in the experiments.

## Research Pipeline (High-level)

1. **Dataset preparation**: MangoLeafDB (clean) split into train/val/test.
2. **MangoLeafDB-C generation**: 19 corruptions × 5 severity levels (95 subsets) following ImageNet-C.
3. **Model training**: Transfer learning for MobileNetV3-Small and EfficientNet-B0; LCNN trained from scratch.
4. **Model conversion**: Keras → TensorFlow Lite (`.tflite`) and optional compression for storage analysis.
5. **Evaluation**:
   - Clean accuracy (clean test set)
   - Robustness (mCE on MangoLeafDB-C)
   - Efficiency: model size + **on-device** latency (P90) and throughput (images/s) on Android and Raspberry Pi
6. **Results analysis**: compare trade-offs and identify Pareto-optimal choices.

## How to Reproduce the Experiments

### Prerequisites (Python / notebooks)

Install the Python dependencies used by the notebooks:

```bash
pip install -r requirements.txt
```

> Note: the on-device benchmarks are not run from the notebooks. Android uses the app in `App/RobustnessMango/`, and Raspberry Pi uses `Raspberry/evaluate.py`.

### 1. Dataset

Option A (recommended for full reproducibility): download the original dataset from Kaggle:

- [MangoLeafDB (Kaggle)](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset/data)

Option B: use the dataset copy already present in this repository under `mangoleaf/`.

### 2. Creating MangoLeafDB-C (corrupted benchmark)

The corruption generation code is located in `external/` (adapted from the ImageNet-C pipeline).

1. Open `external/ImageNet-C/create_c/make_imagenet_c.py`
2. Configure:
   - **source path** (clean MangoLeafDB)
   - **destination path** (output MangoLeafDB-C)
3. Run the script to generate the corrupted benchmark.

### 3. Model Training + Conversion (notebooks)

The training and conversion notebooks are under `scripts/v2/notebooks/`:

- **LCNN**: `scripts/v2/notebooks/lcnn.ipynb`
- **MobileNetV3-Small**: `scripts/v2/notebooks/mobilenetv3_small.ipynb`
- **EfficientNet-B0**: `scripts/v2/notebooks/efficientNetB0.ipynb`

Models and converted artifacts are stored under:

- Keras: `scripts/v2/notebooks/models/`
- TFLite + ZIP: `scripts/v2/notebooks/models/compressed/`

### 4. Robustness (mCE) + Desktop Analysis (notebooks)

- **Corrupted evaluation**:
  - `scripts/v2/notebooks/lccn_evaluate_corrupted_db.ipynb`
  - `scripts/v2/notebooks/mobilenetv3_evaluate_corrupted_db.ipynb`
  - `scripts/v2/notebooks/efficientNetB0_evaluate_corrupted_db.ipynb`
- **mCE + heatmaps**: `scripts/v2/notebooks/evaluate_mce.ipynb`
- **Size / latency notebooks (desktop)**:
  - `scripts/v2/notebooks/evaluate_size.ipynb`
  - `scripts/v2/notebooks/evaluate_latency.ipynb`
  - `scripts/v2/notebooks/latency_vs_mce.ipynb`

### 5. On-device Benchmarks

#### Android (mobile app)

The benchmark application lives in `App/RobustnessMango/` (Android Studio project).

1. Open `App/RobustnessMango/` in Android Studio
2. Connect a physical Android device
3. Build and run the app
4. In the app:
   - **“Iniciar Avaliação”** runs a balanced subset of **400 images** (50/class) and reports latency stats (including **P90**) and throughput (images/s).
   - **“Avaliar Corrupção Natural (5)”** runs the corruption benchmark using severity **5** (to keep the APK size manageable).

The app uses **TFLite CPU** inference and caps the interpreter to **up to 4 threads** (no NNAPI/GPU delegates).

The app exports JSON result files to the device **Downloads** folder.

#### Raspberry Pi (Python script)

The Raspberry Pi benchmark is in `Raspberry/evaluate.py`. It loads the `.tflite` models in `Raspberry/models/` and evaluates a balanced subset of the dataset in `Raspberry/dataset/mangoleaf/`.

Example run:

```bash
cd Raspberry
python3 evaluate.py --max-images 400 --warmups 10 --output results.json
```

Optional:

- `--threads N` to set interpreter threads (if supported by the runtime)
- `--only mobilenetv3_small` to run a subset of models

> Dependency note (Raspberry Pi): `evaluate.py` prefers `tflite_runtime` and falls back to `tf.lite.Interpreter` if TensorFlow is installed.

## Results

- Notebook-generated plots/tables/results: `scripts/v2/results/`
- Consolidated final artifacts (figures + metrics + models): `final_results/`
- Raw/on-device metric exports and analysis notebook: `mobile_analyses/`
- Android on-device logs: exported JSON files in **Downloads**
- Raspberry Pi output: the JSON passed via `--output` (default: `results.json`)

# final_results — paper-ready artifacts (for Figshare)

This folder contains the consolidated, paper-ready artifacts used in the manuscript **“A Comparative Analysis of Lightweight CNNs for Robust and Efficient Mango Leaf Disease Classification”**.  


## What’s included

### Figures (PDF)

- `latency_p90LatencyMs.pdf`: overall P90 latency comparison.
- `device_latency_p90LatencyMs.pdf`: per-device P90 latency comparison.
- `device_throughput.pdf`: per-device throughput (images/s) comparison.

### Clean accuracy (JSON)

- `accuracy_results.json`: clean test accuracy summary for each model.

### On-device latency/throughput (raw exports)

Folder: `latency/`

- `metrics_a12.json`: Android benchmark export (Samsung Galaxy A12).
- `metrics_a54_5g.json`: Android benchmark export (Samsung Galaxy A54 5G).
- `metrics_s24+.json`: Android benchmark export (Samsung Galaxy S24+).
- `results_raspberry.json`: Raspberry Pi benchmark output (Raspberry Pi Zero 2 W).

These JSON files contain per-model metrics such as average/median/P90 latency (ms) and throughput (images/s), computed using a balanced subset of **400 images** and **10 warmup runs**.

### Robustness (MangoLeafDB-C) — corruption errors (JSON/XLSX)

Folder: `mce/`

- `corruption_mobilenetv3_error.json`
- `corruption_efficientNet_error.json`
- `corruption_lcnn_error.json`
- `tabela_mce_transposed.xlsx`: table used for reporting/plotting robustness results.

### Exported models + sizes

Folder: `models/`

- `*.tflite`: exported TensorFlow Lite models used for on-device evaluation.
- `*.zip`: compressed versions of the TFLite models.
- `*.keras`: Keras model files (training outputs / reference).
- `model_sizes.json`: model size summary (MB) for the formats above.
- `size_models.ipynb`: notebook used to compute model size summaries.

### Analysis notebook

- `metrics_analysis.ipynb`: notebook used to generate the final plots and consolidate metrics from the JSON files above.

## How to reproduce the figures locally

1. Clone the repository that contains the code and notebooks.
2. Install the Python dependencies (see the repository-level `requirements.txt`).
3. Run `final_results/metrics_analysis.ipynb` end-to-end.

## Notes / scope

- **Datasets are not redistributed here.** MangoLeafDB can be obtained from Kaggle; MangoLeafDB-C can be generated via the provided corruption scripts in `external/`.
- This package focuses on **final results and artifacts** (metrics, figures, exported models) required to support the claims and enable verification.



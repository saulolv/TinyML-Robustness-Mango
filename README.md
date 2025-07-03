# A Comparative Analysis of Lightweight CNNs for Robust and Efficient Mango Leaf Disease Classification

This repository contains all the code, notebooks, and results for the research paper, "A Comparative Analysis of Lightweight CNNs for Robust and Efficient Mango Leaf Disease Classification".

## Research Summary

This study provides a comparative analysis of three lightweight Convolutional Neural Networks (CNNs)—**LCNN**, **MobileNetV3-Small**, and **EfficientNet-B0**—for the task of mango leaf disease classification. Our goal is to evaluate the critical trade-offs between model accuracy, robustness to real-world image corruptions, and computational efficiency (inference latency and model size) for practical deployment on edge devices in agriculture.

To facilitate a rigorous robustness evaluation, we introduce **MangoLeafDB-C**, a new benchmark dataset. It was created by applying 15 distinct, algorithmically-generated corruptions across 5 levels of severity to the original [MangoLeafDB dataset](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset/data), following the methodology established by the ImageNet-C benchmark. By benchmarking the models on both clean and corrupted data, we provide a multidimensional analysis to guide the development of more reliable and practical computer vision solutions for in-the-field plant disease diagnosis.

The entire research pipeline, from data preparation to final analysis, is documented in this repository to ensure full reproducibility.

## Research Pipeline

The experimental pipeline is organized as follows:

1.  **Dataset Preparation**: Download the original MangoLeafDB dataset and prepare the environment. The corrupted version, MangoLeafDB-C, is generated as part of the evaluation notebooks.
2.  **Model Training**: Train the three lightweight CNN architectures. Transfer learning is used for MobileNetV3-Small and EfficientNet-B0, while LCNN is trained from scratch.
3.  **Model Conversion**: Convert the trained Keras models to the TensorFlow Lite (`.tflite`) format for efficiency benchmarking.
4.  **Performance Evaluation**:
    - Assess classification accuracy on the clean test dataset.
    - Measure robustness on the MangoLeafDB-C benchmark using the mean Corruption Error (mCE) metric.
    - Benchmark inference latency and model size to evaluate computational efficiency.
5.  **Results Analysis**: Visualize and analyze the trade-offs between accuracy, robustness, and efficiency to identify Pareto-optimal models for edge deployment.

## How to Reproduce the Experiments

### Prerequisites

1.  Clone this repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### 1. Dataset

Download the [MangoLeafDB dataset](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset/data) from Kaggle and place it in a known location. The paths to the dataset will need to be updated within the notebooks.

The corrupted dataset, **MangoLeafDB-C**, is generated on-the-fly by the evaluation notebooks.

### 2. Model Training

The training process for each model is contained in its respective Jupyter notebook in the `v2/notebooks/` directory. These notebooks handle data loading, model definition, training, and saving the final weights.

- **LCNN**: `v2/notebooks/lcnn.ipynb`
- **MobileNetV3-Small**: `v2/notebooks/mobilenetv3_small.ipynb`
- **EfficientNet-B0**: `v2/notebooks/efficientNetB0.ipynb`

The trained models in `.keras` format are saved to `v2/models/`. The compressed `.tflite` versions used for evaluation are located in `v2/models/compressed/`.

### 3. Evaluation

The evaluation is split across multiple notebooks, each focusing on a specific aspect of performance.

- **Robustness Evaluation (Corrupted Data)**: The following notebooks apply corruptions to the dataset and evaluate the models' performance:

  - `v2/notebooks/lccn_evaluate_corrupted_db.ipynb`
  - `v2/notebooks/mobilenetv3_evaluate_corrupted_db.ipynb`
  - `v2/notebooks/efficientNetB0_evaluate_corrupted_db.ipynb`

- **Mean Corruption Error (mCE) Calculation**: To calculate the final mCE metric and generate comparison heatmaps, run:

  - `v2/notebooks/evaluate_mce.ipynb`

- **Efficiency Evaluation (Latency and Model Size)**:
  - **Latency**: `v2/notebooks/evaluate_latency.ipynb` measures inference speed (FPS).
  - **Model Size**: `v2/notebooks/evaluate_size.ipynb` compares the storage footprint of the models.

### 4. Results

All generated plots, tables, and raw result files are stored in the `v2/results/` directory. Key visualizations include:

- `latency_vs_mce.png`: The trade-off plot between model robustness (mCE) and inference latency.
- `mce_heatmaps.png`: A matrix of relative mCE values, comparing each model against the others.
- `latency_distribution.png`: Box plots showing the distribution of inference times for each model.

## Citation

If you use this work, please cite the original paper:

```bibtex
@inproceedings{your_conference_key,
  author    = {First Author and Second Author and Third Author},
  title     = {A Comparative Analysis of Lightweight CNNs for Robust and Efficient Mango Leaf Disease Classification},
  booktitle = {Conference Name},
  year      = {2024},
}
```

```

```

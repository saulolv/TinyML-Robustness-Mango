# A Comparative Analysis of Lightweight CNNs for Robust and Efficient Mango Leaf Disease Classification

This repository contains all the code, notebooks, and results for the research paper, "A Comparative Analysis of Lightweight CNNs for Robust and Efficient Mango Leaf Disease Classification".

## Research Summary

This study provides a comparative analysis of three lightweight Convolutional Neural Networks (CNNs)—**LCNN**, **MobileNetV3-Small**, and **EfficientNet-B0**—for the task of mango leaf disease classification. Our goal is to evaluate the critical trade-offs between model accuracy, robustness to real-world image corruptions, and computational efficiency (inference latency and model size) for practical deployment on edge devices in agriculture.

To facilitate a rigorous robustness evaluation, we introduce **MangoLeafDB-C**, a new benchmark dataset. It was created by applying 15 distinct, algorithmically-generated corruptions across 5 levels of severity to the original [MangoLeafDB dataset](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset/data), following the methodology established by the ImageNet-C benchmark. By benchmarking the models on both clean and corrupted data, we provide a multidimensional analysis to guide the development of more reliable and practical computer vision solutions for in-the-field plant disease diagnosis.

The entire research pipeline, from data preparation to final analysis, is documented in this repository to ensure full reproducibility.

## Research Pipeline

The experimental pipeline is organized as follows:

1.  **Dataset Preparation**: Download the original MangoLeafDB dataset and prepare the environment. The corrupted version, MangoLeafDB-C, is generated as part of the evaluation notebooks.
2.  **Creating MangoLeafDB-C**: The **MangoLeafDB-C** dataset, used for robustness evaluation, was generated using an adapted script from Dan Hendrycks' [robustness repository](https://github.com/hendrycks/robustness). The modifications apply the 15 corruptions at 5 severity levels to the original MangoLeafDB images.
3.  **Model Training**: Train the three lightweight CNN architectures. Transfer learning is used for MobileNetV3-Small and EfficientNet-B0, while LCNN is trained from scratch.
4.  **Model Conversion**: Convert the trained Keras models to the TensorFlow Lite (`.tflite`) format for efficiency benchmarking.
5.  **Performance Evaluation**:
    - Assess classification accuracy on the clean test dataset.
    - Measure robustness on the MangoLeafDB-C benchmark using the mean Corruption Error (mCE) metric.
    - Benchmark inference latency and model size to evaluate computational efficiency.
6.  **Results Analysis**: Visualize and analyze the trade-offs between accuracy, robustness, and efficiency to identify Pareto-optimal models for edge deployment.

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

### 2. Creating MangoLeafDB-C

The **MangoLeafDB-C** dataset, used for robustness evaluation, was generated using an adapted script from Dan Hendrycks' [robustness repository](https://github.com/hendrycks/robustness). The modifications apply the 15 corruptions at 5 severity levels to the original MangoLeafDB images.

The script is located in the `external/` directory. To create the dataset:

1.  Open the file `external/ImageNet-C/create_c/make_imagenet_c.py`.
2.  Update the file to set the source path to your MangoLeafDB dataset and the destination path for the output.
3.  Run the script to generate MangoLeafDB-C.

### 3. Model Training

The training process for each model is contained in its respective Jupyter notebook in the `v2/notebooks/` directory. These notebooks handle data loading, model definition, training, and saving the final weights.

- **LCNN**: `v2/notebooks/lcnn.ipynb`
- **MobileNetV3-Small**: `v2/notebooks/mobilenetv3_small.ipynb`
- **EfficientNet-B0**: `v2/notebooks/efficientNetB0.ipynb`

The trained models in `.keras` format are saved to `v2/models/`. The compressed `.tflite` versions used for evaluation are located in `v2/models/compressed/`.

### 4. Evaluation

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

### 5. Results

All generated plots, tables, and raw result files are stored in the `v2/results/` directory.

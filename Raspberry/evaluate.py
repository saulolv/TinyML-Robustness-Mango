import os
import time
import json
import argparse
from types import SimpleNamespace
import numpy as np
from PIL import Image

# Import TFLite interpreter from tflite_runtime or tensorflow
try:
    import tflite_runtime.interpreter as tflite
except Exception:
    try:
        import tensorflow as tf
        tflite = SimpleNamespace(Interpreter=tf.lite.Interpreter)
        print("Warning: using TensorFlow Lite (tf.lite.Interpreter) as fallback.")
    except Exception as import_err:
        raise ImportError(
            "Failed to import tflite_runtime or tensorflow.\n"
            "Install one of them (recommended: python3-tflite-runtime on Raspberry Pi).\n"
            f"Erro original: {import_err}"
        )

# --- Settings ---

# Definition of models to be evaluated (adjust according to the existing files)
MODELS_SPECS = [
    {"name": "mobilenetv3_small", "file": "models/mobilenetv3_model.tflite"},
    {"name": "efficientnet_b0", "file": "models/efficientnet_b0_mango.tflite"},
    {"name": "lcnn", "file": "models/lcnn_model.tflite"},
]

DATASET_ROOT = "dataset/mangoleaf"
LABELS_FILE = "models/labels.txt"
MAX_IMAGES = 400
WARMUP_RUNS = 10
NUM_THREADS = None  # None uses default runtime

# --- Helper Functions ---

def load_labels(path):
    """Load labels from text file."""
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def get_image_paths_balanced(dataset_path, classes, max_images):
    """
    Create a list of images for evaluation with balanced sampling,
    as in the Kotlin code.
    Return a list of tuples (image_path, class_name).
    """
    images_per_class = {cls: [] for cls in classes}
    for cls in classes:
        class_dir = os.path.join(dataset_path, cls)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images_per_class[cls].append(os.path.join(class_dir, img_file))

    per_class_target = max(1, int(max_images / len(classes)))
    sampled_images = []

    for cls in classes:
        # Get the first 'per_class_target' images of each class
        class_images = images_per_class[cls][:per_class_target]
        for img_path in class_images:
            sampled_images.append((img_path, cls))

    # Fill with remaining images if the target was not reached
    if len(sampled_images) < max_images:
        remaining_needed = max_images - len(sampled_images)
        extras = []
        for cls in classes:
            # Get the images that were not used
            remaining_class_images = images_per_class[cls][per_class_target:]
            for img_path in remaining_class_images:
                extras.append((img_path, cls))
        
        sampled_images.extend(extras[:remaining_needed])
        
    return sampled_images[:max_images]


def preprocess_image(image_path, input_details):
    """
    Load and preprocess an image for the TFLite model.
    The AS_IS_FLOAT32 normalization in Kotlin corresponds to converting
    the pixels to float32 without scaling.
    """
    # Get the input shape expected, ex: [1, 224, 224, 3]
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]
    input_dtype = input_details[0]['dtype']

    img = Image.open(image_path).convert('RGB')
    img = img.resize((width, height))
    
    # Convert the image to the dtype expected by the model
    if input_dtype == np.float32:
        # Keep range 0-255 (AS_IS_FLOAT32)
        img_array = np.array(img, dtype=np.float32)
    elif input_dtype == np.uint8:
        img_array = np.array(img, dtype=np.uint8)
    else:
        # Generic fallback
        img_array = np.array(img, dtype=input_dtype)
    
    # Add the batch dimension, ex: [224, 224, 3] -> [1, 224, 224, 3]
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# --- Main Evaluation Function ---

def evaluate_model(model_spec, image_paths, labels, warmup_runs=WARMUP_RUNS, num_threads=NUM_THREADS):
    """
    Execute the evaluation for a single model and return the metrics.
    """
    print(f"\n--- Evaluating model: {model_spec['name']} ---")
    
    # Load the TFLite interpreter
    if num_threads is None:
        interpreter = tflite.Interpreter(model_path=model_spec['file'])
    else:
        try:
            interpreter = tflite.Interpreter(model_path=model_spec['file'], num_threads=int(num_threads))
        except TypeError:
            # Some runtimes do not accept num_threads via kwargs
            interpreter = tflite.Interpreter(model_path=model_spec['file'])
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Warmup
    print(f"Executing {warmup_runs} warmups...")
    if image_paths:
        warmup_image = preprocess_image(image_paths[0][0], input_details)
        for _ in range(warmup_runs):
            interpreter.set_tensor(input_details[0]['index'], warmup_image)
            interpreter.invoke()

    # Evaluation
    latencies_ms = []
    correct_predictions = 0
    total_time_start = time.perf_counter()

    for i, (img_path, true_class) in enumerate(image_paths):
        if (i + 1) % 50 == 0:
            print(f"Processing image {i + 1} of {len(image_paths)}...")

        # Prepare the image
        input_data = preprocess_image(img_path, input_details)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Execute the inference and measure the latency
        start_time = time.perf_counter()
        interpreter.invoke()
        end_time = time.perf_counter()
        
        latencies_ms.append((end_time - start_time) * 1000)
        
        # Get the result
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = np.argmax(output_data)
        predicted_class = labels[predicted_index]
        
        if predicted_class == true_class:
            correct_predictions += 1
            
    total_time_end = time.perf_counter()
    total_time_ms = (total_time_end - total_time_start) * 1000
    
    # Calculate metrics
    images_processed = len(image_paths)
    accuracy = correct_predictions / images_processed if images_processed > 0 else 0
    
    lat_avg = np.mean(latencies_ms)
    lat_min = np.min(latencies_ms)
    lat_max = np.max(latencies_ms)
    lat_median = np.median(latencies_ms)
    lat_p90 = np.percentile(latencies_ms, 90)
    lat_std = np.std(latencies_ms)
    
    throughput = images_processed / (total_time_ms / 1000) if total_time_ms > 0 else 0

    print("\nResults:")
    print(f"  Images Processed: {images_processed}")
    print(f"  Accuracy: {accuracy:.4f} ({correct_predictions}/{images_processed})")
    print(f"  Latency (avg/min/med/p90/max): {lat_avg:.2f} / {lat_min:.2f} / {lat_median:.2f} / {lat_p90:.2f} / {lat_max:.2f} ms")
    print(f"  Standard Deviation (Latency): {lat_std:.2f} ms")
    print(f"  Throughput: {throughput:.2f} img/s")
    print(f"  Total Evaluation Time: {total_time_ms:.2f} ms")

    return {
        "model": model_spec['name'],
        "model_file": model_spec['file'],
        "images_processed": images_processed,
        "correct": int(correct_predictions),
        "accuracy": float(accuracy),
        "latency_ms": {
            "avg": float(lat_avg),
            "min": float(lat_min),
            "median": float(lat_median),
            "p90": float(lat_p90),
            "max": float(lat_max),
            "std": float(lat_std),
        },
        "throughput_img_s": float(throughput),
        "total_time_ms": float(total_time_ms),
    }

# --- Entry Point ---

def parse_args():
    parser = argparse.ArgumentParser(description="TFLite model evaluator for mango leaves")
    parser.add_argument("--dataset-root", default=DATASET_ROOT, help="Root directory of the dataset (with subdirectories by class)")
    parser.add_argument("--labels", default=LABELS_FILE, help="Path to the labels file")
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES, help="Maximum number of images to evaluate (balanced by class)")
    parser.add_argument("--warmups", type=int, default=WARMUP_RUNS, help="Number of inference warmups (warmup)")
    parser.add_argument("--threads", type=int, default=None, help="Number of threads for the interpreter (if supported)")
    parser.add_argument("--only", nargs="*", default=None, help="Only models with names containing these terms")
    parser.add_argument("--output", default="results.json", help="Path to the output JSON file")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = args.dataset_root
    labels_file = args.labels
    max_images = args.max_images
    warmups = args.warmups
    threads = args.threads

    if not os.path.exists(dataset_root):
        print(f"Error: Dataset directory not found at '{dataset_root}'")
        return 2

    try:
        labels = load_labels(labels_file)
        class_names = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
        print(f"Found {len(class_names)} classes and {len(labels)} labels.")

        image_paths_to_process = get_image_paths_balanced(dataset_root, class_names, max_images)
        print(f"Balanced sampling resulted in {len(image_paths_to_process)} images for evaluation.")

        if not image_paths_to_process:
            print("No images found for evaluation. Check the folder structure.")
            return 1

        # Filter models if --only is used
        model_specs = MODELS_SPECS
        if args.only:
            lower_terms = [t.lower() for t in args.only]
            model_specs = [m for m in MODELS_SPECS if any(t in m['name'].lower() for t in lower_terms)]

        results = []
        for spec in model_specs:
            model_file = spec.get('file')
            if not model_file or not os.path.exists(model_file):
                print(f"Warning: model file not found, skipping: {spec}")
                results.append({
                    "model": spec.get('name', 'desconhecido'),
                    "model_file": model_file,
                    "error": "model not found"
                })
                continue
            try:
                metrics = evaluate_model(spec, image_paths_to_process, labels, warmup_runs=warmups, num_threads=threads)
                results.append(metrics)
            except Exception as e:
                print(f"Error evaluating {spec['name']}: {e}")
                results.append({
                    "model": spec['name'],
                    "model_file": spec['file'],
                    "error": str(e)
                })

        report = {
            "dataset_root": dataset_root,
            "labels_file": labels_file,
            "num_classes": len(class_names),
            "num_images": len(image_paths_to_process),
            "sampling": "balanced_per_class",
            "warmups": warmups,
            "threads": threads,
            "timestamp_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "results": results,
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to: {args.output}")
        return 0

    except FileNotFoundError:
        print(f"Error: Labels file '{labels_file}' not found.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

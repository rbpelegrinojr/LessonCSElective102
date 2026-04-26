# Lesson 19: Deploying Your CNN Model

## Learning Objectives

By the end of this lesson, you will be able to:

- Save and load models in both `.h5` and `SavedModel` formats
- Convert a Keras model to TensorFlow Lite for mobile/edge deployment
- Build a complete inference pipeline with preprocessing and postprocessing
- Compare single vs. batch inference performance
- Understand deployment options: TFLite, Flask, TF Serving, ONNX

---

## Detailed Explanation

### Saving Models: .h5 vs SavedModel

Keras supports two primary save formats. The **HDF5 (`.h5`)** format is a single file containing weights and architecture, widely compatible but limited to Keras models. The **SavedModel** format is TensorFlow's native format — a directory containing the full computation graph, weights, and metadata. SavedModel is recommended for production because it supports TensorFlow Serving, TFLite conversion, and is language-agnostic.

```python
model.save('model.h5')           # HDF5 format
model.save('saved_model_dir/')   # SavedModel format (directory)
```

Loading is symmetric: `tf.keras.models.load_model('model.h5')` or `tf.keras.models.load_model('saved_model_dir/')`.

### Model Optimization for Deployment

Training a model prioritizes accuracy. Deploying a model prioritizes **speed, size, and efficiency**. Three main techniques bridge this gap:

**TensorFlow Lite (TFLite)**: Converts a SavedModel into a compact `.tflite` file optimized for mobile and embedded devices (Android, iOS, Raspberry Pi, microcontrollers). TFLite uses a flat buffer format and a lightweight runtime interpreter.

**Quantization**: Reduces numeric precision of weights and activations from 32-bit floats (`float32`) to 8-bit integers (`int8`). Benefits:
- Model size reduced by ~4×
- Inference speed increased (integer ops are faster on most hardware)
- Minimal accuracy loss (typically < 1%)

Types of quantization:
- **Post-training dynamic range quantization**: Easiest, quantizes weights only
- **Post-training integer quantization**: Quantizes both weights and activations; requires a representative dataset
- **Quantization-aware training (QAT)**: Simulates quantization during training for best accuracy

**Pruning**: Identifies and removes weights with near-zero magnitude, creating a sparse model. Sparse models can be stored more compactly and run faster on supported hardware. TensorFlow Model Optimization Toolkit provides pruning APIs.

### Inference Pipeline

A production inference pipeline has three stages:

```
Raw Input Image
      │
      ▼
┌─────────────────────────┐
│   PREPROCESSING          │
│  Resize → Normalize      │
│  → Add batch dimension   │
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│   MODEL INFERENCE        │
│  model.predict() or      │
│  TFLite interpreter      │
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│   POSTPROCESSING         │
│  argmax → class label    │
│  → confidence score      │
└─────────────────────────┘
              │
              ▼
     Prediction + Confidence
```

### Batch Inference vs. Single Inference

Single-image inference is convenient but inefficient — GPU/CPU overhead per call is high relative to computation. **Batch inference** processes multiple images in a single forward pass, dramatically improving throughput. For a REST API serving many requests, batching requests together (micro-batching) is a key optimization.

### Deployment Options

| Option | Best For | Complexity |
|---|---|---|
| Keras `model.predict()` | Prototyping, scripts | Low |
| Flask / FastAPI REST API | Web services, demos | Medium |
| TensorFlow Serving | Enterprise, high-throughput | High |
| TensorFlow Lite | Mobile / edge devices | Medium |
| ONNX | Cross-framework interoperability | Medium |

**Flask/FastAPI**: Wrap your inference pipeline in an HTTP endpoint. Clients send images as base64 or multipart form data; the server returns JSON predictions. FastAPI is preferred for modern Python APIs due to async support and automatic documentation.

**TensorFlow Serving**: A production-grade server optimized for ML models. Supports model versioning, A/B testing, and high-concurrency gRPC/REST endpoints. Run as a Docker container: `docker run -p 8501:8501 -v /model_dir:/models/my_model tensorflow/serving`.

**ONNX (Open Neural Network Exchange)**: A portable model format supported by PyTorch, TensorFlow, scikit-learn, and many inference runtimes (ONNX Runtime, TensorRT, OpenVINO). Useful for running TF-trained models in environments where TF is not available.

### Latency vs. Throughput

- **Latency**: Time to process one request (critical for interactive applications)
- **Throughput**: Number of requests processed per second (critical for batch jobs)

Reducing model size (quantization, pruning, smaller architecture) improves both. Using larger batches improves throughput but increases latency per request. Hardware accelerators (GPU, TPU, mobile NPU) improve both.

### Production Monitoring

Deployed models degrade over time as real-world data distributions shift (called **data drift** or **concept drift**). Monitor:
- **Prediction distribution**: Are class probabilities shifting?
- **Input statistics**: Are incoming images significantly different from training data?
- **Business metrics**: Is downstream performance (e.g., defect detection rate) declining?

Set up alerts and a retraining pipeline to respond to detected drift.

### Hardware Considerations

| Hardware | Best For |
|---|---|
| CPU | Development, low-throughput inference |
| GPU (NVIDIA) | Training, high-throughput batch inference |
| TPU | Massive-scale training and inference on Google Cloud |
| Mobile NPU | On-device inference (Apple Neural Engine, Qualcomm AI Engine) |
| Edge TPU (Coral) | Ultra-low-power edge devices |

---

## Key Concepts Table

| Concept | Definition |
|---|---|
| SavedModel | TF's recommended production save format (directory-based) |
| TFLite | Lightweight TF runtime for mobile and embedded devices |
| Quantization | Reducing weight precision (float32→int8) for smaller, faster models |
| Pruning | Removing near-zero weights to create sparse, efficient models |
| Inference Pipeline | Preprocessing → model prediction → postprocessing |
| Batch Inference | Processing multiple inputs in a single forward pass |
| Data Drift | Shift in real-world data distribution over time |
| ONNX | Cross-framework model exchange format |
| TF Serving | Production-grade TF model serving system |
| Latency | Time to process a single inference request |

---

## Code Reference

```python
# Save and convert to TFLite with quantization
model.save('saved_model/')

converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/')
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Dynamic quantization
tflite_model = converter.convert()

with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

# Run TFLite inference
interpreter = tf.lite.Interpreter(model_path='model_quantized.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])
```

---

## Activities

1. **Save and Reload:** Save a trained Keras model in both `.h5` format and `SavedModel` format using `model.save()`. Use `os.path.getsize()` (for `.h5`) and `os.path.getsize()` on each file in the `SavedModel` directory (sum them) to compare sizes. Reload each with `tf.keras.models.load_model()` and assert that predictions on 10 test images are identical.

2. **TFLite Conversion and Timing:** Convert the same Keras model to TFLite using `tf.lite.TFLiteConverter.from_keras_model()`. Load the `.tflite` file with `tf.lite.Interpreter`, run inference on a single image 100 times, and print the average inference time. Compare it to the average Keras inference time for the same image.
## Review Questions

1. What is the difference between `.h5` and `SavedModel` formats? When would you prefer each?
2. What is quantization, and what are its benefits and trade-offs?
3. What are the three stages of an inference pipeline?
4. Why is batch inference more efficient than processing one image at a time?
5. What is data drift and why does it matter for deployed models?
6. When would you use TensorFlow Lite vs. TensorFlow Serving?
7. What is ONNX and when is it useful?
8. What hardware would you choose for ultra-low-power on-device inference?

---

## Further Reading

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [TF Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [ONNX Official Site](https://onnx.ai/)
- [FastAPI for ML APIs](https://fastapi.tiangolo.com/)
- [Deploying Deep Learning Models in Production – Chip Huyen](https://huyenchip.com/2020/12/27/real-time-machine-learning.html)

# Lesson 19: Deploying Your CNN Model

## Learning Objectives
- Describe the complete model lifecycle from training through saving, serving, and monitoring
- Save and load models in multiple formats: Keras HDF5, TensorFlow SavedModel, and TensorFlow Lite
- Apply post-training quantisation to reduce model size and improve inference latency
- Build a simple prediction function with proper preprocessing that mirrors training-time preprocessing
- Demonstrate batch inference versus single-image inference and explain the performance difference
- Outline the key production considerations: latency, throughput, memory footprint, and versioning

## Detailed Explanation

### The Model Lifecycle

Training a model is only the beginning. A real-world ML pipeline has many more stages:

```
  ┌─────────────┐
  │   Collect   │
  │    Data     │
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │   Train &   │
  │   Evaluate  │
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │    Save /   │
  │  Serialise  │
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │  Optimise   │  ← quantisation, pruning, distillation
  │  (optional) │
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │    Serve    │  ← REST API, mobile app, edge device
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │   Monitor   │  ← data drift, performance degradation
  └─────────────┘
```

Each stage introduces its own requirements, tools, and failure modes. Understanding the full pipeline prevents the all-too-common situation where a model works perfectly on a researcher's laptop but fails in production.

### Model Serialisation Formats

#### Keras HDF5 (`.h5`)

The legacy Keras format stores the model architecture, weights, and training configuration in a single HDF5 binary file.

```python
model.save('model.h5')                  # save
model = tf.keras.models.load_model('model.h5')  # load
```

**Pros:** Single file, easy to share, human-readable structure with HDF5 tools.
**Cons:** Not recommended for production — less portable, slower loading for large models.

#### TensorFlow SavedModel

The recommended production format. SavedModel saves the model as a directory containing:
- `saved_model.pb` — the computational graph (protobuf format)
- `variables/` — the weight values
- `assets/` — optional auxiliary files (tokeniser vocabulary, label lists)

```python
model.save('saved_model_dir/')              # save as SavedModel
model = tf.saved_model.load('saved_model_dir/')  # load
```

**Pros:** Language-agnostic (usable in Python, C++, Java, JavaScript), supports serving infrastructure, preserves custom layers and functions, supports `tf.function` compilation.
**Cons:** Directory format (not a single file) requires archive (zip/tar) for transfer.

#### TensorFlow Lite (`.tflite`)

TFLite is a lightweight runtime designed for **mobile** (Android, iOS) and **embedded** (Raspberry Pi, microcontrollers) deployment. The conversion process:

```
Keras Model
    ↓  tf.lite.TFLiteConverter
TFLite FlatBuffer (.tflite)
    ↓  TFLite Interpreter (C++ runtime)
Inference on mobile / edge device
```

TFLite models are typically 3–4× smaller than SavedModels and run 2–10× faster on mobile hardware via hardware acceleration (NNAPI on Android, Core ML on iOS).

#### ONNX (Open Neural Network Exchange)

ONNX is an open standard format supported by PyTorch, TensorFlow, scikit-learn, and many commercial inference engines. It allows a model trained in one framework to be deployed in another.

```
TensorFlow Model → tf2onnx → ONNX Model → ONNX Runtime → fast C++ inference
```

ONNX is particularly valuable in enterprise environments where the training team uses TensorFlow but the deployment team uses a different stack.

### Model Optimisation Techniques

Training a model at full float32 precision produces the most accurate weights, but float32 is expensive at inference time — each weight takes 4 bytes of memory and requires full-precision arithmetic. Optimisation techniques reduce this cost.

#### Post-Training Quantisation

**Quantisation** converts float32 weights and activations to lower-bit integers (typically int8), reducing model size by ~4× and inference latency by 2–3× on compatible hardware.

```
float32 weight: 0.6734201...  (4 bytes, 32-bit)
  ↓  quantise
int8 weight:    86            (1 byte, 8-bit)
  ↓  dequantise during inference
≈ 0.673               (slight precision loss, < 1% accuracy drop typically)
```

Three levels of TFLite quantisation:
1. **Dynamic range quantisation**: quantises weights only; fast and easy, ~4× size reduction
2. **Full integer quantisation**: quantises weights AND activations; requires a small calibration dataset (~100–500 images)
3. **Float16 quantisation**: 16-bit floats; compatible with GPU acceleration; 2× size reduction with minimal accuracy loss

#### Model Pruning

Pruning removes weights close to zero and makes them exactly zero (sparse network). A pruned model can be compressed significantly using sparse representations. With 50 % sparsity, file size shrinks by ~2× with < 1 % accuracy loss in well-trained networks.

```
Before pruning:  [0.71, 0.03, -0.82, 0.01, 0.64, -0.02, 0.49]
After pruning:   [0.71, 0.00, -0.82, 0.00, 0.64,  0.00, 0.49]
```

#### Knowledge Distillation

Train a small **student** network to mimic the outputs of a large **teacher** network. The student learns from soft probability distributions (the teacher's softmax outputs), not just hard labels, and achieves accuracy much closer to the teacher than if it had been trained from scratch on hard labels alone.

```
Teacher: ResNet50 (25M params, 93% acc)
    ↓  soft labels  ← distillation loss
Student: MobileNetV2 (3.4M params, ~90% acc with distillation vs ~88% without)
```

### Building a Prediction Function

A production prediction function must exactly replicate the preprocessing applied during training. Mismatches in preprocessing are one of the most common production bugs:

```python
def predict_image(model, image_path, class_names):
    # 1. Load image at the correct resolution
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    # 2. Convert to numpy array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # 3. Apply EXACT same normalisation used during training
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    # 4. Add batch dimension: (H, W, C) → (1, H, W, C)
    img_batch = np.expand_dims(img_array, axis=0)
    # 5. Run inference
    predictions = model.predict(img_batch)
    # 6. Return human-readable result
    top_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return top_class, confidence
```

### Single vs Batch Inference

**Single-image inference** is convenient but inefficient: the GPU spends most of its time waiting, not computing. **Batch inference** amortises the GPU setup overhead across many images simultaneously.

```
Single inference:    GPU utilisation ~5–10%
                     Throughput: ~30 images/second

Batch inference (32): GPU utilisation ~80–95%
                      Throughput: ~300 images/second
                      (10× throughput for same hardware!)
```

In production, requests are often accumulated into mini-batches over a short window (5–50 ms) before being processed together — a technique called **dynamic batching** used in TensorFlow Serving.

### TensorFlow Serving

TensorFlow Serving is a production-grade server for serving TensorFlow models via a REST or gRPC API:

```
[Client] --HTTP POST /v1/models/mymodel:predict--> [TF Serving]
                                                          ↓
                                                   [SavedModel]
                                                          ↓
                                                   [Predictions]
                                                          ↓
<-------- JSON response {"predictions": [...]} ----------[Client]
```

TF Serving handles: versioned model loading, automatic model updates, batching, GPU/CPU dispatch, and metrics export to Prometheus. In production at Google scale, it processes billions of inferences per day.

### Simple Flask REST API

For smaller deployments or prototypes, a Flask REST API is a lightweight alternative to TF Serving:

```
POST /predict
Content-Type: multipart/form-data
Body: image file

Response:
{
  "class": "airplane",
  "confidence": 0.9342,
  "top5": [
    {"class": "airplane", "confidence": 0.9342},
    {"class": "bird",     "confidence": 0.0421},
    ...
  ]
}
```

### Production Considerations

| Consideration | Why It Matters | Mitigation |
|---|---|---|
| Latency (p99) | Tail latency degrades user experience | TFLite, quantisation, batching |
| Throughput (req/s) | Limits concurrent users | Horizontal scaling, dynamic batching |
| Memory footprint | Limits deployment hardware | Pruning, quantisation, smaller architecture |
| Input validation | Malformed inputs cause silent errors | Schema validation before preprocessing |
| Preprocessing mismatch | Silent accuracy degradation | Use the same preprocessing function for training and inference |
| Model versioning | Rollback if new model underperforms | Canary deployment, A/B testing |
| Data drift monitoring | Real-world distribution shifts over time | Monitor input statistics; schedule retraining |

## Key Concepts

| Term | Definition | Why It Matters |
|---|---|---|
| SavedModel | TensorFlow's portable, production-ready model format stored as a directory | Recommended format for serving; language-agnostic |
| HDF5 (`.h5`) | Legacy Keras model format storing architecture and weights in one file | Good for sharing experiments; not recommended for production |
| TFLite | Lightweight TensorFlow runtime for mobile and embedded devices | Enables on-device inference without network connectivity |
| Quantisation | Reducing weight precision from float32 to int8 or float16 | 4× smaller model, 2–3× faster inference, minimal accuracy drop |
| Pruning | Setting small-magnitude weights to exactly zero | Enables compression; paired with sparse inference engines |
| Knowledge Distillation | Training a small student model to mimic a large teacher's outputs | Achieves near-teacher accuracy in a fraction of the parameters |
| TF Serving | Production server for TensorFlow models with REST and gRPC APIs | Handles versioning, batching, scaling in enterprise deployments |
| Batch Inference | Processing multiple inputs in a single forward pass | 10× throughput improvement over sequential single-image inference |
| Preprocessing Mismatch | Training-time and inference-time preprocessing differ | Causes silent accuracy regression — extremely common production bug |
| Model Card | Structured documentation of a model's intended use, metrics, and limitations | Enables informed deployment decisions and stakeholder communication |

## Code Reference

See the fully runnable demonstration in **`code/lesson_19.py`**.

## Activities

1. **Format Comparison** — Save a trained CIFAR-10 model in both `.h5` and SavedModel formats. Compare the file sizes using `os.path.getsize()`. Then save a TFLite version with dynamic range quantisation. Create a table showing size, load time, and inference time for all three formats.

2. **Preprocessing Pipeline** — Write a prediction function that loads an image from a file path, applies the correct preprocessing for MobileNetV2, and returns the top-3 predictions with confidence scores. Test it on 5 different images and verify the outputs are sensible.

3. **Batch Inference Benchmark** — Using `time.perf_counter()`, measure the time to run inference on 1,000 CIFAR-10 test images in three ways: (a) one image at a time, (b) batches of 32, (c) batches of 128. Plot throughput (images/second) versus batch size.

4. **TFLite Quantisation Accuracy** — Convert a model to TFLite using dynamic range quantisation, then full integer quantisation. Run both on the CIFAR-10 test set. Compare accuracy of: full model, dynamic quantisation, and full integer quantisation. What is the accuracy-size trade-off?

5. **Simple Prediction Server** — Write a Flask application (without actually running a server) that: (a) loads a SavedModel at startup, (b) accepts a POST request with an image, (c) preprocesses the image, (d) runs inference, (e) returns a JSON response with class name and confidence. Test the preprocessing logic without a live server.

## Review Questions

1. What are the key differences between TensorFlow SavedModel and HDF5 formats? In what scenario would you choose each?
2. Explain post-training quantisation: what is being quantised, by how much, and what is the trade-off in model accuracy?
3. Why is batch inference typically 5–10× more efficient than sequential single-image inference? What hardware characteristic explains this?
4. Describe the "preprocessing mismatch" problem in production ML. What steps can you take to prevent it?
5. You have a trained CNN model that needs to run on Android smartphones without internet access. What format would you use, and what optimisation techniques would you apply? Explain your reasoning.

## Further Reading

- **TensorFlow Lite Guide** — `https://www.tensorflow.org/lite/guide` — complete documentation for TFLite conversion, quantisation, and deployment across platforms
- **"Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"** — Jacob et al. (2018) — the foundational paper for quantisation-aware training
- **TensorFlow Serving Documentation** — `https://www.tensorflow.org/tfx/guide/serving` — official guide to production model serving with REST and gRPC
- **"Distilling the Knowledge in a Neural Network"** — Hinton, Vinyals, Dean (2015) — the original knowledge distillation paper; concise and highly readable
- **"MLOps: Continuous delivery and automation pipelines in machine learning"** — Google (2021) — practical guide to the full model lifecycle in production, covering monitoring, retraining, and versioning

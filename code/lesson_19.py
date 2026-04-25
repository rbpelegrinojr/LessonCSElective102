"""
Lesson 19: Deploying Your CNN Model
=====================================
Demonstrates saving/loading models in multiple formats, TFLite conversion,
quantisation, batch inference benchmarking, and a prediction function.
"""
import os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

tf.random.set_seed(42)
np.random.seed(42)

print("=" * 60)
print("LESSON 19: Deploying Your CNN Model")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")

# ===================================================================
# SECTION 1: TRAIN / LOAD MODEL
# ===================================================================
print("\n" + "=" * 60)
print("SECTION 1: Train / Load Model")
print("=" * 60)

CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
NUM_CLASSES, IMG_SIZE, BATCH_SIZE = 10, 96, 64

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

def make_ds(x, y, shuffle=False):
    def _pre(imgs, lbls):
        imgs = tf.cast(imgs, tf.float32)
        imgs = tf.image.resize(imgs, [IMG_SIZE, IMG_SIZE])
        imgs = mobilenet_preprocess(imgs)
        lbls = tf.squeeze(lbls, -1)
        lbls = tf.one_hot(lbls, NUM_CLASSES)
        return imgs, lbls
    ds = tf.data.Dataset.from_tensor_slices((x, y)).map(_pre, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle: ds = ds.shuffle(2000)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

SAVED_PATH = 'lesson_17_finetuned_model.keras'
if os.path.exists(SAVED_PATH):
    print(f"Loading model from {SAVED_PATH}")
    model = keras.models.load_model(SAVED_PATH)
else:
    print("No saved model found — building and training quick model...")
    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = False
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.Dropout(0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = keras.Model(inp, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(make_ds(x_train[:8000], y_train[:8000], shuffle=True),
              validation_data=make_ds(x_train[8000:10000], y_train[8000:10000]),
              epochs=10, callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
              verbose=1)

test_ds = make_ds(x_test, y_test)
_, acc = model.evaluate(test_ds, verbose=0)
print(f"Model test accuracy: {acc*100:.2f}%")

# ===================================================================
# SECTION 2: SAVE IN MULTIPLE FORMATS AND COMPARE SIZES
# ===================================================================
print("\n" + "=" * 60)
print("SECTION 2: Save in Multiple Formats")
print("=" * 60)

# --- SavedModel ---
SAVEDMODEL_DIR = 'lesson_19_savedmodel'
print(f"Saving as TF SavedModel → {SAVEDMODEL_DIR}/")
model.save(SAVEDMODEL_DIR)

# --- HDF5 ---
H5_PATH = 'lesson_19_model.h5'
print(f"Saving as HDF5 → {H5_PATH}")
model.save(H5_PATH)

# --- TFLite: dynamic range quantisation ---
TFLITE_PATH = 'lesson_19_model.tflite'
print("Converting to TFLite (dynamic range quantisation)...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # enables dynamic range quant
tflite_model = converter.convert()
with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)
print(f"Saved TFLite model → {TFLITE_PATH}")

# --- TFLite: float16 quantisation ---
TFLITE_FP16_PATH = 'lesson_19_model_fp16.tflite'
converter_fp16 = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_fp16.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter_fp16.convert()
with open(TFLITE_FP16_PATH, 'wb') as f:
    f.write(tflite_fp16)
print(f"Saved TFLite float16 model → {TFLITE_FP16_PATH}")

# Measure sizes
def dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            total += os.path.getsize(os.path.join(root, fname))
    return total / 1e6

sizes = {
    'SavedModel':      dir_size_mb(SAVEDMODEL_DIR),
    'HDF5 (.h5)':      os.path.getsize(H5_PATH) / 1e6,
    'TFLite (int8)':   os.path.getsize(TFLITE_PATH) / 1e6,
    'TFLite (fp16)':   os.path.getsize(TFLITE_FP16_PATH) / 1e6,
}
print("\nModel Size Comparison:")
print(f"{'Format':<20} {'Size (MB)':>10}")
print("-" * 32)
for fmt, size in sizes.items():
    print(f"{fmt:<20} {size:>10.2f}")

# Bar chart
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(list(sizes.keys()), list(sizes.values()), color=['steelblue','tomato','green','orange'], edgecolor='black')
ax.set_ylabel('Size (MB)'); ax.set_title('Model Size by Format'); ax.grid(axis='y', alpha=0.3)
for i, (k, v) in enumerate(sizes.items()):
    ax.text(i, v + 0.1, f'{v:.1f}MB', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('lesson_19_model_sizes.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson_19_model_sizes.png")

# ===================================================================
# SECTION 3: PREDICTION FUNCTION WITH PROPER PREPROCESSING
# ===================================================================
print("\n" + "=" * 60)
print("SECTION 3: Prediction Function with Proper Preprocessing")
print("=" * 60)

def predict_image_array(model, raw_img_uint8, class_names, top_k=3):
    """
    Run inference on a single raw uint8 image (H×W×3).
    Applies EXACT same preprocessing as during training.
    Returns top-k predictions as a list of (class_name, confidence) tuples.
    """
    # Step 1: cast to float and resize
    img = tf.cast(raw_img_uint8, tf.float32)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    # Step 2: MobileNetV2 normalisation [-1, 1]
    img = mobilenet_preprocess(img)
    # Step 3: add batch dimension
    img_batch = tf.expand_dims(img, axis=0)
    # Step 4: inference
    preds = model.predict(img_batch, verbose=0)[0]
    # Step 5: decode top-k
    top_idx = np.argsort(preds)[::-1][:top_k]
    return [(class_names[i], float(preds[i])) for i in top_idx]

print("\nRunning single-image prediction on 5 test samples:")
for i in range(5):
    raw = x_test[i]
    true_label = CLASS_NAMES[y_test[i][0]]
    top3 = predict_image_array(model, raw, CLASS_NAMES, top_k=3)
    print(f"  True: {true_label:<12} | "
          f"Top-1: {top3[0][0]:<12} ({top3[0][1]*100:.1f}%) | "
          f"Top-2: {top3[1][0]:<12} ({top3[1][1]*100:.1f}%)")

# ===================================================================
# SECTION 4: BATCH INFERENCE BENCHMARK
# ===================================================================
print("\n" + "=" * 60)
print("SECTION 4: Batch Inference Benchmark")
print("=" * 60)

# Pre-process 1000 test images once
N_BENCH = 500
bench_imgs = np.stack([
    tf.image.resize(tf.cast(x_test[i], tf.float32), [IMG_SIZE, IMG_SIZE]).numpy()
    for i in range(N_BENCH)
])
bench_imgs = mobilenet_preprocess(bench_imgs.astype(np.float32))

batch_sizes    = [1, 8, 32, 64, 128]
throughputs    = []
latencies_ms   = []

for bs in batch_sizes:
    start = time.perf_counter()
    for start_idx in range(0, N_BENCH, bs):
        batch = bench_imgs[start_idx:start_idx + bs]
        if len(batch) == 0:
            break
        model.predict(batch, verbose=0)
    elapsed = time.perf_counter() - start
    tput = N_BENCH / elapsed
    throughputs.append(tput)
    latencies_ms.append((elapsed / (N_BENCH / bs)) * 1000)
    print(f"  Batch size {bs:>4}: throughput={tput:>7.1f} img/s  "
          f"avg latency/batch={latencies_ms[-1]:.1f} ms")

# Plot throughput vs batch size
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Lesson 19: Batch Inference Benchmarks", fontsize=13, fontweight='bold')
axes[0].plot(batch_sizes, throughputs, 'o-', color='steelblue', linewidth=2, markersize=8)
axes[0].set_xlabel('Batch Size'); axes[0].set_ylabel('Throughput (images/sec)')
axes[0].set_title('Throughput vs Batch Size'); axes[0].grid(alpha=0.3)
for bs, tput in zip(batch_sizes, throughputs):
    axes[0].annotate(f'{tput:.0f}', (bs, tput), textcoords='offset points', xytext=(0, 8), ha='center')

axes[1].plot(batch_sizes, latencies_ms, 's-', color='tomato', linewidth=2, markersize=8)
axes[1].set_xlabel('Batch Size'); axes[1].set_ylabel('Avg Latency per Batch (ms)')
axes[1].set_title('Latency vs Batch Size'); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('lesson_19_inference_benchmark.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: lesson_19_inference_benchmark.png")

# TFLite inference demo
print("\nRunning TFLite inference on 5 test samples...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for i in range(5):
    img = bench_imgs[i:i+1].astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_class = CLASS_NAMES[np.argmax(out)]
    true_class = CLASS_NAMES[y_test[i][0]]
    print(f"  TFLite  True: {true_class:<12} Pred: {pred_class:<12} conf={np.max(out)*100:.1f}%")

print("\n" + "=" * 60)
print("LESSON 19 COMPLETE")
print("Key takeaway: SavedModel for production, TFLite for mobile.")
print("Batch inference is 5-10x more efficient than single-image inference.")
print("=" * 60)

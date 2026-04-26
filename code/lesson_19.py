"""
Lesson 19: Deploying Your CNN Model
Demonstrates model saving, loading, TFLite conversion, and inference pipeline.
"""

import matplotlib
matplotlib.use('Agg')

# Standard imports
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models

print("TensorFlow version:", tf.__version__)

# ============================================================
# === SECTION 1: Train and Save Model in Multiple Formats ===
# ============================================================
print("\n" + "=" * 60)
print("SECTION 1: Train and Save Model in Multiple Formats")
print("=" * 60)

# Load MNIST for a quick demo model
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[..., np.newaxis].astype("float32") / 255.0
x_test  = x_test[..., np.newaxis].astype("float32") / 255.0

print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")

# Build a simple CNN
def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ], name="mnist_cnn")
    return model

model = build_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train for 2 epochs (fast demo)
print("\nTraining for 2 epochs...")
model.fit(x_train[:10000], y_train[:10000], epochs=2, batch_size=128,
          validation_split=0.1, verbose=1)

# Save as .h5
h5_path = '/tmp/model_lesson19.h5'
model.save(h5_path)
h5_size = os.path.getsize(h5_path) / 1024
print(f"\nSaved .h5 model: {h5_path}  ({h5_size:.1f} KB)")

# Save as SavedModel format
sm_path = '/tmp/saved_model_lesson19'
model.save(sm_path)
# Compute total size of SavedModel directory
sm_size = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, _, filenames in os.walk(sm_path)
    for f in filenames
) / 1024
print(f"Saved SavedModel: {sm_path}/  ({sm_size:.1f} KB)")

# ============================================================
# === SECTION 2: Load and Run Inference with Timing ===
# ============================================================
print("\n" + "=" * 60)
print("SECTION 2: Load and Run Inference with Timing")
print("=" * 60)

# Load .h5 model
model_h5 = tf.keras.models.load_model(h5_path)
print("Loaded .h5 model successfully")

# Load SavedModel
model_sm = tf.keras.models.load_model(sm_path)
print("Loaded SavedModel successfully")

# Prepare 10 test samples
samples = x_test[:10]
true_labels = y_test[:10]

# Time .h5 inference
t0 = time.time()
preds_h5 = model_h5.predict(samples, verbose=0)
t1 = time.time()
print(f"\n.h5 inference on 10 images: {(t1 - t0)*1000:.2f} ms")

# Time SavedModel inference
t0 = time.time()
preds_sm = model_sm.predict(samples, verbose=0)
t1 = time.time()
print(f"SavedModel inference on 10 images: {(t1 - t0)*1000:.2f} ms")

# Verify they produce identical results
h5_classes = np.argmax(preds_h5, axis=1)
sm_classes = np.argmax(preds_sm, axis=1)
match = np.all(h5_classes == sm_classes)
print(f"\n.h5 predictions:         {h5_classes}")
print(f"SavedModel predictions:  {sm_classes}")
print(f"True labels:             {true_labels}")
print(f"Predictions match: {match}")

# ============================================================
# === SECTION 3: Convert to TensorFlow Lite ===
# ============================================================
print("\n" + "=" * 60)
print("SECTION 3: Convert to TensorFlow Lite")
print("=" * 60)

# Convert the SavedModel to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(sm_path)
tflite_model = converter.convert()

# Save the TFLite model
tflite_path = '/tmp/model_lesson19.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

tflite_size = os.path.getsize(tflite_path) / 1024
print(f"TFLite model saved: {tflite_path}  ({tflite_size:.1f} KB)")
print(f"Size reduction: {h5_size:.1f} KB → {tflite_size:.1f} KB  "
      f"({100*(1 - tflite_size/h5_size):.1f}% smaller)")

# Run inference with TFLite Interpreter
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"\nTFLite input shape:  {input_details[0]['shape']}")
print(f"TFLite output shape: {output_details[0]['shape']}")

tflite_preds = []
for img in samples[:5]:
    inp = img[np.newaxis, ...].astype(np.float32)  # Add batch dim
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    tflite_preds.append(np.argmax(out))

print(f"\nTFLite predictions (5 samples): {tflite_preds}")
print(f"Original predictions:           {list(h5_classes[:5])}")
print(f"True labels:                    {list(true_labels[:5])}")

# ============================================================
# === SECTION 4: Batch Inference Performance Comparison ===
# ============================================================
print("\n" + "=" * 60)
print("SECTION 4: Batch Inference Performance Comparison")
print("=" * 60)

n_images = 100
test_images = x_test[:n_images]

# Single-image inference loop
print("Timing single-image inference (100 calls)...")
t0 = time.time()
for img in test_images:
    _ = model_h5.predict(img[np.newaxis, ...], verbose=0)
t1 = time.time()
single_time = t1 - t0
print(f"Single-image loop: {single_time*1000:.1f} ms total  "
      f"({single_time*1000/n_images:.2f} ms/image)")

# Batch inference
print("Timing batch inference (1 call with 100 images)...")
t0 = time.time()
_ = model_h5.predict(test_images, batch_size=100, verbose=0)
t1 = time.time()
batch_time = t1 - t0
print(f"Batch inference:   {batch_time*1000:.1f} ms total  "
      f"({batch_time*1000/n_images:.2f} ms/image)")
print(f"Speedup factor: {single_time / batch_time:.1f}x")

# Plot comparison
fig, ax = plt.subplots(figsize=(7, 4))
labels = ['Single-image\nloop (100 calls)', 'Batch inference\n(1 call)']
times  = [single_time * 1000, batch_time * 1000]
bars = ax.bar(labels, times, color=['#e74c3c', '#2ecc71'], width=0.4)
ax.set_ylabel('Time (ms)')
ax.set_title('Inference Time: Single vs Batch (100 images)')
for bar, val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f'{val:.0f} ms', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('/tmp/inference_timing.png', dpi=100)
print("Saved /tmp/inference_timing.png")

# ============================================================
# === SECTION 5: Production Inference Pipeline ===
# ============================================================
print("\n" + "=" * 60)
print("SECTION 5: Production Inference Pipeline")
print("=" * 60)

CLASS_NAMES = [str(i) for i in range(10)]  # MNIST digit names


def predict_image(model, image_input, class_names, input_size=(28, 28)):
    """
    Full production inference pipeline.

    Parameters
    ----------
    model       : Keras model
    image_input : numpy array (H, W) or (H, W, C) or (H, W, 1)
    class_names : list of string class labels
    input_size  : (height, width) expected by the model

    Returns
    -------
    dict with 'class', 'confidence', 'top3'
    """
    # 1. Ensure 2-D or 3-D array
    img = np.array(image_input, dtype=np.float32)

    # 2. Add channel dim if needed
    if img.ndim == 2:
        img = img[..., np.newaxis]

    # 3. Resize if necessary (simple crop/pad for MNIST demo)
    h, w = img.shape[:2]
    target_h, target_w = input_size
    if (h, w) != (target_h, target_w):
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray((img.squeeze() * 255).astype(np.uint8))
        pil_img = pil_img.resize((target_w, target_h))
        img = np.array(pil_img, dtype=np.float32)[..., np.newaxis]

    # 4. Normalise to [0, 1]
    if img.max() > 1.0:
        img = img / 255.0

    # 5. Add batch dimension → (1, H, W, C)
    img_batch = img[np.newaxis, ...]

    # 6. Run inference
    probs = model.predict(img_batch, verbose=0)[0]  # shape (num_classes,)

    # 7. Top-1 prediction
    pred_idx    = int(np.argmax(probs))
    confidence  = float(probs[pred_idx])
    pred_class  = class_names[pred_idx]

    # 8. Top-3 predictions
    top3_idx  = np.argsort(probs)[::-1][:3]
    top3 = [(class_names[i], float(probs[i])) for i in top3_idx]

    return {
        'class':      pred_class,
        'confidence': confidence,
        'top3':       top3,
    }


# Test on 5 sample images
print("\nRunning predict_image() on 5 test samples:")
print("-" * 50)
for i in range(5):
    result = predict_image(model_h5, x_test[i], CLASS_NAMES)
    true_label = CLASS_NAMES[y_test[i]]
    correct = "✓" if result['class'] == true_label else "✗"
    print(f"  Sample {i+1}: true={true_label}  pred={result['class']}  "
          f"conf={result['confidence']*100:.1f}%  {correct}")
    top3_str = ", ".join(f"{c}({p*100:.1f}%)" for c, p in result['top3'])
    print(f"           top-3: {top3_str}")

print("\nLesson 19 complete — model saved, converted to TFLite, and deployment pipeline built.")
plt.show()

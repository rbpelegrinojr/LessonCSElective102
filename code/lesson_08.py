"""
Lesson 8: Building Your First Complete CNN
CNN Image Classification Course

Sections:
  1. Implement LeNet-5 exactly in Keras
  2. Build a modern tiny CNN from scratch (3 conv blocks)
  3. Model summary analysis — explain each layer's output shape
  4. Compile and do a single forward pass
  5. Train on MNIST for 3 epochs and plot accuracy
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input

print("=" * 60)
print("LESSON 8: Building Your First Complete CNN")
print("TensorFlow version:", tf.__version__)
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# SECTION 1: Implement LeNet-5 Exactly in Keras
# ──────────────────────────────────────────────────────────────
print("\n--- Section 1: LeNet-5 Implementation ---")

def build_lenet5():
    """
    LeNet-5 as described by LeCun et al. (1998).
    Original uses average pooling and tanh; we keep that faithfully.
    Input: 32×32×1 (pad MNIST from 28→32 at inference)
    """
    model = models.Sequential([
        # C1: 6 filters, 5×5, valid padding → 28×28×6
        layers.Conv2D(6, kernel_size=(5, 5), activation='tanh',
                      input_shape=(32, 32, 1), padding='valid', name='C1'),
        # S2: Average pooling 2×2, stride 2 → 14×14×6
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='S2'),
        # C3: 16 filters, 5×5, valid padding → 10×10×16
        layers.Conv2D(16, kernel_size=(5, 5), activation='tanh',
                      padding='valid', name='C3'),
        # S4: Average pooling 2×2, stride 2 → 5×5×16
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='S4'),
        # Flatten: 5×5×16 = 400
        layers.Flatten(name='Flatten'),
        # F5: Fully connected 120
        layers.Dense(120, activation='tanh', name='F5'),
        # F6: Fully connected 84
        layers.Dense(84, activation='tanh', name='F6'),
        # Output: 10 classes
        layers.Dense(10, activation='softmax', name='Output'),
    ], name='LeNet-5')
    return model

lenet = build_lenet5()
lenet.summary(line_length=70)

total_params = lenet.count_params()
print(f"\n  Total parameters: {total_params:,}")
print("  (Original LeNet-5 had ~61,706 parameters)")

print("\n  Layer-by-layer parameter breakdown:")
for layer in lenet.layers:
    params = layer.count_params()
    if params > 0:
        out_shape = layer.output_shape
        print(f"    {layer.name:<12}: output {str(out_shape[1:]):<18} "
              f"params = {params:>6,}")

# ──────────────────────────────────────────────────────────────
# SECTION 2: Build a Modern Tiny CNN (3 Conv Blocks)
# ──────────────────────────────────────────────────────────────
print("\n--- Section 2: Modern Tiny CNN with 3 Conv Blocks ---")

def build_tiny_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Modern tiny CNN with standard blocks:
    Conv2D → BatchNorm → ReLU → MaxPooling
    Ends with GlobalAveragePooling + Dense head.
    """
    inp = Input(shape=input_shape)

    # Block 1: 32 filters
    x = layers.Conv2D(32, (3, 3), padding='same', name='conv1')(inp)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)

    # Block 2: 64 filters
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu', name='relu2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)

    # Block 3: 128 filters
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('relu', name='relu3')(x)

    # Classification head
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(128, activation='relu', name='fc1')(x)
    out = layers.Dense(num_classes, activation='softmax', name='output')(x)

    return models.Model(inp, out, name='TinyCNN')

tiny_cnn = build_tiny_cnn()
print("  Modern Tiny CNN summary:")
tiny_cnn.summary(line_length=70)

print(f"\n  Total parameters: {tiny_cnn.count_params():,}")
print("\n  Architecture follows: Conv → BN → ReLU → MaxPool pattern")
print("  Filters double each block: 32 → 64 → 128 (standard practice)")

# ──────────────────────────────────────────────────────────────
# SECTION 3: Model Summary Analysis
# ──────────────────────────────────────────────────────────────
print("\n--- Section 3: Model Summary Analysis ---")

print("  Explaining each layer's role and parameter count:\n")

layer_explanations = {
    'conv1': "Learns 32 edge/gradient detectors. (3×3×1+1)×32 = 320 params",
    'bn1':   "Normalizes conv output. 4 params per channel × 32 = 128",
    'relu1': "Non-linearity. Zero trainable params.",
    'pool1': "Downsamples 28×28 → 14×14. Zero params.",
    'conv2': "Learns 64 texture detectors. (3×3×32+1)×64 = 18,496 params",
    'bn2':   "Normalizes. 4×64 = 256 params",
    'relu2': "Non-linearity. Zero params.",
    'pool2': "Downsamples 14×14 → 7×7. Zero params.",
    'conv3': "Learns 128 high-level features. (3×3×64+1)×128 = 73,856 params",
    'bn3':   "Normalizes. 4×128 = 512 params",
    'relu3': "Non-linearity. Zero params.",
    'gap':   "Averages each 7×7 map to 1 value. Zero params.",
    'fc1':   "Combines features. (128+1)×128 = 16,512 params",
    'output':"Class probabilities. (128+1)×10 = 1,290 params",
}

for layer in tiny_cnn.layers:
    n = layer.name
    params = layer.count_params()
    out_shape = layer.output_shape
    explanation = layer_explanations.get(n, "")
    print(f"  [{n:<10}]  output={str(out_shape[1:]):<18}  "
          f"params={params:>7,}  ← {explanation}")

print("\n  Manual parameter count verification:")
manual_total = sum([320, 128, 0, 0, 18496, 256, 0, 0, 73856, 512, 0, 0, 16512, 1290])
print(f"    Trainable params (manual):  {manual_total:,}")
print(f"    Keras count (trainable):    "
      f"{sum(v.numpy().size for v in tiny_cnn.trainable_variables):,}")

# ──────────────────────────────────────────────────────────────
# SECTION 4: Compile and Do a Single Forward Pass
# ──────────────────────────────────────────────────────────────
print("\n--- Section 4: Compile and Forward Pass ---")

tiny_cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("  Model compiled with Adam (lr=0.001), SparseCategoricalCrossentropy")

(X_train, y_train), _ = tf.keras.datasets.mnist.load_data()
X_sample = X_train[:8].astype('float32') / 255.0
X_sample = X_sample[..., np.newaxis]  # (8, 28, 28, 1)
y_sample = y_train[:8]

print(f"\n  Input batch shape: {X_sample.shape}")
print(f"  True labels:       {y_sample.tolist()}")

predictions = tiny_cnn.predict(X_sample, verbose=0)
print(f"  Output shape:      {predictions.shape}")

print("\n  Predictions for first 8 samples:")
print(f"  {'Sample':>8} | {'True Label':>12} | {'Predicted':>12} | {'Confidence':>12}")
print("  " + "-" * 52)
for i in range(8):
    pred_class = np.argmax(predictions[i])
    confidence = predictions[i, pred_class]
    match = "✓" if pred_class == y_sample[i] else "✗"
    print(f"  {i:>8} | {y_sample[i]:>12} | {pred_class:>12} {match} | {confidence:>11.2%}")

# Intermediate layer outputs (feature maps)
print("\n  Intermediate feature map shapes (first sample):")
layer_model = tf.keras.Model(
    inputs=tiny_cnn.input,
    outputs=[l.output for l in tiny_cnn.layers if 'conv' in l.name or 'pool' in l.name or 'gap' in l.name]
)
intermediates = layer_model.predict(X_sample[:1], verbose=0)
for layer, feat in zip([l for l in tiny_cnn.layers if 'conv' in l.name or 'pool' in l.name or 'gap' in l.name], intermediates):
    print(f"    {layer.name:<12}: {feat.shape}")

# ──────────────────────────────────────────────────────────────
# SECTION 5: Train on MNIST for 3 Epochs and Plot Accuracy
# ──────────────────────────────────────────────────────────────
print("\n--- Section 5: Training on MNIST for 3 Epochs ---")

(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train_full = X_train_full.astype('float32') / 255.0
X_test       = X_test.astype('float32')       / 255.0

X_train_full = X_train_full[..., np.newaxis]
X_test       = X_test[..., np.newaxis]

# Use 10K samples for speed
X_tr = X_train_full[:10000]
y_tr = y_train_full[:10000]
X_val = X_train_full[10000:12000]
y_val = y_train_full[10000:12000]

print(f"  Training on {len(X_tr)} samples, validating on {len(X_val)}")

fresh_cnn = build_tiny_cnn()
fresh_cnn.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

history = fresh_cnn.fit(
    X_tr, y_tr,
    epochs=3,
    batch_size=64,
    validation_data=(X_val, y_val),
    verbose=1
)

print("\n  Training results:")
print(f"  {'Epoch':>6} | {'Train Loss':>12} | {'Train Acc':>12} | "
      f"{'Val Loss':>10} | {'Val Acc':>10}")
print("  " + "-" * 60)
for ep in range(3):
    print(f"  {ep+1:>6} | {history.history['loss'][ep]:>12.4f} | "
          f"{history.history['accuracy'][ep]:>11.4f} | "
          f"{history.history['val_loss'][ep]:>10.4f} | "
          f"{history.history['val_accuracy'][ep]:>9.4f}")

test_loss, test_acc = fresh_cnn.evaluate(X_test, y_test, verbose=0)
print(f"\n  Test accuracy after 3 epochs: {test_acc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("MNIST Training (3 Epochs) — TinyCNN", fontsize=13, fontweight='bold')

epochs_range = range(1, 4)
axes[0].plot(epochs_range, history.history['loss'],     'b-o', label='Train Loss')
axes[0].plot(epochs_range, history.history['val_loss'], 'r--o', label='Val Loss')
axes[0].set_title("Loss Curves")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history.history['accuracy'],     'b-o', label='Train Accuracy')
axes[1].plot(epochs_range, history.history['val_accuracy'], 'r--o', label='Val Accuracy')
axes[1].set_title("Accuracy Curves")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("section8_training_curves.png", dpi=100, bbox_inches='tight')
plt.close()
print("  Saved: section8_training_curves.png")

print("\n" + "=" * 60)
print("Lesson 8 Complete!")
print("Generated images:")
print("  section8_training_curves.png")
print("=" * 60)

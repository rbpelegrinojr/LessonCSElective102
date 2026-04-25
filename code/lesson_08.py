"""
Lesson 08: Building Your First Complete CNN
============================================
This module demonstrates how to build complete CNN architectures using Keras.
It covers the Sequential API, Functional API, model.summary() interpretation,
training on MNIST, learning curve visualization, and implements LeNet-5
as the classic historical example. Multiple CNN depths are compared.
"""

# === IMPORTS ===
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow version:", tf.__version__)
print("=" * 60)


# =============================================================================
# === SECTION 1: BUILD CNN WITH SEQUENTIAL API AND MODEL.SUMMARY() ===
# =============================================================================

print("\n[SECTION 1] Building a Complete CNN with the Sequential API")
print("-" * 60)

# Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize to [0, 1] and add channel dimension
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0
x_train = x_train[..., np.newaxis]  # (60000, 28, 28) → (60000, 28, 28, 1)
x_test  = x_test[...,  np.newaxis]

print(f"  Training data: {x_train.shape}, labels: {y_train.shape}")
print(f"  Test data:     {x_test.shape}")

# Build a complete CNN using the Sequential API
# Pattern: Conv → ReLU → Pool, repeated, then Flatten → Dense → Output
model_sequential = keras.Sequential([
    # ----- BLOCK 1: Learn basic edges and gradients -----
    layers.Conv2D(32, (3, 3), activation='relu', padding='valid',
                  input_shape=(28, 28, 1),
                  name='conv1'),          # 28×28×1 → 26×26×32
    layers.MaxPooling2D((2, 2),
                        name='pool1'),   # 26×26×32 → 13×13×32

    # ----- BLOCK 2: Learn textures and shapes -----
    layers.Conv2D(64, (3, 3), activation='relu', padding='valid',
                  name='conv2'),          # 13×13×32 → 11×11×64
    layers.MaxPooling2D((2, 2),
                        name='pool2'),   # 11×11×64 → 5×5×64

    # ----- CLASSIFICATION HEAD -----
    layers.Flatten(name='flatten'),       # 5×5×64 = 1600 → 1600
    layers.Dense(128, activation='relu',
                 name='dense1'),          # 1600 → 128
    layers.Dropout(0.4, name='dropout'), # regularization: randomly zero 40% of units
    layers.Dense(10, activation='softmax',
                 name='output'),          # 128 → 10 class probabilities
], name='Complete_CNN_Sequential')

print("\n  model.summary() output:")
print("  " + "="*65)
model_sequential.summary()

# Compile the model
model_sequential.compile(
    optimizer='adam',                           # adaptive learning rate optimizer
    loss='sparse_categorical_crossentropy',     # for integer labels (not one-hot)
    metrics=['accuracy']
)


# =============================================================================
# === SECTION 2: FUNCTIONAL API EQUIVALENT ===
# =============================================================================

print("\n[SECTION 2] Building the Same CNN with the Functional API")
print("-" * 60)
print("  The Functional API makes the computation graph explicit.")
print("  Each layer call takes a tensor and returns a tensor.")

# Functional API: define input → apply layers → define model
inputs = keras.Input(shape=(28, 28, 1), name='image_input')

# Block 1
x = layers.Conv2D(32, (3, 3), activation='relu', padding='valid', name='conv1_f')(inputs)
x = layers.MaxPooling2D((2, 2), name='pool1_f')(x)

# Block 2
x = layers.Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv2_f')(x)
x = layers.MaxPooling2D((2, 2), name='pool2_f')(x)

# Classification head
x = layers.Flatten(name='flatten_f')(x)
x = layers.Dense(128, activation='relu', name='dense1_f')(x)
x = layers.Dropout(0.4, name='dropout_f')(x)
outputs = layers.Dense(10, activation='softmax', name='output_f')(x)

model_functional = keras.Model(inputs=inputs, outputs=outputs,
                                name='Complete_CNN_Functional')

model_functional.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Verify both models have the same parameter count
print(f"\n  Sequential model params:  {model_sequential.count_params():,}")
print(f"  Functional model params:  {model_functional.count_params():,}")
print("  → Identical architectures, just different API syntax!")

# --- Manual shape calculation walkthrough ---
print("\n  --- Manual Shape Calculation Walkthrough ---")
print(f"  Input:                    28×28×1")
print(f"  After Conv2D(32, 3×3, valid): (28-3+1)×(28-3+1)×32 = 26×26×32")
print(f"  After MaxPool(2×2):       (26//2)×(26//2)×32 = 13×13×32")
print(f"  After Conv2D(64, 3×3, valid): (13-3+1)×(13-3+1)×64 = 11×11×64")
print(f"  After MaxPool(2×2):       (11//2)×(11//2)×64 = 5×5×64")
print(f"  After Flatten:            5×5×64 = 1600")
print(f"  After Dense(128):         128")
print(f"  After Dense(10):          10  ← one probability per digit class")

# --- Parameter counting demonstration ---
print("\n  --- Parameter Count Explanation ---")
conv1_params = (3*3*1 + 1) * 32   # (filter_h × filter_w × in_channels + bias) × filters
conv2_params = (3*3*32 + 1) * 64  # input to conv2 has 32 channels
flatten_to_dense = 5*5*64         # = 1600 input features to Dense
dense1_params = (flatten_to_dense + 1) * 128  # +1 for bias per neuron
dense2_params = (128 + 1) * 10
total = conv1_params + conv2_params + dense1_params + dense2_params

print(f"  Conv1(32 filters, 3×3, 1 channel): (3×3×1+1)×32 = {conv1_params}")
print(f"  Conv2(64 filters, 3×3, 32 channels): (3×3×32+1)×64 = {conv2_params}")
print(f"  Dense1(1600→128):  (1600+1)×128 = {dense1_params}")
print(f"  Dense2(128→10):    (128+1)×10   = {dense2_params}")
print(f"  Total (manual):    {total:,}")


# =============================================================================
# === SECTION 3: LENET-5 IMPLEMENTATION ===
# =============================================================================

print("\n[SECTION 3] LeNet-5 — The Classic CNN Architecture (1998)")
print("-" * 60)
print("  LeNet-5 was the first successful CNN, used for postal digit recognition.")
print("  Original used tanh + average pooling + 5×5 filters.")
print("  We modernize it with ReLU + max pooling.")

# Pad MNIST images from 28×28 to 32×32 to match original LeNet-5 input
x_train_lenet = np.pad(x_train, ((0,0),(2,2),(2,2),(0,0)), mode='constant')
x_test_lenet  = np.pad(x_test,  ((0,0),(2,2),(2,2),(0,0)), mode='constant')
print(f"  Padded input shape: {x_train_lenet.shape}")

# Build LeNet-5 architecture (modernized with ReLU and MaxPool)
lenet5 = keras.Sequential([
    # C1: 6 filters, 5×5 kernel  (32×32 → 28×28×6)
    layers.Conv2D(6, (5, 5), activation='relu',
                  input_shape=(32, 32, 1), name='C1'),
    # S2: Average pooling 2×2     (28×28 → 14×14×6)
    layers.AveragePooling2D((2, 2), name='S2'),

    # C3: 16 filters, 5×5 kernel  (14×14 → 10×10×16)
    layers.Conv2D(16, (5, 5), activation='relu', name='C3'),
    # S4: Average pooling 2×2     (10×10 → 5×5×16)
    layers.AveragePooling2D((2, 2), name='S4'),

    # Flatten                     (5×5×16 = 400)
    layers.Flatten(name='Flatten'),

    # F5: Fully connected 120 neurons
    layers.Dense(120, activation='relu', name='F5'),

    # F6: Fully connected 84 neurons
    layers.Dense(84, activation='relu', name='F6'),

    # Output: 10 classes
    layers.Dense(10, activation='softmax', name='Output'),
], name='LeNet_5')

lenet5.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

print("\n  LeNet-5 Summary:")
lenet5.summary()


# =============================================================================
# === SECTION 4: TRAIN AND EVALUATE — LEARNING CURVES AND PREDICTIONS ===
# =============================================================================

print("\n[SECTION 4] Training the CNN and Visualizing Results")
print("-" * 60)

# Use a subset for faster training in demo mode
TRAIN_SIZE = 20000
TEST_SIZE  = 5000
EPOCHS = 12

x_tr = x_train[:TRAIN_SIZE]
y_tr = y_train[:TRAIN_SIZE]
x_te = x_test[:TEST_SIZE]
y_te = y_test[:TEST_SIZE]

print(f"  Training on {TRAIN_SIZE} samples for {EPOCHS} epochs...")

history = model_sequential.fit(
    x_tr, y_tr,
    epochs=EPOCHS,
    batch_size=64,
    validation_split=0.1,  # use 10% of training data for validation
    verbose=1
)

# Evaluate on test set
test_loss, test_acc = model_sequential.evaluate(x_te, y_te, verbose=0)
print(f"\n  Test Loss:     {test_loss:.4f}")
print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# --- Plot training history ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Lesson 08: CNN Training on MNIST — Learning Curves',
             fontsize=14, fontweight='bold')

epochs_range = range(1, EPOCHS + 1)

# Loss curves
ax1.plot(epochs_range, history.history['loss'],
         'steelblue', linewidth=2, label='Training Loss')
ax1.plot(epochs_range, history.history['val_loss'],
         'steelblue', linewidth=2, linestyle='--', label='Validation Loss')
ax1.set_title('Loss Over Epochs', fontsize=12)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (Cross-Entropy)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy curves
ax2.plot(epochs_range, history.history['accuracy'],
         'darkorange', linewidth=2, label='Training Accuracy')
ax2.plot(epochs_range, history.history['val_accuracy'],
         'darkorange', linewidth=2, linestyle='--', label='Validation Accuracy')
ax2.axhline(y=test_acc, color='green', linewidth=1.5,
            linestyle=':', label=f'Test Acc: {test_acc:.3f}')
ax2.set_title('Accuracy Over Epochs', fontsize=12)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.8, 1.02)

plt.tight_layout()
plt.savefig('code/lesson_08_training_curves.png', dpi=120, bbox_inches='tight')
print("  Saved: code/lesson_08_training_curves.png")
plt.close()

# --- Visualize predictions on test samples ---
predictions = model_sequential.predict(x_te[:16], verbose=0)
pred_classes = np.argmax(predictions, axis=1)  # class with highest probability

fig, axes = plt.subplots(2, 8, figsize=(16, 5))
fig.suptitle('Lesson 08: CNN Predictions on MNIST Test Images',
             fontsize=13, fontweight='bold')

for i, ax in enumerate(axes.flat):
    ax.imshow(x_te[i, :, :, 0], cmap='gray')
    true_label = y_te[i]
    pred_label = pred_classes[i]
    confidence = predictions[i][pred_label]

    # Color green for correct, red for incorrect predictions
    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(f'P:{pred_label} T:{true_label}\n{confidence:.2f}',
                 fontsize=8, color=color)
    ax.axis('off')

plt.tight_layout()
plt.savefig('code/lesson_08_predictions.png', dpi=120, bbox_inches='tight')
print("  Saved: code/lesson_08_predictions.png")
plt.close()

# Count correct vs incorrect
n_correct = np.sum(pred_classes == y_te[:16])
print(f"\n  Out of first 16 test images: {n_correct}/16 correct")
print(f"  (Green titles = correct, Red titles = wrong prediction)")

print("\n" + "=" * 60)
print("Lesson 08 complete! Generated files:")
print("  - code/lesson_08_training_curves.png")
print("  - code/lesson_08_predictions.png")
print("Key takeaways:")
print("  1. Complete CNN: [Conv→ReLU→Pool] × N → Flatten → Dense → Softmax")
print("  2. Sequential API: simple stack; Functional API: arbitrary graphs")
print("  3. model.summary() shows shapes and parameter counts")
print("  4. Dense layers dominate parameter count after Flatten")
print("  5. LeNet-5 (1998) established the template all CNNs still follow")

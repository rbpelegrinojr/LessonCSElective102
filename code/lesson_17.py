"""
Lesson 17: Fine-Tuning Pretrained Models
Demonstrates two-phase training: feature extraction then fine-tuning with MobileNetV2.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

print("TensorFlow version:", tf.__version__)
print("=" * 60)

# === SECTION 1: Phase 1 - Feature Extraction ===
print("\n=== SECTION 1: Phase 1 - Feature Extraction ===")

# Load MobileNetV2 pretrained on ImageNet without the top classification layer
base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')

# Freeze ALL base model layers so we only train the new head
base_model.trainable = False
print(f"Base model loaded: {base_model.name}")
print(f"Base model layers frozen: {len(base_model.layers)}")

# Build full model: pretrained base + custom classification head
inputs = keras.Input(shape=(96, 96, 3))
x = base_model(inputs, training=False)          # Run base in inference mode
x = layers.GlobalAveragePooling2D()(x)          # Pool spatial features to vector
x = layers.Dense(128, activation='relu')(x)     # Intermediate dense layer
outputs = layers.Dense(10, activation='softmax')(x)  # 10-class output
model = keras.Model(inputs, outputs)

# Compile with relatively high learning rate for feature extraction phase
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Create synthetic CIFAR-10-like data: 100 samples, 96x96 RGB images
np.random.seed(42)
x_train = np.random.rand(100, 96, 96, 3).astype('float32')
y_train = np.random.randint(0, 10, size=(100,)).astype('int32')
x_val = np.random.rand(20, 96, 96, 3).astype('float32')
y_val = np.random.randint(0, 10, size=(20,)).astype('int32')

print("\nTraining Phase 1: Feature Extraction (3 epochs)...")
history_phase1 = model.fit(x_train, y_train,
                           validation_data=(x_val, y_val),
                           epochs=3, batch_size=16, verbose=1)

# Print trainable status for first few layers
print("\nLayer trainable status (first 5):")
for layer in model.layers[:5]:
    print(f"  {layer.name}: trainable={layer.trainable}")

print("\nPhase 1 training complete.")

# === SECTION 2: Inspect Frozen vs Trainable Layers ===
print("\n=== SECTION 2: Inspect Frozen vs Trainable Layers ===")

# Count total, trainable, and non-trainable parameters
total_params = model.count_params()
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
non_trainable_params = total_params - trainable_params

print(f"Total parameters:         {total_params:,}")
print(f"Trainable parameters:     {trainable_params:,}")
print(f"Non-trainable parameters: {non_trainable_params:,}")
frozen_pct = 100.0 * non_trainable_params / total_params
print(f"Percentage frozen:        {frozen_pct:.1f}%")

# Show first 5 layers
print("\nFirst 5 layers:")
for layer in model.layers[:5]:
    print(f"  {layer.name:40s} trainable={layer.trainable}")

# Show last 5 layers
print("\nLast 5 layers:")
for layer in model.layers[-5:]:
    print(f"  {layer.name:40s} trainable={layer.trainable}")

# === SECTION 3: Phase 2 - Fine-Tuning ===
print("\n=== SECTION 3: Phase 2 - Fine-Tuning ===")

# Unfreeze the entire base model first
base_model.trainable = True

# Re-freeze all but the last 30 layers
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Count how many layers were unfrozen
unfrozen = [l for l in base_model.layers if l.trainable]
print(f"Unfroze last 30 layers of base model ({len(unfrozen)} layers)")
print("Sample unfrozen layers:")
for layer in base_model.layers[-5:]:
    print(f"  {layer.name}: trainable={layer.trainable}")

# Recompile with much smaller learning rate to avoid destroying pretrained weights
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nTraining Phase 2: Fine-Tuning (3 epochs)...")
history_phase2 = model.fit(x_train, y_train,
                           validation_data=(x_val, y_val),
                           epochs=3, batch_size=16, verbose=1)

print("\nPhase 2 fine-tuning complete.")

# === SECTION 4: Compare Phase 1 vs Phase 2 Learning Curves ===
print("\n=== SECTION 4: Compare Phase 1 vs Phase 2 Learning Curves ===")

# Extract metrics from both phases
p1_loss = history_phase1.history['loss']
p1_val_loss = history_phase1.history['val_loss']
p1_acc = history_phase1.history['accuracy']
p1_val_acc = history_phase1.history['val_accuracy']

p2_loss = history_phase2.history['loss']
p2_val_loss = history_phase2.history['val_loss']
p2_acc = history_phase2.history['accuracy']
p2_val_acc = history_phase2.history['val_accuracy']

# Epochs for each phase
epochs_p1 = range(1, len(p1_loss) + 1)
epochs_p2 = range(len(p1_loss) + 1, len(p1_loss) + len(p2_loss) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
axes[0].plot(epochs_p1, p1_loss, 'b-o', label='Phase1 Train Loss')
axes[0].plot(epochs_p1, p1_val_loss, 'b--o', label='Phase1 Val Loss')
axes[0].plot(epochs_p2, p2_loss, 'r-o', label='Phase2 Train Loss')
axes[0].plot(epochs_p2, p2_val_loss, 'r--o', label='Phase2 Val Loss')
axes[0].axvline(x=len(p1_loss) + 0.5, color='green', linestyle=':', label='Fine-tune start')
axes[0].set_title('Loss: Phase 1 vs Phase 2')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

# Accuracy plot
axes[1].plot(epochs_p1, p1_acc, 'b-o', label='Phase1 Train Acc')
axes[1].plot(epochs_p1, p1_val_acc, 'b--o', label='Phase1 Val Acc')
axes[1].plot(epochs_p2, p2_acc, 'r-o', label='Phase2 Train Acc')
axes[1].plot(epochs_p2, p2_val_acc, 'r--o', label='Phase2 Val Acc')
axes[1].axvline(x=len(p1_loss) + 0.5, color='green', linestyle=':', label='Fine-tune start')
axes[1].set_title('Accuracy: Phase 1 vs Phase 2')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.suptitle('Lesson 17: Fine-Tuning Learning Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/lesson_17_curves.png', dpi=100)
print("Saved learning curves to /tmp/lesson_17_curves.png")

best_p1 = max(p1_val_acc)
best_p2 = max(p2_val_acc)
print(f"Best validation accuracy Phase 1: {best_p1:.4f}")
print(f"Best validation accuracy Phase 2: {best_p2:.4f}")

# === SECTION 5: Save and Reload Fine-Tuned Model ===
print("\n=== SECTION 5: Save and Reload Fine-Tuned Model ===")

# Save the fine-tuned model in HDF5 format
model.save('/tmp/fine_tuned_model.h5')
print("Model saved to /tmp/fine_tuned_model.h5")

# Reload the model from disk
loaded_model = keras.models.load_model('/tmp/fine_tuned_model.h5')
print("Model loaded from /tmp/fine_tuned_model.h5")

# Run inference on 5 sample images
sample_images = x_val[:5]
original_preds = model.predict(sample_images, verbose=0)
loaded_preds = loaded_model.predict(sample_images, verbose=0)

print("\nInference comparison (original vs loaded):")
for i in range(5):
    orig_class = np.argmax(original_preds[i])
    load_class = np.argmax(loaded_preds[i])
    match = "✓ MATCH" if orig_class == load_class else "✗ MISMATCH"
    print(f"  Sample {i+1}: original={orig_class}, loaded={load_class}  {match}")

# Verify all predictions match
all_match = np.allclose(original_preds, loaded_preds, atol=1e-5)
print(f"\nAll predictions match: {all_match}")

plt.show()
print("\nLesson 17 complete!")

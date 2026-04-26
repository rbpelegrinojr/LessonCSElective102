"""
Lesson 15: Batch Normalization & Dropout
=========================================
Demonstrates internal covariate shift (without vs with BN),
BN layer analysis, BN placement (before/after activation),
SpatialDropout2D vs Dropout, and a full CIFAR-10 model with BN+Dropout.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

print("=" * 60)
print("LESSON 15: Batch Normalization & Dropout")
print("=" * 60)

# ── Load CIFAR-10 ─────────────────────────────────────────────
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0
y_train = y_train.flatten()
y_test  = y_test.flatten()

N_TRAIN = 15000
X_tr  = X_train[:N_TRAIN]
y_tr  = y_train[:N_TRAIN]
X_val = X_train[N_TRAIN:N_TRAIN + 3000]
y_val = y_train[N_TRAIN:N_TRAIN + 3000]

EPOCHS = 20
BATCH  = 64

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ──────────────────────────────────────────────────────────────
# SECTION 1: Internal Covariate Shift — With vs Without BN
# ──────────────────────────────────────────────────────────────
print("\n--- Section 1: Internal Covariate Shift Demo ---")

# Deep MLP WITHOUT batch normalisation — prone to covariate shift and instability
def deep_mlp_no_bn(n_layers=8, units=128):
    """Deep MLP without BN — activations will shift wildly during training."""
    inputs = keras.Input(shape=(3072,))
    x = inputs
    for _ in range(n_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    return keras.Model(inputs, outputs, name='no_bn')

# Deep MLP WITH batch normalisation — stabilised activations
def deep_mlp_with_bn(n_layers=8, units=128):
    """Deep MLP with BN after each Dense layer."""
    inputs = keras.Input(shape=(3072,))
    x = inputs
    for _ in range(n_layers):
        x = keras.layers.Dense(units, use_bias=False)(x)   # bias redundant with BN's β
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    return keras.Model(inputs, outputs, name='with_bn')

# Flatten CIFAR images for MLP input
X_tr_flat  = X_tr.reshape(-1, 3072)
X_val_flat = X_val.reshape(-1, 3072)

# Record activation mean/std at a middle layer after each epoch
class ActivationMonitor(keras.callbacks.Callback):
    """Callback that records mean and std of a specified layer's output."""
    def __init__(self, layer_name, x_sample):
        super().__init__()
        self.layer_name = layer_name
        self.x_sample   = x_sample
        self.means = []
        self.stds  = []

    def on_epoch_end(self, epoch, logs=None):
        # Build a sub-model that outputs the monitored layer
        sub_model = keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(self.layer_name).output
        )
        activations = sub_model.predict(self.x_sample, verbose=0)
        self.means.append(float(np.mean(activations)))
        self.stds.append(float(np.std(activations)))

x_monitor = X_tr_flat[:200]   # small sample for monitoring speed

# Train WITHOUT BN
print("  Training deep MLP WITHOUT batch normalisation...")
model_no_bn = deep_mlp_no_bn()
# Name the layer we want to monitor (4th dense block)
monitor_no_bn = ActivationMonitor(
    layer_name=model_no_bn.layers[5].name,  # ~middle layer
    x_sample=x_monitor
)
model_no_bn.compile(optimizer=keras.optimizers.Adam(1e-3),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
hist_no_bn = model_no_bn.fit(
    X_tr_flat, y_tr,
    validation_data=(X_val_flat, y_val),
    epochs=EPOCHS, batch_size=BATCH,
    callbacks=[monitor_no_bn], verbose=0
)
print(f"    Final val accuracy (no BN): {hist_no_bn.history['val_accuracy'][-1]:.4f}")

# Train WITH BN
print("  Training deep MLP WITH batch normalisation...")
model_with_bn = deep_mlp_with_bn()
monitor_with_bn = ActivationMonitor(
    layer_name=model_with_bn.layers[5].name,
    x_sample=x_monitor
)
model_with_bn.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
hist_with_bn = model_with_bn.fit(
    X_tr_flat, y_tr,
    validation_data=(X_val_flat, y_val),
    epochs=EPOCHS, batch_size=BATCH,
    callbacks=[monitor_with_bn], verbose=0
)
print(f"    Final val accuracy (with BN): {hist_with_bn.history['val_accuracy'][-1]:.4f}")

# Plot activation statistics and validation accuracy
epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(epochs_range, monitor_no_bn.means,   'b-', linewidth=2, label='No BN')
axes[0].plot(epochs_range, monitor_with_bn.means, 'r-', linewidth=2, label='With BN')
axes[0].set_title('Activation Mean at Middle Layer')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Mean')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, monitor_no_bn.stds,   'b-', linewidth=2, label='No BN')
axes[1].plot(epochs_range, monitor_with_bn.stds, 'r-', linewidth=2, label='With BN')
axes[1].set_title('Activation Std at Middle Layer')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Std')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(hist_no_bn.history['val_accuracy'],   'b-', linewidth=2, label='No BN')
axes[2].plot(hist_with_bn.history['val_accuracy'], 'r-', linewidth=2, label='With BN')
axes[2].set_title('Validation Accuracy')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Accuracy')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Internal Covariate Shift: With vs Without Batch Normalization', fontsize=12)
plt.tight_layout()
plt.savefig('lesson15_covariate_shift.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson15_covariate_shift.png")

# ──────────────────────────────────────────────────────────────
# SECTION 2: Batch Normalization Layer Analysis
# ──────────────────────────────────────────────────────────────
print("\n--- Section 2: Batch Normalization Layer Analysis ---")

# Inspect BN layer parameters after training
for layer in model_with_bn.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        weights = layer.get_weights()
        # weights: [gamma, beta, moving_mean, moving_variance]
        gamma        = weights[0]
        beta         = weights[1]
        running_mean = weights[2]
        running_var  = weights[3]

        print(f"\n  BN Layer: {layer.name}")
        print(f"    Trainable parameters:")
        print(f"      gamma (scale):  mean={gamma.mean():.4f}  std={gamma.std():.4f}  "
              f"range=[{gamma.min():.4f}, {gamma.max():.4f}]")
        print(f"      beta  (shift):  mean={beta.mean():.4f}  std={beta.std():.4f}  "
              f"range=[{beta.min():.4f}, {beta.max():.4f}]")
        print(f"    Running statistics (used at inference):")
        print(f"      moving_mean:    mean={running_mean.mean():.4f}  std={running_mean.std():.4f}")
        print(f"      moving_var:     mean={running_var.mean():.4f}  std={running_var.std():.4f}")
        break  # just show the first BN layer

# Visualise gamma and beta distributions for all BN layers
bn_layers = [l for l in model_with_bn.layers if isinstance(l, keras.layers.BatchNormalization)]
fig, axes = plt.subplots(2, min(4, len(bn_layers)), figsize=(14, 6))
if len(bn_layers) < 4:
    axes = np.array(axes).reshape(2, -1)

for col, bn_layer in enumerate(bn_layers[:4]):
    gamma, beta, running_mean, running_var = bn_layer.get_weights()
    axes[0, col].hist(gamma, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, col].set_title(f'{bn_layer.name}\nGamma (scale)', fontsize=8)
    axes[0, col].axvline(1.0, color='red', linestyle='--', linewidth=1, label='init=1')
    axes[0, col].legend(fontsize=7)
    axes[0, col].grid(True, alpha=0.3)

    axes[1, col].hist(beta, bins=30, color='darkorange', alpha=0.7, edgecolor='black')
    axes[1, col].set_title(f'{bn_layer.name}\nBeta (shift)', fontsize=8)
    axes[1, col].axvline(0.0, color='red', linestyle='--', linewidth=1, label='init=0')
    axes[1, col].legend(fontsize=7)
    axes[1, col].grid(True, alpha=0.3)

plt.suptitle('BN Learned Parameters: Gamma and Beta Distributions', fontsize=12)
plt.tight_layout()
plt.savefig('lesson15_bn_parameters.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson15_bn_parameters.png")

# ──────────────────────────────────────────────────────────────
# SECTION 3: BN Placement — Before vs After Activation
# ──────────────────────────────────────────────────────────────
print("\n--- Section 3: BN Before vs After Activation ---")

def cnn_bn_before_activation():
    """Conv → BN → ReLU  (original BN paper, most common convention)."""
    inputs = keras.Input(shape=(32, 32, 3))
    x = keras.layers.Conv2D(64, 3, padding='same', use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    return keras.Model(inputs, outputs, name='bn_before_relu')

def cnn_bn_after_activation():
    """Conv → ReLU → BN  (alternative ordering)."""
    inputs = keras.Input(shape=(32, 32, 3))
    x = keras.layers.Conv2D(64, 3, padding='same')(inputs)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(128, 3, padding='same')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    return keras.Model(inputs, outputs, name='bn_after_relu')

def cnn_no_bn():
    """Same CNN without any BN (baseline)."""
    return keras.Sequential([
        keras.layers.Conv2D(64, 3, padding='same', activation='relu',
                            input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ], name='no_bn')

bn_placement_configs = {
    'No BN':           cnn_no_bn(),
    'BN Before ReLU':  cnn_bn_before_activation(),
    'BN After ReLU':   cnn_bn_after_activation(),
}

histories_bn = {}
for name, m in bn_placement_configs.items():
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    print(f"  Training: {name}...")
    h = m.fit(X_tr, y_tr,
              validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH, verbose=0)
    histories_bn[name] = h
    print(f"    val accuracy = {h.history['val_accuracy'][-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = ['blue', 'orange', 'green']
for (name, h), color in zip(histories_bn.items(), colors):
    axes[0].plot(h.history['val_accuracy'], color=color, linewidth=2, label=name)
    axes[1].plot(h.history['val_loss'],     color=color, linewidth=2, label=name)

axes[0].set_title('Val Accuracy: BN Placement Comparison')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Val Loss: BN Placement Comparison')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lesson15_bn_placement.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson15_bn_placement.png")

# ──────────────────────────────────────────────────────────────
# SECTION 4: SpatialDropout2D vs Standard Dropout
# ──────────────────────────────────────────────────────────────
print("\n--- Section 4: SpatialDropout2D vs Standard Dropout ---")

def cnn_with_spatial_dropout(rate=0.2):
    """CNN using SpatialDropout2D (drops entire feature maps)."""
    return keras.Sequential([
        keras.layers.Conv2D(64, 3, padding='same', activation='relu',
                            input_shape=(32, 32, 3)),
        keras.layers.SpatialDropout2D(rate),        # drops full channels
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        keras.layers.SpatialDropout2D(rate),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(rate * 2),             # heavier dropout for dense layer
        keras.layers.Dense(10, activation='softmax'),
    ], name='spatial_dropout')

def cnn_with_standard_dropout(rate=0.2):
    """CNN using standard Dropout (drops individual activations)."""
    return keras.Sequential([
        keras.layers.Conv2D(64, 3, padding='same', activation='relu',
                            input_shape=(32, 32, 3)),
        keras.layers.Dropout(rate),                 # drops individual pixels
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        keras.layers.Dropout(rate),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(rate * 2),
        keras.layers.Dense(10, activation='softmax'),
    ], name='standard_dropout')

def cnn_no_dropout():
    """CNN without any dropout (baseline for comparison)."""
    return keras.Sequential([
        keras.layers.Conv2D(64, 3, padding='same', activation='relu',
                            input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ], name='no_dropout')

dropout_configs = {
    'No Dropout':           cnn_no_dropout(),
    'Standard Dropout':     cnn_with_standard_dropout(0.2),
    'SpatialDropout2D':     cnn_with_spatial_dropout(0.2),
}

histories_do = {}
for name, m in dropout_configs.items():
    m.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    print(f"  Training: {name}...")
    h = m.fit(X_tr, y_tr,
              validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH, verbose=0)
    histories_do[name] = h
    train_acc = h.history['accuracy'][-1]
    val_acc   = h.history['val_accuracy'][-1]
    print(f"    train={train_acc:.4f}  val={val_acc:.4f}  gap={train_acc - val_acc:.4f}")

# Demonstrate inverted dropout scaling
print("\n  Inverted Dropout scaling demo:")
np.random.seed(0)
activations = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
p = 0.5   # drop rate

# Standard (scale at inference)
mask_standard = (np.random.rand(len(activations)) > p).astype(float)
dropped_standard = activations * mask_standard
print(f"  Original activations:        {activations}")
print(f"  Mask (drop {p*100:.0f}%):           {mask_standard}")
print(f"  Dropped (no rescaling):      {dropped_standard}  → mean={dropped_standard.mean():.2f}")

# Inverted dropout (scale at training time)
np.random.seed(0)
mask_inverted = (np.random.rand(len(activations)) > p).astype(float)
dropped_inverted = activations * mask_inverted / (1 - p)   # divide by keep prob
print(f"  Inverted drop (÷{1-p}):       {dropped_inverted}  → mean={dropped_inverted.mean():.2f}")
print(f"  No rescaling needed at inference: activations = {activations}  → mean={activations.mean():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = ['blue', 'orange', 'green']
for (name, h), color in zip(histories_do.items(), colors):
    axes[0].plot(h.history['val_accuracy'], color=color, linewidth=2, label=name)
    axes[1].plot(h.history['val_loss'],     color=color, linewidth=2, label=name)

axes[0].set_title('SpatialDropout2D vs Standard Dropout — Val Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title('SpatialDropout2D vs Standard Dropout — Val Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lesson15_spatial_dropout.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson15_spatial_dropout.png")

# ──────────────────────────────────────────────────────────────
# SECTION 5: Full Model — BN + Dropout on CIFAR-10
# ──────────────────────────────────────────────────────────────
print("\n--- Section 5: Full CNN with BN + Dropout on CIFAR-10 ---")

def build_full_model():
    """
    Production-style CNN combining BN and Dropout.
    Pattern per conv block: Conv → BN → ReLU → SpatialDropout2D
    Pattern per dense block: Dense → BN → ReLU → Dropout
    """
    inputs = keras.Input(shape=(32, 32, 3), name='input')

    # Block 1
    x = keras.layers.Conv2D(64, 3, padding='same', use_bias=False, name='conv1')(inputs)
    x = keras.layers.BatchNormalization(name='bn1')(x)
    x = keras.layers.Activation('relu', name='relu1')(x)
    x = keras.layers.Conv2D(64, 3, padding='same', use_bias=False, name='conv2')(x)
    x = keras.layers.BatchNormalization(name='bn2')(x)
    x = keras.layers.Activation('relu', name='relu2')(x)
    x = keras.layers.MaxPooling2D(2, name='pool1')(x)
    x = keras.layers.SpatialDropout2D(0.2, name='sdrop1')(x)   # BN before dropout

    # Block 2
    x = keras.layers.Conv2D(128, 3, padding='same', use_bias=False, name='conv3')(x)
    x = keras.layers.BatchNormalization(name='bn3')(x)
    x = keras.layers.Activation('relu', name='relu3')(x)
    x = keras.layers.Conv2D(128, 3, padding='same', use_bias=False, name='conv4')(x)
    x = keras.layers.BatchNormalization(name='bn4')(x)
    x = keras.layers.Activation('relu', name='relu4')(x)
    x = keras.layers.MaxPooling2D(2, name='pool2')(x)
    x = keras.layers.SpatialDropout2D(0.3, name='sdrop2')(x)

    # Block 3
    x = keras.layers.Conv2D(256, 3, padding='same', use_bias=False, name='conv5')(x)
    x = keras.layers.BatchNormalization(name='bn5')(x)
    x = keras.layers.Activation('relu', name='relu5')(x)

    # Global pooling + classifier head
    x = keras.layers.GlobalAveragePooling2D(name='gap')(x)
    x = keras.layers.Dense(512, use_bias=False, name='dense1')(x)
    x = keras.layers.BatchNormalization(name='bn6')(x)
    x = keras.layers.Activation('relu', name='relu6')(x)
    x = keras.layers.Dropout(0.5, name='drop1')(x)              # BN before dropout
    outputs = keras.layers.Dense(10, activation='softmax', name='output')(x)

    return keras.Model(inputs, outputs, name='full_bn_dropout_cnn')

full_model = build_full_model()
full_model.summary()

print(f"\nTotal parameters: {full_model.count_params():,}")

# Count trainable BN parameters
bn_params = sum(
    np.prod(w.shape) for l in full_model.layers
    if isinstance(l, keras.layers.BatchNormalization)
    for w in l.trainable_weights
)
print(f"BN trainable parameters (γ and β only): {bn_params:,}")

# Train with a higher learning rate (BN allows this)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=4, verbose=1, min_lr=1e-6
)
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=8, restore_best_weights=True
)

full_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining full model (BN + Dropout) on CIFAR-10 subset...")
hist_full = full_model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=40, batch_size=BATCH,
    callbacks=[reduce_lr, early_stop],
    verbose=1
)

best_val_acc = max(hist_full.history['val_accuracy'])
print(f"\nBest validation accuracy: {best_val_acc:.4f}")

# Evaluate on test set
test_loss, test_acc = full_model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

# Plot training curves
epochs_ran = len(hist_full.history['loss'])
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(hist_full.history['accuracy'],     'b-', linewidth=2, label='Train')
axes[0].plot(hist_full.history['val_accuracy'], 'r--', linewidth=2, label='Validation')
axes[0].axhline(test_acc, color='green', linestyle=':', linewidth=2,
                label=f'Test acc = {test_acc:.4f}')
axes[0].set_title('Full CNN (BN + Dropout) — Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(hist_full.history['loss'],     'b-', linewidth=2, label='Train Loss')
axes[1].plot(hist_full.history['val_loss'], 'r--', linewidth=2, label='Val Loss')
axes[1].set_title('Full CNN (BN + Dropout) — Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lesson15_full_model.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson15_full_model.png")

# --- Summary: compare all section results ---
print("\nSECTION SUMMARY")
print(f"  No BN (Section 1):          val acc = {hist_no_bn.history['val_accuracy'][-1]:.4f}")
print(f"  With BN (Section 1):        val acc = {hist_with_bn.history['val_accuracy'][-1]:.4f}")
print(f"  BN Before ReLU (Section 3): val acc = {histories_bn['BN Before ReLU'].history['val_accuracy'][-1]:.4f}")
print(f"  BN After ReLU (Section 3):  val acc = {histories_bn['BN After ReLU'].history['val_accuracy'][-1]:.4f}")
print(f"  Full BN+Dropout (Section 5): test acc = {test_acc:.4f}")

print("\n" + "=" * 60)
print("Lesson 15 complete. Outputs saved:")
print("  lesson15_covariate_shift.png")
print("  lesson15_bn_parameters.png")
print("  lesson15_bn_placement.png")
print("  lesson15_spatial_dropout.png")
print("  lesson15_full_model.png")
print("=" * 60)

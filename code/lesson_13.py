"""
Lesson 13: Overfitting & Regularization Techniques
====================================================
Demonstrates overfitting, L1/L2 regularization, Dropout,
early stopping, and a side-by-side comparison of all methods.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

print("=" * 60)
print("LESSON 13: Overfitting & Regularization Techniques")
print("=" * 60)

# ── Shared Data Setup ─────────────────────────────────────────
# Use a small subset of CIFAR-10 to make overfitting obvious
(X_all, y_all), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_all  = X_all.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_all  = y_all.flatten()
y_test = y_test.flatten()

# Small training set (1 000 examples) so overfitting is fast and visible
N_SMALL = 1000
X_tr_s = X_all[:N_SMALL]
y_tr_s = y_all[:N_SMALL]
X_val  = X_all[N_SMALL:N_SMALL + 2000]
y_val  = y_all[N_SMALL:N_SMALL + 2000]

EPOCHS = 40
BATCH  = 64

# ──────────────────────────────────────────────────────────────
# SECTION 1: Demonstrate Overfitting
# ──────────────────────────────────────────────────────────────
print("\n--- Section 1: Demonstrating Overfitting ---")

def large_model():
    """Deep, wide model — will easily overfit a small dataset."""
    return keras.Sequential([
        keras.layers.Conv2D(128, 3, padding='same', activation='relu',
                            input_shape=(32, 32, 3)),
        keras.layers.Conv2D(128, 3, activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
        keras.layers.Conv2D(256, 3, activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ])

model_overfit = large_model()
model_overfit.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

print(f"Model parameters: {model_overfit.count_params():,}")
print("Training large model on small dataset (overfitting expected)...")

hist_overfit = model_overfit.fit(
    X_tr_s, y_tr_s,
    validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH, verbose=0
)

train_acc_final = hist_overfit.history['accuracy'][-1]
val_acc_final   = hist_overfit.history['val_accuracy'][-1]
print(f"  Train Accuracy: {train_acc_final:.4f}")
print(f"  Val   Accuracy: {val_acc_final:.4f}")
print(f"  Overfitting gap: {train_acc_final - val_acc_final:.4f}")

# Plot diverging accuracy curves
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(hist_overfit.history['accuracy'],     'b-', label='Train', linewidth=2)
axes[0].plot(hist_overfit.history['val_accuracy'], 'r--', label='Validation', linewidth=2)
axes[0].fill_between(
    range(EPOCHS),
    hist_overfit.history['accuracy'],
    hist_overfit.history['val_accuracy'],
    alpha=0.15, color='red', label='Overfit gap'
)
axes[0].set_title('Overfitting: Training vs Validation Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(hist_overfit.history['loss'],     'b-', label='Train Loss', linewidth=2)
axes[1].plot(hist_overfit.history['val_loss'], 'r--', label='Val Loss',  linewidth=2)
axes[1].set_title('Overfitting: Training vs Validation Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lesson13_overfitting.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson13_overfitting.png")

# ──────────────────────────────────────────────────────────────
# SECTION 2: L1 and L2 Regularization
# ──────────────────────────────────────────────────────────────
print("\n--- Section 2: L1 and L2 Regularization ---")

def model_with_l1(lam=0.001):
    """Model with L1 weight regularization on Dense layers."""
    reg = keras.regularizers.l1(lam)
    return keras.Sequential([
        keras.layers.Conv2D(64, 3, padding='same', activation='relu',
                            input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu', kernel_regularizer=reg),
        keras.layers.Dense(10, activation='softmax'),
    ])

def model_with_l2(lam=0.001):
    """Model with L2 weight regularization on Dense layers."""
    reg = keras.regularizers.l2(lam)
    return keras.Sequential([
        keras.layers.Conv2D(64, 3, padding='same', activation='relu',
                            input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu', kernel_regularizer=reg),
        keras.layers.Dense(10, activation='softmax'),
    ])

def model_baseline():
    """Same architecture without any regularization."""
    return keras.Sequential([
        keras.layers.Conv2D(64, 3, padding='same', activation='relu',
                            input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ])

configs = {
    'No Regularization': model_baseline(),
    'L1 (λ=0.001)':      model_with_l1(0.001),
    'L2 (λ=0.001)':      model_with_l2(0.001),
}

histories_reg = {}
for name, m in configs.items():
    m.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    print(f"  Training: {name}...")
    h = m.fit(X_tr_s, y_tr_s,
              validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH, verbose=0)
    histories_reg[name] = h
    gap = h.history['accuracy'][-1] - h.history['val_accuracy'][-1]
    print(f"    Final train={h.history['accuracy'][-1]:.4f}  "
          f"val={h.history['val_accuracy'][-1]:.4f}  "
          f"gap={gap:.4f}")

# Examine weight magnitudes for L1 vs L2
print("\n  Weight analysis (L1 tends to zero out weights):")
for name, m in configs.items():
    dense_weights = m.layers[-2].get_weights()[0].flatten()  # Dense(256) kernel
    n_near_zero = np.sum(np.abs(dense_weights) < 0.01)
    print(f"  {name:<25}: weights near zero = {n_near_zero:5d} / {len(dense_weights)}")

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['blue', 'orange', 'green']
for (name, h), color in zip(histories_reg.items(), colors):
    ax.plot(h.history['val_accuracy'], color=color, linewidth=2, label=name)
ax.set_title('L1 vs L2 Regularization — Validation Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lesson13_l1_l2.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson13_l1_l2.png")

# ──────────────────────────────────────────────────────────────
# SECTION 3: Dropout
# ──────────────────────────────────────────────────────────────
print("\n--- Section 3: Dropout ---")

def model_with_dropout(rate=0.5):
    """Model with Dropout after Dense and Conv layers."""
    return keras.Sequential([
        keras.layers.Conv2D(64, 3, padding='same', activation='relu',
                            input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(rate / 2),                   # lighter dropout for conv
        keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(rate / 2),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(rate),                        # heavier dropout for dense
        keras.layers.Dense(10, activation='softmax'),
    ])

dropout_configs = {
    'No Dropout (rate=0)':    model_baseline(),
    'Dropout rate=0.3':       model_with_dropout(0.3),
    'Dropout rate=0.5':       model_with_dropout(0.5),
}

histories_drop = {}
for name, m in dropout_configs.items():
    m.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    print(f"  Training: {name}...")
    h = m.fit(X_tr_s, y_tr_s,
              validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH, verbose=0)
    histories_drop[name] = h
    gap = h.history['accuracy'][-1] - h.history['val_accuracy'][-1]
    print(f"    Final train={h.history['accuracy'][-1]:.4f}  "
          f"val={h.history['val_accuracy'][-1]:.4f}  "
          f"gap={gap:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for (name, h), color in zip(histories_drop.items(), ['blue', 'orange', 'green']):
    axes[0].plot(h.history['accuracy'],     color=color, linestyle='-',  label=f'{name} (train)', linewidth=2)
    axes[0].plot(h.history['val_accuracy'], color=color, linestyle='--', label=f'{name} (val)',   linewidth=2)
    axes[1].plot(h.history['val_loss'],     color=color, linewidth=2, label=name)

axes[0].set_title('Dropout: Train vs Validation Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(fontsize=7)
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Dropout: Validation Loss Comparison')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Val Loss')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lesson13_dropout.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson13_dropout.png")

# ──────────────────────────────────────────────────────────────
# SECTION 4: Early Stopping Callback
# ──────────────────────────────────────────────────────────────
print("\n--- Section 4: Early Stopping ---")

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,               # stop if val_loss doesn't improve for 8 epochs
    restore_best_weights=True # roll back to the best checkpoint
)

m_es = model_with_dropout(0.4)
m_es.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

print("  Training with early stopping (max 100 epochs, patience=8)...")
hist_es = m_es.fit(
    X_tr_s, y_tr_s,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=BATCH,
    callbacks=[early_stop], verbose=0
)

epochs_run = len(hist_es.history['loss'])
best_epoch  = np.argmin(hist_es.history['val_loss']) + 1
print(f"  Training stopped at epoch {epochs_run} (best epoch = {best_epoch})")
print(f"  Best val loss: {min(hist_es.history['val_loss']):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(hist_es.history['accuracy'],     'b-', linewidth=2, label='Train Acc')
axes[0].plot(hist_es.history['val_accuracy'], 'r--', linewidth=2, label='Val Acc')
axes[0].axvline(best_epoch - 1, color='green', linestyle=':', linewidth=2,
                label=f'Best epoch ({best_epoch})')
axes[0].set_title('Early Stopping: Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(hist_es.history['loss'],     'b-', linewidth=2, label='Train Loss')
axes[1].plot(hist_es.history['val_loss'], 'r--', linewidth=2, label='Val Loss')
axes[1].axvline(best_epoch - 1, color='green', linestyle=':', linewidth=2,
                label=f'Stopped (ep {epochs_run})')
axes[1].set_title('Early Stopping: Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lesson13_early_stopping.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson13_early_stopping.png")

# ──────────────────────────────────────────────────────────────
# SECTION 5: Side-by-Side Comparison of All Regularization Methods
# ──────────────────────────────────────────────────────────────
print("\n--- Section 5: All Regularization Methods Compared ---")

def model_combined():
    """Model with L2 + Dropout + early stopping."""
    l2 = keras.regularizers.l2(0.001)
    return keras.Sequential([
        keras.layers.Conv2D(64, 3, padding='same', activation='relu',
                            input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu', kernel_regularizer=l2),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax'),
    ])

final_configs = {
    'No Regularization':           (model_baseline,     {}),
    'L2 (λ=0.001)':               (model_with_l2,      {'lam': 0.001}),
    'Dropout (p=0.5)':             (model_with_dropout, {'rate': 0.5}),
    'L2 + Dropout + EarlyStopping': (model_combined,    {}),
}

results_summary = {}
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = ['blue', 'orange', 'green', 'red']

for (label, (builder, kwargs)), color in zip(final_configs.items(), colors):
    m = builder(**kwargs)
    m.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    callbacks = []
    if 'EarlyStopping' in label:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=8, restore_best_weights=True
        ))

    h = m.fit(X_tr_s, y_tr_s,
              validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH,
              callbacks=callbacks, verbose=0)

    final_train = h.history['accuracy'][-1]
    final_val   = h.history['val_accuracy'][-1]
    results_summary[label] = {'train': final_train, 'val': final_val,
                               'gap': final_train - final_val}

    axes[0].plot(h.history['val_accuracy'], color=color, linewidth=2, label=label)
    axes[1].plot(h.history['val_loss'],     color=color, linewidth=2, label=label)

    print(f"  {label:<38}: train={final_train:.4f}  val={final_val:.4f}  "
          f"gap={final_train - final_val:.4f}")

axes[0].set_title('Validation Accuracy — All Methods')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(fontsize=7)
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Validation Loss — All Methods')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(fontsize=7)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lesson13_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson13_comparison.png")

# Summary table
print("\nSUMMARY TABLE")
print(f"{'Method':<38} {'Train':>6} {'Val':>6} {'Gap':>6}")
print("-" * 60)
for label, vals in results_summary.items():
    print(f"{label:<38} {vals['train']:>6.4f} {vals['val']:>6.4f} {vals['gap']:>6.4f}")
print("  → Smaller gap = less overfitting")

print("\n" + "=" * 60)
print("Lesson 13 complete. Outputs saved:")
print("  lesson13_overfitting.png")
print("  lesson13_l1_l2.png")
print("  lesson13_dropout.png")
print("  lesson13_early_stopping.png")
print("  lesson13_comparison.png")
print("=" * 60)

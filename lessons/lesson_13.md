# Lesson 13: Overfitting and Regularization Techniques

## Learning Objectives
- Define overfitting and underfitting using the bias-variance tradeoff framework
- Interpret learning curves to diagnose whether a model is overfitting, underfitting, or well-fitted
- Derive the L1 (Lasso) and L2 (Ridge) regularization formulas and explain how they constrain model weights
- Explain the dropout mechanism and justify why it acts as an ensemble-like regularizer
- Implement early stopping and explain how it prevents wasted computation and overfitting
- Apply multiple regularization strategies in combination and evaluate their effect on generalization

---

## Detailed Explanation

### The Problem: Memorizing vs. Generalizing

A student who memorizes the exact answers to last year's exam will fail when the questions change slightly. A student who understands the underlying concepts will perform well on new questions. Neural networks face exactly this challenge.

**Overfitting** occurs when a model learns the training data too well — including its noise, random patterns, and idiosyncratic details — such that its performance on unseen data is significantly worse than on training data. The model has memorized rather than generalized.

**Underfitting** is the opposite: the model is too simple to capture the patterns in the data and performs poorly on both training and test data.

Signs of overfitting:
- Training loss continues decreasing; validation loss starts increasing
- Training accuracy is very high (95%+); validation accuracy is much lower
- The model performs poorly on examples that are slightly different from training data

Signs of underfitting:
- Both training and validation loss remain high
- Model cannot even fit the training data well
- Performance barely exceeds random chance

---

### The Bias-Variance Tradeoff

Every model's prediction error can be decomposed into three parts:

```
Total Error = Bias² + Variance + Irreducible Noise
```

**Bias** measures how far the average model prediction is from the true value. High bias = underfitting. A linear model trying to fit a complex nonlinear pattern will have high bias.

**Variance** measures how much the model's prediction changes when trained on different subsets of data. High variance = overfitting. A deep network with millions of parameters trained on only 500 examples will have high variance — small changes in training data lead to very different models.

The tradeoff: as model complexity increases, bias typically decreases but variance increases. The "sweet spot" is a model complex enough to fit the true pattern but not so complex that it memorizes noise.

```
         Bias²          Variance      Total Error
Complex:  low      +       high     =   high (overfit)
Simple:   high     +       low      =   high (underfit)
Balanced: medium   +       medium   =   low  (optimal)
```

---

### Learning Curves: Your Diagnostic Tool

A learning curve plots training and validation loss (or accuracy) over epochs. Each scenario produces a distinct pattern:

```
OVERFITTING:                    UNDERFITTING:                  WELL-FITTED:
Loss                            Loss                           Loss
|  train_loss                   |  val_loss                    |  val_loss
|     *                         |  train_loss  *  *  *         |     *
|      *                        |       *  *  *                |    * * * * _
|       *____                   |    *  *                      |   *
|  val_loss  ----*               |  *                           |  *
|                 ----*          |                              |
+--------epochs->               +--------epochs->              +--------epochs->
```

In the overfitting plot, the gap between training loss and validation loss is the "overfitting gap." The goal of regularization is to close this gap.

---

### L2 Regularization (Ridge / Weight Decay)

L2 regularization adds a penalty to the loss function proportional to the **sum of squared weights**:

```
L_total = L_original + λ * Σ w²
```

Where λ (lambda) is the regularization strength. The gradient of this term pushes all weights toward zero during training:

```
w_new = w_old - lr * (∂L/∂w + 2λw)
       = w_old * (1 - 2λ*lr) - lr * ∂L/∂w
```

The factor `(1 - 2λ*lr)` multiplies weights by a number slightly less than 1 at each step — hence the name **weight decay**. L2 does not zero out weights; it makes them small. This prevents any single weight from dominating, encouraging the model to spread information across many neurons.

---

### L1 Regularization (Lasso)

L1 regularization adds the **sum of absolute values of weights**:

```
L_total = L_original + λ * Σ |w|
```

The gradient of |w| is +1 for positive weights and -1 for negative weights (the sign function). This means L1 pushes weights by a constant amount toward zero regardless of their magnitude. Crucially, this can push small weights all the way to exactly zero, producing **sparse models** where many weights are exactly 0. This is useful for feature selection and compression.

**L1 vs. L2:**
- L2: Keeps all weights small; smooth gradients; more commonly used in deep learning (weight decay)
- L1: Makes many weights exactly zero; useful for sparsity and feature selection
- L1+L2 combined is called **Elastic Net**

In Keras: `kernel_regularizer=regularizers.l2(0.001)` or `regularizers.l1(0.001)` on any layer.

---

### Dropout

**Dropout** is arguably the most widely used regularization technique for deep neural networks. At each training step, each neuron is independently and randomly "dropped" (set to zero) with probability `p`. The remaining neurons are scaled up by `1/(1-p)` to maintain the expected sum.

```
Without dropout:  [0.8, 0.2, 0.5, 0.9, 0.3]
With dropout p=0.5: [0.0, 0.4, 1.0, 0.0, 0.6]  <- 2 neurons dropped, others scaled
```

**Why does this work?**

1. **Ensemble intuition**: Each training step uses a different "thinned" network. With N neurons and dropout probability 0.5, there are 2^N possible sub-networks. The final model approximates an ensemble of these sub-networks, which generalizes better than any single network.

2. **Co-adaptation prevention**: Without dropout, neurons can develop complex co-dependencies where neuron A only works because neuron B feeds it exactly the right value. Dropout forces each neuron to work independently and redundantly, making representations more robust.

3. **Implicit L2 regularization**: There is a mathematical connection between dropout and L2 regularization; both shrink weight magnitudes.

**During inference (testing)**, dropout is turned OFF, and all neurons are active. Keras handles this automatically via the `training` flag.

**Dropout rates:**
- Fully connected layers: 0.3–0.5 typical
- Convolutional layers: 0.1–0.25 (or use Spatial Dropout)
- Too high: underfitting; too low: insufficient regularization

**Spatial Dropout2D**: For CNNs, drops entire feature maps (channels) rather than individual neurons. This is more appropriate because neighboring pixels in a feature map are highly correlated.

---

### Early Stopping

**Early stopping** monitors validation loss during training and stops when it stops improving. This prevents the model from continuing to memorize training data after the optimal generalization point is passed.

Implementation in Keras:
```python
callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```

The `patience` parameter specifies how many epochs to wait after the last improvement before stopping. `restore_best_weights=True` ensures the model is rolled back to the best checkpoint, not the final (potentially worse) state.

Early stopping is elegant because it is adaptive — it finds the natural "sweet spot" for the given dataset and architecture, without requiring you to pre-specify the number of epochs.

---

### Model Complexity and Overfitting

```
Model Capacity vs. Generalization:

Generalization
Error
   |
   |      *              <- underfitting (too simple)
   |       *
   |        *
   |          *
   |            *  *    <- optimal range
   |               * *
   |                  * *  * <- overfitting (too complex)
   +------------------------------> Model Capacity
   Simple                         Complex
```

Strategies to reduce overfitting beyond regularization:
- Collect more training data (most effective!)
- Use data augmentation (synthetic data)
- Reduce model architecture complexity (fewer layers, smaller filters)
- Use transfer learning (pre-trained weights already generalize)

---

### Common Misconceptions

1. **"Dropout slows down training because fewer neurons are active"** — Actually, dropout can speed training by preventing over-reliance. Per-epoch time increases slightly, but convergence in terms of generalization is often faster.
2. **"L2 regularization is the same as early stopping"** — They both prevent overfitting, but through different mechanisms. L2 constrains weight magnitudes; early stopping constrains training duration. Both can be used together.
3. **"The validation loss must always be higher than training loss"** — In practice, yes, usually. But with aggressive data augmentation, training loss can be higher because augmented training batches are harder.

---

## Key Concepts

| Term | Definition | Why It Matters |
|------|-----------|----------------|
| Overfitting | Model learns training noise; high train accuracy, low test accuracy | Core problem in ML; must be diagnosed and addressed |
| Underfitting | Model too simple to capture patterns; poor on both train and test | Indicates need for more capacity or better features |
| Bias | Systematic error from wrong assumptions in the model | High bias = underfitting |
| Variance | Sensitivity of model to training data fluctuations | High variance = overfitting |
| Bias-Variance Tradeoff | Increasing capacity reduces bias but increases variance | Guides model selection and regularization strategy |
| L2 Regularization | Adds λ*Σw² to loss; shrinks all weights toward zero | Most common regularization in deep learning (weight decay) |
| L1 Regularization | Adds λ*Σ|w| to loss; pushes many weights to exactly zero | Useful for sparsity and feature selection |
| Dropout | Randomly zeroes neurons during training with probability p | Ensemble-like regularization; very effective for deep nets |
| Early Stopping | Stop training when validation loss stops improving | Prevents wasted computation and late-stage overfitting |
| Learning Curve | Plot of training/validation loss over epochs | Primary diagnostic tool for overfitting/underfitting |
| Weight Decay | Another name for L2 regularization; weights decay toward zero | Standard default regularizer in most optimizers |
| Spatial Dropout | Drops entire feature maps in CNNs | More appropriate for convolutional layers than neuron dropout |

---

## Code Reference

See the full runnable demo in **code/lesson_13.py**

---

## Activities

1. **Intentional Overfitting**: Design a CNN with 4+ convolutional layers and no regularization. Train it on only 500 CIFAR-10 examples. Plot training vs. validation accuracy curves. Document when overfitting begins.

2. **Regularization Comparison**: Take the overfitted model from Activity 1. Create three variants: one with L2 regularization (λ=0.001), one with dropout (p=0.4), and one with both. Compare their validation accuracy curves on the same plot.

3. **Early Stopping Callback**: Add `EarlyStopping(patience=5, restore_best_weights=True)` to a training run. Log at which epoch training stops. Compare final accuracy to a model trained for the full epoch budget. How many epochs were saved?

4. **Bias-Variance Visualization**: Train models of increasing complexity (linear → small CNN → large CNN) on CIFAR-10. Plot training accuracy and test accuracy for each. Mark the region of underfitting, optimal fit, and overfitting on the resulting graph.

5. **Lambda Sensitivity**: Train the same architecture with L2 regularization strengths λ ∈ {0, 0.0001, 0.001, 0.01, 0.1}. Plot final validation accuracy vs. λ on a log scale. What happens when λ is too large?

---

## Review Questions

1. Define overfitting in your own words. Why is achieving 100% training accuracy not necessarily a success?
2. Draw and label a learning curve for an overfitted model. What does the "overfitting gap" represent and how wide is acceptable?
3. Explain how L1 and L2 regularization both reduce overfitting but through different mechanisms. Why does L1 produce sparse models while L2 does not?
4. Why is dropout referred to as an "ensemble method"? What ensemble does it approximate at inference time?
5. When would you prefer early stopping over L2 regularization? Are they mutually exclusive? Explain.

---

## Further Reading

- "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" — Srivastava et al., JMLR 2014
- Deep Learning Book, Chapter 7: Regularization for Deep Learning — Goodfellow, Bengio, Courville
- "An Introduction to the Bias-Variance Tradeoff" — Scott Fortmann-Roe (scott.fortmann-roe.com)
- "Regularization and Variable Selection via the Elastic Net" — Zou & Hastie, 2005
- Keras documentation: `keras.regularizers` module and `EarlyStopping` callback

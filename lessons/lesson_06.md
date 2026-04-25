# Lesson 06: Activation Functions

## Learning Objectives
- Explain why non-linearity is essential in neural networks and what happens without it
- Describe the mathematical formulas and behaviors of ReLU, sigmoid, tanh, and softmax
- Identify the dying ReLU problem and explain how Leaky ReLU and ELU address it
- Choose the appropriate activation function for hidden layers versus output layers
- Recognize the vanishing gradient problem and explain how it relates to activation function choice
- Visualize and compare activation function outputs using Python and Matplotlib

---

## Detailed Explanation

### Why Neural Networks Need Non-Linearity

Imagine you are stacking transparent glass sheets on top of each other. No matter how many sheets you stack, light still passes through in the same straight line. Now replace each sheet with a prism — each one bends the light differently, and stacking multiple prisms creates extraordinarily complex light patterns that no single prism could produce.

This analogy captures the core problem with linear-only neural networks. If every layer in a network performs only a linear transformation (multiply by weights, add bias), then the *entire network* is mathematically equivalent to a single linear transformation. You could have one thousand layers, and they would collectively perform nothing more powerful than a single matrix multiplication. This means a deep linear network cannot learn any pattern that isn't linearly separable — it can't recognize curves, can't distinguish complex shapes, can't learn the features that make a cat look different from a dog.

**The mathematical proof is straightforward:**

If Layer 1 computes `y = W1 * x + b1` and Layer 2 computes `z = W2 * y + b2`, then substituting:

```
z = W2 * (W1 * x + b1) + b2
z = (W2*W1) * x + (W2*b1 + b2)
z = W_combined * x + b_combined
```

No matter how many layers you stack, without non-linearity you always collapse to `W_combined * x + b_combined` — a single linear transformation. Activation functions break this collapse by introducing non-linear "bends" into the computation.

---

### ReLU — Rectified Linear Unit

ReLU is the most widely used activation function in modern deep learning. Its formula is elegantly simple:

```
ReLU(x) = max(0, x)
```

In plain English: if the input is positive, pass it through unchanged; if the input is negative or zero, output zero.

```
Output
  |         /
  |        /
  |       /
  |      /
  |     /
  |    /
  |---/-----------> Input
      0
```

**Why ReLU works so well:**
1. **Computationally cheap** — no exponentials, just a comparison
2. **Sparse activation** — roughly half the neurons output zero, creating efficient representations
3. **No vanishing gradient for positive inputs** — the gradient is exactly 1 for any positive input, allowing gradients to flow freely during backpropagation

**The Dying ReLU Problem:** If a neuron's weights get updated such that it always receives negative inputs, it will always output zero. Its gradient is also zero, so no further weight updates occur — the neuron is permanently "dead." This commonly happens with large learning rates. Solutions include:
- Careful weight initialization
- Using Leaky ReLU instead
- Batch normalization before activation

---

### Leaky ReLU and ELU

**Leaky ReLU** solves the dying ReLU problem by allowing a small, non-zero gradient for negative inputs:

```
Leaky ReLU(x) = x        if x > 0
              = alpha * x  if x <= 0   (alpha is typically 0.01)
```

Neurons can still recover from negative activations because the gradient is `alpha` instead of zero.

**ELU (Exponential Linear Unit)** goes further:

```
ELU(x) = x              if x > 0
       = alpha*(e^x - 1) if x <= 0
```

ELU has negative saturation for large negative inputs (unlike Leaky ReLU which is unbounded negative), which can make learned representations more robust. However, it involves an exponential computation, making it slower than ReLU.

---

### Sigmoid

The sigmoid function maps any real number to a value between 0 and 1:

```
sigmoid(x) = 1 / (1 + e^(-x))
```

```
Output
  1 |            ___________
    |          /
0.5 |         |
    |       /
  0 |______/
           0
           Input
```

**Where sigmoid is useful:** Binary classification output layers, where you need a probability between 0 and 1. However, sigmoid is rarely used in hidden layers today because:

1. **Vanishing gradient:** For very large or very small inputs, the sigmoid curve becomes nearly flat. The derivative (gradient) approaches zero, meaning gradients shrink exponentially as they backpropagate through sigmoid layers. Deep networks trained with sigmoid in hidden layers barely update early-layer weights.

2. **Not zero-centered:** Sigmoid outputs are always positive (between 0 and 1), which means gradients during backpropagation always have the same sign. This causes inefficient "zig-zag" weight updates.

---

### Tanh (Hyperbolic Tangent)

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

Tanh is similar to sigmoid but maps inputs to the range (-1, 1). It is zero-centered, which fixes the sign problem of sigmoid but still suffers from vanishing gradients at extreme inputs. Tanh is sometimes used in recurrent neural networks (RNNs) for hidden states.

---

### Softmax — For Multi-Class Output

When your network must choose among multiple classes (e.g., digits 0–9), the output layer uses softmax:

```
softmax(x_i) = e^(x_i) / sum(e^(x_j) for all j)
```

Softmax takes a vector of raw scores (called **logits**) and converts them into a probability distribution that sums to exactly 1. The class with the highest probability is the network's prediction.

**Example:**
```
Logits:    [2.0,  1.0,  0.5]
Softmax:   [0.59, 0.24, 0.17]   ← sums to 1.0
```

---

### Choosing Activation Functions: A Practical Guide

| Layer Type       | Recommended Activation | Reason                                          |
|------------------|------------------------|-------------------------------------------------|
| Hidden (Conv)    | ReLU                   | Fast, avoids vanishing gradient                 |
| Hidden (deep)    | Leaky ReLU or ELU      | Avoids dying ReLU in very deep networks         |
| Output (binary)  | Sigmoid                | Outputs probability between 0 and 1            |
| Output (multi)   | Softmax                | Outputs probability distribution over classes  |
| Output (regression) | Linear (none)       | Outputs any real value                          |

---

### The Vanishing Gradient Problem

When training deep networks with backpropagation, gradients are computed layer by layer using the chain rule. If each layer's activation function has a gradient less than 1 (like sigmoid or tanh in saturation regions), those gradients get multiplied together across many layers. With 10 layers and gradients of 0.2 each: 0.2^10 ≈ 0.0000001. Early layers receive essentially zero gradient and learn nothing.

ReLU largely solves this for modern CNNs by providing a gradient of exactly 1 for positive inputs, allowing gradients to flow backward without shrinking.

---

## Key Concepts

| Term | Definition | Why It Matters |
|------|-----------|----------------|
| Activation Function | A non-linear mathematical function applied element-wise after a layer's linear transformation | Without it, deep networks are equivalent to a single linear layer |
| ReLU | max(0, x) — outputs input if positive, zero otherwise | Most widely used hidden-layer activation; fast and gradient-preserving |
| Sigmoid | 1/(1+e^-x) — squashes input to (0,1) | Used in binary classification output layers |
| Tanh | Maps input to (-1, 1) using hyperbolic tangent | Zero-centered; better than sigmoid but still vanishes |
| Softmax | Converts logit vector to probability distribution summing to 1 | Standard output for multi-class classification |
| Dying ReLU | Phenomenon where neurons permanently output zero due to always-negative inputs | Reduces network capacity; solved by Leaky ReLU/ELU |
| Leaky ReLU | Like ReLU but with small non-zero slope for negative inputs (alpha*x) | Prevents dying ReLU while keeping computational simplicity |
| ELU | Exponential linear unit; smooth negative side using alpha*(e^x - 1) | Smooth gradient; can produce more robust representations |
| Vanishing Gradient | Phenomenon where gradients shrink exponentially in deep networks using saturating activations | Makes early layers fail to learn; motivates ReLU use |
| Non-linearity | Any mathematical transformation that is not a straight line | The key ingredient that gives neural networks universal approximation power |

---

## Code Reference

See [`code/lesson_06.py`](../code/lesson_06.py) for fully runnable demonstrations of all activation functions, visualizations, and comparisons.

---

## Activities

1. **Plot All Activations:** Using Matplotlib, plot ReLU, Leaky ReLU, ELU, sigmoid, and tanh on the same figure for inputs from -5 to 5. Label each curve and add horizontal/vertical reference lines at y=0 and x=0. Which curves are bounded? Which are unbounded?

2. **Gradient Comparison:** Manually compute the derivative (gradient) of ReLU, sigmoid, and tanh at x = 0, x = 2, and x = -2. Make a table of your results. At which input values does each function have the largest gradient? Smallest?

3. **Dying ReLU Simulation:** Create a simple 3-layer network using only ReLU activations. Initialize the weights to large negative values. Forward-pass a batch of all-positive inputs and count how many neurons output exactly zero in the middle layer. Then switch to Leaky ReLU and repeat. Discuss the difference.

4. **Softmax by Hand:** Given the logit vector `[3.0, 1.0, 0.2, -1.5]`, manually compute the softmax probabilities step by step using the formula. Verify that your probabilities sum to 1. Which class would the network predict?

5. **Activation Function Ablation:** Train two simple networks on MNIST — one using sigmoid activations in hidden layers and one using ReLU. Train both for 10 epochs and compare final accuracy and the shape of the training loss curve. Write a paragraph explaining the differences you observe.

---

## Review Questions

1. Why does stacking multiple linear layers without activation functions not improve model capacity? Provide the mathematical argument.

2. Explain the vanishing gradient problem. Which activation functions suffer from it most, and why does ReLU largely avoid it for positive inputs?

3. What is the dying ReLU problem? Describe two practical ways to prevent or recover from it.

4. When would you choose sigmoid over softmax for the output layer? Give a concrete example of each use case.

5. Why is tanh considered better than sigmoid for hidden layers in older architectures, even though both suffer from vanishing gradients?

---

## Further Reading

- **"Deep Learning" by Goodfellow, Bengio, and Courville** — Chapter 6 covers activation functions with rigorous mathematical treatment
- **"Empirical Evaluation of Rectified Activations"** (Xu et al., 2015) — Systematic comparison of ReLU variants
- **Keras Activation Layers documentation** — https://keras.io/api/layers/activation_layers/ — practical implementation guide
- **"Dying ReLU and Initialization"** — Blog post by Andrej Karpathy on weight initialization's role in preventing dead neurons
- **"Understanding the difficulty of training deep feedforward neural networks"** (Glorot & Bengio, 2010) — The paper that introduced Xavier initialization to combat vanishing gradients

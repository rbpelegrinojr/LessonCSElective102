# Lesson 6: Activation Functions

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain why activation functions are essential in neural networks
- Describe the properties, formulas, and use cases of Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, and Softmax
- Identify the vanishing gradient problem and explain which activations suffer from it
- Choose the correct activation function for any layer in a CNN
- Implement activation functions in Keras and visualize their behavior

---

## Detailed Explanation

### Why Activation Functions Exist

A neural network without activation functions is just a stack of matrix multiplications. No matter how many layers you add, the entire network collapses into a single linear transformation — it can only draw straight lines through data, no matter how deep it is. Activation functions introduce **non-linearity** into the network, allowing it to learn complex, curved decision boundaries that can separate real-world data.

Think of it this way: a linear model can tell whether a point is above or below a line. But to classify images of cats versus dogs, you need a model that can learn incredibly intricate boundaries in a very high-dimensional space. Only non-linear transformations can build this capability layer by layer.

Every neuron in a neural network performs the computation:

```
output = activation(weights · inputs + bias)
```

The activation function is applied element-wise to the weighted sum before passing the result to the next layer. Without it, composing layers would look like:

```
Layer 1: W1 * x + b1
Layer 2: W2 * (W1 * x + b1) + b2 = (W2*W1)*x + (W2*b1 + b2)
```

This collapses to a single linear transformation `Ax + c`, regardless of depth. Activation functions break this collapse.

---

### Sigmoid Function

**Formula:** σ(x) = 1 / (1 + e^−x)

**Range:** (0, 1)

The sigmoid function maps any real number to a value between 0 and 1, which makes it feel like a probability. It was historically the default activation function for neural networks.

```
1.0 |          .....-----
    |       ...
0.5 |      .
    |   ...
0.0 |----.....
    +----------------------
       -5    0    5
```

**Strengths:**
- Smooth and differentiable everywhere
- Output is bounded between 0 and 1

**Weaknesses — The Vanishing Gradient Problem:**
The sigmoid's gradient (derivative) is: σ'(x) = σ(x) · (1 − σ(x))

The maximum value of this derivative is 0.25, occurring at x = 0. For large positive or negative values of x, the gradient approaches 0. When you backpropagate through many layers, you multiply these small gradients together. With 10 layers, you could be multiplying 0.25 ten times: 0.25^10 ≈ 0.000001. The gradient essentially vanishes before reaching the early layers, which means those layers stop learning. This is the **vanishing gradient problem**.

**When to use sigmoid:** Only at the output layer for **binary classification**, where you want a probability output.

---

### Tanh Function

**Formula:** tanh(x) = (e^x − e^−x) / (e^x + e^−x)

**Range:** (−1, 1)

Tanh is a rescaled sigmoid. It is **zero-centered**, meaning its outputs are symmetric around 0. This makes optimization easier because gradients can be both positive and negative, preventing the zig-zagging in gradient descent that non-zero-centered outputs cause.

```
 1.0 |        .....------
     |     ...
 0.0 |-----.
     |  ...
-1.0 |---.....
     +----------------------
        -5    0    5
```

**Strengths:**
- Zero-centered outputs
- Stronger gradients near zero than sigmoid

**Weaknesses:**
- Still suffers from vanishing gradients for large |x| values
- Computationally more expensive than ReLU

**When to use tanh:** Recurrent neural networks (LSTMs, GRUs) still commonly use tanh. Rarely used in CNNs today.

---

### ReLU — Rectified Linear Unit

**Formula:** ReLU(x) = max(0, x)

**Range:** [0, ∞)

ReLU is the most widely used activation function in modern deep learning. It is devastatingly simple: if the input is negative, output 0; otherwise output the input unchanged.

```
4 |           /
3 |          /
2 |         /
1 |        /
0 |-------/
  +----------------------
     -3   0   3
```

**Strengths:**
- Computationally trivial (just a max operation)
- Does not saturate for positive values — no vanishing gradient on the positive side
- **Sparse activation:** roughly 50% of neurons output 0, creating sparse representations which are computationally efficient and act as implicit regularization
- Trains much faster than sigmoid/tanh in practice

**Weaknesses — The Dying ReLU Problem:**
If a neuron receives a large negative input, it outputs 0, and its gradient is also 0. If this happens consistently during training, the neuron's weights will never update — it "dies." Networks can lose a significant fraction of neurons this way, especially with high learning rates.

**When to use ReLU:** Default choice for all **hidden layers** in CNNs and dense networks.

---

### Leaky ReLU

**Formula:** LeakyReLU(x) = x if x > 0, else α·x (typically α = 0.01)

Leaky ReLU is a direct fix for the dying ReLU problem. Instead of outputting 0 for negative inputs, it outputs a small negative value with slope α. This means the gradient is never exactly zero, so neurons cannot permanently die.

```
4 |           /
2 |          /
0 |--------/
-0.1|      /   (slope = 0.01)
    +----------------------
       -3   0   3
```

**When to use Leaky ReLU:** When you observe dying ReLU neurons in your network, or as a conservative default over standard ReLU in deep networks.

---

### ELU — Exponential Linear Unit

**Formula:**
- ELU(x) = x, if x > 0
- ELU(x) = α · (e^x − 1), if x ≤ 0 (typically α = 1.0)

ELU provides smooth, non-zero outputs for negative values (unlike the sharp corner of Leaky ReLU). Its negative region approaches −α asymptotically, and the smooth curve helps reduce bias shifts during training.

**Strengths:**
- Smooth everywhere (differentiable at x = 0)
- Negative outputs help push mean activations closer to zero
- Robust to noise

**Weaknesses:**
- More computationally expensive due to the exponential

**When to use ELU:** When you want a smoother alternative to Leaky ReLU, especially in very deep networks.

---

### Softmax Function

**Formula:** Softmax(x_i) = e^(x_i) / Σ e^(x_j) for all j

Softmax is special — it operates on a **vector** of values (logits), not a single number. It converts raw output scores into a probability distribution: all outputs are positive and they sum to exactly 1.

For example, if the raw logits for a 3-class problem are [2.0, 1.0, 0.1], softmax converts them to approximately [0.66, 0.24, 0.10]. The class with the highest logit gets the highest probability.

**When to use Softmax:** Exclusively at the **output layer for multi-class classification** (more than 2 classes). Combine with categorical cross-entropy loss.

---

### Vanishing Gradient: Mathematical Intuition

During backpropagation, gradients are computed by the chain rule — multiplying derivatives layer by layer going backward. If each layer's activation function has a maximum derivative of d_max < 1, then after L layers the gradient magnitude is at most d_max^L.

- Sigmoid: d_max = 0.25 → after 10 layers: 0.25^10 ≈ 9.5 × 10^−7
- ReLU: d_max = 1 (for positive region) → gradients do not shrink from activation alone

This is why ReLU and its variants dominate modern deep networks — they enable training of networks with dozens or hundreds of layers.

---

### Comparison Table

| Function     | Range      | Zero-Centered | Vanishing Gradient | Dying Neurons | Typical Use          |
|--------------|------------|---------------|--------------------|---------------|----------------------|
| Sigmoid      | (0, 1)     | No            | Yes (severe)       | No            | Binary output        |
| Tanh         | (−1, 1)    | Yes           | Yes (moderate)     | No            | RNNs                 |
| ReLU         | [0, ∞)     | No            | No (positive side) | Yes           | Hidden layers (CNN)  |
| Leaky ReLU   | (−∞, ∞)    | No            | No                 | No            | Hidden layers        |
| ELU          | (−α, ∞)    | Approximately | No                 | No            | Deep hidden layers   |
| Softmax      | (0, 1)     | N/A           | N/A                | N/A           | Multi-class output   |

---

## Key Concepts Table

| Term                   | Definition                                                                 |
|------------------------|----------------------------------------------------------------------------|
| Non-linearity          | A transformation that cannot be expressed as Ax + b                       |
| Vanishing gradient     | Gradients becoming near-zero in early layers, halting learning             |
| Dying ReLU             | Neurons stuck at 0 output and 0 gradient permanently                       |
| Sparse activation      | Most neurons outputting 0, creating efficient representations              |
| Logits                 | Raw, unnormalized outputs from the final Dense layer before softmax        |
| Saturation             | When a function's gradient approaches 0 at extreme input values            |

---

## Code Reference

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation

# ReLU in a Dense layer
layer = Dense(128, activation='relu')

# Sigmoid for binary output
output = Dense(1, activation='sigmoid')

# Softmax for multi-class output
output = Dense(10, activation='softmax')

# Leaky ReLU
from tensorflow.keras.layers import LeakyReLU
layer = Dense(128)
activation = LeakyReLU(alpha=0.01)

# ELU
layer = Dense(128, activation='elu')

# Manual sigmoid
import numpy as np
sigmoid = lambda x: 1 / (1 + np.exp(-x))

# Manual ReLU
relu = lambda x: np.maximum(0, x)

# Manual softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return exp_x / exp_x.sum()
```

---

## Activities

### Activity 6.1 — Plot and Compare
Run `code/lesson_06.py` Section 1. Observe how each activation function shapes its output. Which function has the steepest gradient near zero?

### Activity 6.2 — Vanishing Gradient Experiment
In Section 2 of the code, observe the gradient magnitude through layers using sigmoid versus ReLU. After how many layers does the sigmoid gradient become effectively zero?

### Activity 6.3 — Training Speed Comparison
Run Section 3. Build two identical networks, one with sigmoid activations and one with ReLU activations. Train both on MNIST for 5 epochs. Record the accuracy at the end of each epoch. Which converges faster?

### Activity 6.4 — Softmax Exploration
Manually compute softmax for these logits: [3.0, 1.0, 0.2]. Verify that the outputs sum to 1. What happens if you multiply all logits by 2?

### Activity 6.5 — Dead ReLU Hunt
In Section 5, create a network with a very high learning rate. Count what fraction of neurons output 0 after a single forward pass. Repeat with Leaky ReLU and compare.

---

## Review Questions

1. Why does a network without activation functions collapse into a linear model, regardless of depth?
2. What is the maximum gradient of the sigmoid function, and why does this cause problems in deep networks?
3. How does ReLU solve the vanishing gradient problem? What problem does it introduce?
4. What is the key difference between Leaky ReLU and standard ReLU?
5. Why is Tanh considered better than sigmoid for hidden layers?
6. For a 10-class image classification problem, which activation function should the output layer use, and why?
7. What does it mean for a neuron to "die" in a ReLU network?
8. Why is numerical stability important when computing softmax, and how do we achieve it?

---

## Further Reading

- [Understanding Activation Functions in Neural Networks — Avinash Sharma](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)
- [CS231n: Neural Networks Part 1 — Stanford](https://cs231n.github.io/neural-networks-1/)
- [Deep Learning Book, Chapter 6 — Goodfellow et al.](https://www.deeplearningbook.org/contents/mlp.html)
- [Dying ReLU and Initialization — Kaiming He et al.](https://arxiv.org/abs/1502.01852)
- [ELU Paper — Clevert et al., 2015](https://arxiv.org/abs/1511.07289)

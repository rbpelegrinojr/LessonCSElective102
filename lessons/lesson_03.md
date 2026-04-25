# Lesson 3: Introduction to Neural Networks

## Learning Objectives

By the end of this lesson, you will be able to:

- **Describe the biological inspiration** behind artificial neural networks and the perceptron model
- **Define the components** of a single artificial neuron: inputs, weights, bias, and activation function
- **Explain the structure** of a multi-layer perceptron (MLP): input layer, hidden layers, output layer
- **Trace a forward pass** step-by-step through a neural network, computing intermediate values
- **Describe gradient descent** intuitively and explain its role in training neural networks
- **Explain what a loss function** measures and name examples used for classification
- **Describe backpropagation** conceptually using chain rule intuition
- **Explain why depth** (many layers) matters for learning complex functions

---

## Detailed Explanation

### Biological Inspiration

The artificial neural network is loosely inspired by the human brain. The brain contains approximately 86 billion **neurons** — specialized cells that receive, process, and transmit electrical signals. Each neuron collects signals from other neurons through **dendrites**, processes them in the **cell body (soma)**, and transmits an output signal through its **axon** to the dendrites of other neurons.

A neuron fires (sends an output) only when the total incoming signal exceeds a certain **threshold**. This is an all-or-nothing mechanism: below threshold, silence; above threshold, a signal propagates.

Artificial neurons model this behavior mathematically:
- **Dendrites → inputs** (x₁, x₂, ..., xₙ): signals arriving from previous neurons or raw data
- **Synaptic strengths → weights** (w₁, w₂, ..., wₙ): how much influence each input has
- **Cell body → weighted sum + bias**: z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
- **Threshold/firing → activation function**: output = f(z)

The key insight: by adjusting the weights, we adjust what the neuron "cares about." Learning is the process of finding the right weights.

### The Perceptron

The **perceptron**, proposed by Frank Rosenblatt in 1958, is the simplest artificial neural network — a single neuron. Given an input vector **x** = [x₁, x₂, ..., xₙ], a weight vector **w** = [w₁, w₂, ..., wₙ], and a bias b, the perceptron computes:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b  =  wᵀx + b

output = activation(z)
```

Common activation functions include:
- **Step function**: output 1 if z > 0, else 0 (original perceptron; non-differentiable)
- **Sigmoid**: σ(z) = 1 / (1 + e⁻ᶻ) — squashes to (0, 1)
- **ReLU** (Rectified Linear Unit): max(0, z) — simple, effective, dominates modern networks
- **Tanh**: (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ) — squashes to (-1, 1)

Activation functions are critical because they introduce **non-linearity**. Without them, a neural network of any depth would compute only linear transformations — equivalent to a single matrix multiplication. Non-linearity allows networks to learn complex, curved decision boundaries.

### Multi-Layer Perceptron (MLP)

A single perceptron can only learn linearly separable patterns (problems where classes can be separated by a straight line or hyperplane). Real problems — including image classification — are not linearly separable.

The solution is to **stack neurons in layers** to form a **Multi-Layer Perceptron (MLP)**:

```
┌────────────────────────────────────────────────────────────────┐
│                  NEURAL NETWORK STRUCTURE                      │
│                                                                │
│  INPUT LAYER    HIDDEN LAYER 1   HIDDEN LAYER 2   OUTPUT LAYER│
│                                                                │
│     x₁ ───────▶ [h₁₁] ────────▶ [h₂₁] ──────▶ [o₁] (class 0)│
│        ╲  ╱     [h₁₂] ────────▶ [h₂₂] ──────▶ [o₂] (class 1)│
│     x₂ ──╳────▶ [h₁₃] ────────▶ [h₂₃] ──────▶ [o₃] (class 2)│
│        ╱  ╲     [h₁₄]           [h₂₄]                         │
│     x₃ ───────▶                                               │
│                                                                │
│   (3 inputs)  (4 hidden)       (4 hidden)     (3 outputs)     │
│                                                                │
│  Every neuron in one layer connects to every neuron           │
│  in the next layer → "fully connected" or "dense" layer       │
└────────────────────────────────────────────────────────────────┘
```

- **Input layer**: receives raw features (e.g., flattened pixel values)
- **Hidden layers**: intermediate representations; the network learns what features to extract
- **Output layer**: one neuron per class; outputs class probabilities after softmax

### Forward Pass: Step by Step

Consider a tiny network with 2 inputs, one hidden layer with 3 neurons, and 2 output neurons.

**Step 1 — Input:**
```
x = [0.5, 0.8]
```

**Step 2 — Hidden layer (using ReLU):**
```
z_hidden = W_hidden @ x + b_hidden
a_hidden = ReLU(z_hidden)  →  max(0, z_hidden)
```

**Step 3 — Output layer (using softmax):**
```
z_output = W_output @ a_hidden + b_output
a_output = softmax(z_output)  →  probabilities summing to 1
```

**Step 4 — Prediction:**
```
predicted_class = argmax(a_output)
```

Each layer transforms its inputs into a new representation. Early layers learn low-level features (edges, corners); later layers learn high-level abstractions (object parts, semantic concepts).

### How Networks Learn: Gradient Descent

After a forward pass, the network produces predictions. A **loss function** measures how wrong those predictions are. For multi-class classification, the standard loss function is **categorical cross-entropy**:

```
L = -Σᵢ yᵢ × log(ŷᵢ)
```

where **y** is the one-hot true label vector and **ŷ** is the predicted probability vector. When the model is confident and correct, the loss is near 0. When it is wrong or uncertain, the loss is high.

**Gradient descent** minimizes the loss. The gradient of the loss with respect to each weight tells us which direction to adjust the weight to reduce the loss. We take a small step in the negative gradient direction:

```
w ← w - η × ∂L/∂w
```

where η (eta) is the **learning rate** — a hyperparameter controlling step size. Too large: oscillates and diverges. Too small: learns extremely slowly.

In practice, we use **stochastic gradient descent (SGD)** or more sophisticated optimizers like **Adam** that adapt the learning rate automatically.

### Backpropagation

The gradients needed for gradient descent are computed using **backpropagation** — an efficient application of the **chain rule** from calculus.

The chain rule states that for a composed function f(g(x)):
```
df/dx = (df/dg) × (dg/dx)
```

In a neural network, the loss depends on the output, which depends on the hidden layer activations, which depend on the weights. Backpropagation applies the chain rule layer by layer, propagating error gradients from the output back to the input:

```
∂L/∂w¹ = (∂L/∂a³) × (∂a³/∂z³) × (∂z³/∂a²) × (∂a²/∂z²) × (∂z²/∂a¹) × (∂a¹/∂z¹) × (∂z¹/∂w¹)
```

This is why activation functions must be differentiable (or have useful sub-gradients like ReLU). Without gradients, backpropagation cannot work.

Modern deep learning frameworks (TensorFlow, PyTorch) implement **automatic differentiation** — they automatically compute these gradients, freeing you from manual derivative calculations.

### Loss Functions for Classification

| Loss Function | Use Case | Notes |
|---------------|----------|-------|
| **Binary Cross-Entropy** | 2-class (binary) classification | Output layer: 1 sigmoid neuron |
| **Categorical Cross-Entropy** | Multi-class (one label per image) | Output layer: softmax, one-hot labels |
| **Sparse Categorical Cross-Entropy** | Multi-class, integer labels | Same as above but no one-hot encoding needed |
| **Mean Squared Error (MSE)** | Regression (not classification) | Less appropriate for classification tasks |

### Why Depth Matters

A shallow network (one hidden layer) can theoretically approximate any function — this is the **universal approximation theorem**. However, in practice, a shallow network requires an exponentially large number of neurons to do so.

Deep networks learn **hierarchical representations**:
- Layer 1 detects edges and corners
- Layer 2 combines edges into curves and shapes
- Layer 3 combines shapes into object parts (wheels, eyes, wings)
- Layer 4 combines parts into whole objects

Each layer builds on the previous, composing simple features into complex ones. This compositional learning is the key advantage of depth. A 10-layer network can represent patterns that a 2-layer network would need millions of neurons to match.

This is directly analogous to how human visual cortex processes images: V1 → V2 → V4 → IT cortex, with increasing abstraction at each stage.

---

## Key Concepts Table

| Term | Definition | Why It Matters |
|------|------------|----------------|
| **Neuron** | The basic computational unit of a neural network; computes a weighted sum plus bias, then applies an activation | Understanding neurons enables understanding of the full network |
| **Weight** | A learnable parameter that scales an input; determines how much a neuron "cares" about each input | Weights encode all learned knowledge; training = adjusting weights |
| **Bias** | A learnable offset added to the weighted sum; shifts the activation function | Without bias, the neuron is forced to pass through the origin — limited expressiveness |
| **Activation Function** | A non-linear function applied to the neuron's output (ReLU, sigmoid, tanh) | Introduces non-linearity; without it, deep networks collapse to linear functions |
| **Forward Pass** | Computing the network's output for a given input | Required for both training and inference |
| **Loss Function** | A metric measuring how wrong the model's predictions are | Gradient descent minimizes this; it's the target that guides learning |
| **Backpropagation** | Algorithm to compute gradients of the loss w.r.t. all weights using the chain rule | Makes training deep networks computationally feasible |
| **Gradient Descent** | Optimization algorithm that updates weights in the direction that reduces the loss | The core learning algorithm for neural networks |
| **Learning Rate** | Step size for gradient descent updates (η) | Too high = divergence; too low = slow convergence |
| **Epoch** | One complete pass through the entire training dataset | Training typically runs for many epochs until convergence |

---

## Code Reference

See `code/lesson_03.py` for hands-on examples demonstrating:
- A single neuron implemented from scratch in NumPy
- A 2-layer MLP built entirely from scratch with forward and backward pass
- The same network built with Keras Sequential API
- A detailed forward pass walkthrough with print statements
- Training on the XOR problem with a loss curve visualization

---

## Activities

1. **Neuron by Hand:** Given inputs x = [1.0, 2.0, -1.0], weights w = [0.5, -0.3, 0.8], and bias b = 0.1, manually compute the pre-activation z and then apply the ReLU activation. Show all steps.

2. **Activation Function Graphs:** Plot (on paper or in code) the sigmoid, tanh, and ReLU functions over the range x = [-5, 5]. Identify the output range, derivatives at 0, and saturation regions for each.

3. **XOR by Hand:** XOR outputs 1 when inputs differ and 0 when they match. Why can a single perceptron NOT learn XOR? Draw the truth table and attempt to draw a separating hyperplane. Then explain why a 2-layer network can solve it.

4. **Loss Intuition:** For a 3-class problem, compute the cross-entropy loss for the following cases: (a) prediction=[0.9, 0.05, 0.05], true class=0; (b) prediction=[0.33, 0.34, 0.33], true class=1; (c) prediction=[0.1, 0.1, 0.8], true class=0. Which has the highest loss and why?

5. **Architecture Design:** Design (on paper) a neural network for classifying MNIST digits (28×28 grayscale). Specify: how many input neurons, how many hidden layers and their sizes, how many output neurons, and what activation functions you would use where and why.

---

## Review Questions

1. What role does the activation function play in a neural network? What would happen if you used no activation functions between layers?

2. Explain gradient descent in plain English. What does the gradient tell us, and how do we use it to improve the model?

3. What is backpropagation? Why is it more efficient than computing gradients by finite differences (perturbing each weight slightly and measuring the change in loss)?

4. What is the universal approximation theorem? If a single hidden layer can approximate any function, why do we use very deep networks (50+ layers)?

5. Compare categorical cross-entropy and mean squared error as loss functions for a 10-class image classification problem. Which is more appropriate and why?

---

## Further Reading

- **Neural Networks and Deep Learning** by Michael Nielsen — Free online book at neuralnetworksanddeeplearning.com. Exceptionally clear explanations of backpropagation.
- **Deep Learning** by Goodfellow, Bengio & Courville — Chapters 6 (Deep Feedforward Networks) and 8 (Optimization). Free at deeplearningbook.org.
- **"Understanding the difficulty of training deep feedforward neural networks"** — Glorot & Bengio (2010). Explains why deep networks are hard to train and introduces Xavier initialization.
- **3Blue1Brown "Neural Networks" YouTube series** — Superb visual intuition for how neural networks and backpropagation work. Highly recommended.
- **Keras documentation** — keras.io. Official reference for building networks with the Keras API.

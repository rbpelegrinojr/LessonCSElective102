# Lesson 03: Introduction to Neural Networks

## Learning Objectives
- Describe the analogy between biological neurons and artificial neurons, and explain where the analogy holds and where it breaks down
- Define the perceptron and explain how weights, biases, and activation functions combine to produce an output
- Explain the purpose of hidden layers and describe how stacking layers creates increasingly abstract representations
- Trace the forward pass through a simple multi-layer network, including matrix multiplication
- Describe the training loop: forward pass, loss computation, backpropagation, and gradient descent weight update
- Build and train a simple dense neural network on MNIST using Keras and interpret the training curves

---

## Detailed Explanation

### The Biological Inspiration

The human brain contains approximately 86 billion neurons, each connected to thousands of others. A neuron receives electrical signals from upstream neurons through branch-like structures called dendrites. These signals accumulate in the cell body. If the total incoming signal exceeds a threshold, the neuron "fires" — it sends an electrical pulse down a long fibre called the axon to the dendrites of downstream neurons. The strength of the connection between two neurons (the synapse) changes with experience, which is how learning occurs.

Artificial neural networks borrow this vocabulary but are mathematical, not biological. An artificial neuron is a mathematical function that:
1. Takes several numerical inputs
2. Multiplies each input by a corresponding weight
3. Sums the weighted inputs and adds a bias term
4. Passes the sum through an activation function to produce an output

```
Biological neuron:               Artificial neuron:
  Dendrites (inputs)               x₁, x₂, x₃ (inputs)
  Synaptic strength               w₁, w₂, w₃ (weights)
  Cell body (integration)          z = w₁x₁ + w₂x₂ + w₃x₃ + b
  Threshold / firing               activation function f(z)
  Axon (output)                    output = f(z)
```

The analogy is instructive but imperfect. Biological neurons process signals in massively parallel, asynchronous, and chemically complex ways that artificial networks do not attempt to replicate. Artificial networks are best understood as powerful mathematical function approximators, not brain simulations.

### The Perceptron: The Simplest Neural Network

The perceptron, proposed by Frank Rosenblatt in 1958, is the ancestor of all modern neural networks. A single perceptron takes N numerical inputs, computes a weighted sum, adds a bias, and produces a binary output (0 or 1):

```
  Inputs:    x₁ = 0.8,  x₂ = 0.3,  x₃ = 0.5
  Weights:   w₁ = 0.4,  w₂ = 0.7,  w₃ = -0.2
  Bias:      b  = 0.1

  Weighted sum:
    z = (0.8 × 0.4) + (0.3 × 0.7) + (0.5 × -0.2) + 0.1
    z = 0.32 + 0.21 - 0.10 + 0.10
    z = 0.53

  Step activation (threshold at 0.5):
    output = 1  (since 0.53 > 0.5)
```

The weights determine how much each input contributes. A large positive weight means "this input strongly supports a positive prediction." A large negative weight means "this input strongly argues against a positive prediction." The bias shifts the threshold — it is not tied to any input and allows the neuron to fire even when all inputs are zero.

### Activation Functions

The step function (output is 0 or 1) makes the perceptron non-differentiable, which breaks the learning algorithm. Modern networks replace it with smooth, differentiable activation functions:

**Sigmoid:** Maps any value to the range (0, 1). Historically used in hidden layers; now mainly used in output layers for binary classification.
```
  σ(z) = 1 / (1 + e^(-z))
  Range: (0, 1)
  Problem: "vanishing gradients" in deep networks
```

**Tanh:** Maps to (-1, 1). Similar to sigmoid but zero-centred, which helps during optimisation.
```
  tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
  Range: (-1, 1)
```

**ReLU (Rectified Linear Unit):** The most widely used activation in hidden layers. Extremely simple: output the input if it is positive, otherwise output zero.
```
  ReLU(z) = max(0, z)

  Example:
    ReLU(-2.3) = 0
    ReLU( 0.0) = 0
    ReLU( 3.7) = 3.7
```

ReLU solves the vanishing gradient problem for shallow-to-medium depth networks and is computationally very cheap. Its main failure mode is "dying ReLU" — if weights are initialised badly, a neuron can become stuck always outputting zero.

**Softmax:** Used in the *output* layer for multi-class classification. Converts a vector of raw scores (logits) into a probability distribution that sums to 1.
```
  softmax(zᵢ) = e^zᵢ / Σⱼ e^zⱼ

  Example with 3 classes:
    Raw scores (logits): [2.0, 1.0, 0.1]
    Softmax output:      [0.659, 0.242, 0.099]
    Interpretation:      65.9% probability class 0,
                         24.2% probability class 1,
                          9.9% probability class 2
```

### Layers: Input, Hidden, Output

A multi-layer neural network (also called a Multi-Layer Perceptron or MLP) organises neurons into layers:

```
  INPUT LAYER      HIDDEN LAYER 1   HIDDEN LAYER 2   OUTPUT LAYER
  (784 neurons     (128 neurons     (64 neurons      (10 neurons
   for MNIST)       + ReLU)          + ReLU)          + Softmax)

  ○ ○ ○ ○ ○  →→→  ○ ○ ○ ○  →→→  ○ ○ ○  →→→  ○ ○ ○ ○
  (flat pixels)                                (class probs)
```

**Input layer:** Not a computation layer — it simply holds the input features. For MNIST (28×28 images), the 784-pixel array is flattened into a 784-element vector.

**Hidden layers:** Perform the actual computation. Each neuron in a hidden layer is connected to every neuron in the previous layer ("fully connected" or "dense"). Hidden layers learn increasingly abstract representations: early layers might detect simple pixel patterns; later layers might detect higher-level structures.

**Output layer:** Produces the final prediction. For 10-class classification (MNIST digits 0–9), the output layer has 10 neurons with softmax activation, producing 10 probabilities.

### Weights and Biases as Parameters

Every connection between neurons has an associated weight. Every neuron (except input) has a bias. These are the **learnable parameters** — numbers that the training process adjusts to make the network produce better predictions.

For a dense layer with 784 inputs and 128 neurons:
```
  Weights matrix:  784 × 128 = 100,352 parameters
  Bias vector:              128 parameters
  Total:                100,480 parameters (just for this one layer!)
```

### The Forward Pass

The forward pass is the computation of the network's prediction given a fixed set of inputs and weights. It is entirely matrix multiplication plus activation functions:

```
  Given:
    X = input vector (shape 1×784 for one MNIST image)
    W = weight matrix (shape 784×128)
    b = bias vector (shape 1×128)

  Hidden layer 1 output:
    Z₁ = X · W₁ + b₁       (matrix multiply + broadcast add)
    A₁ = ReLU(Z₁)           (element-wise activation)

  Output layer:
    Z₂ = A₁ · W₂ + b₂
    ŷ = Softmax(Z₂)          (class probability distribution)
```

The dot product X · W₁ computes the weighted sum for all 128 neurons simultaneously, which is why GPUs (which excel at matrix multiplication) provide such enormous speed advantages.

### The Loss Function

After the forward pass produces predictions ŷ, we need to measure how wrong those predictions are. The **loss function** quantifies this error as a single scalar number. For multi-class classification, the standard choice is **categorical cross-entropy**:

```
  Loss = -Σᵢ yᵢ · log(ŷᵢ)

  Where:
    yᵢ = true probability (1 for the correct class, 0 for others — "one-hot")
    ŷᵢ = predicted probability for class i

  Perfect prediction:    Loss → 0
  Terrible prediction:   Loss → ∞
```

The intuition: cross-entropy penalises confident wrong predictions very harshly (log of a near-zero probability is a very large negative number). It rewards the network for being confidently correct.

### Gradient Descent and Backpropagation

The training algorithm asks: "How should each weight be changed to reduce the loss?" Gradient descent answers this by computing the gradient of the loss with respect to each weight — the direction of steepest increase — and then moving each weight a small step in the *opposite* direction (downhill).

```
  New weight = Old weight - Learning Rate × Gradient

  Example:
    weight     = 0.5
    gradient   = 0.3   (loss increases as weight increases)
    learn rate = 0.01
    new weight = 0.5 - 0.01 × 0.3 = 0.497
```

The learning rate controls step size. Too large and the optimiser oscillates or diverges; too small and training is painfully slow.

**Backpropagation** is the algorithm that computes gradients for every weight in the network efficiently, using the chain rule of calculus to propagate error signals from the output layer back through every hidden layer. TensorFlow and Keras implement this automatically via automatic differentiation — you never need to compute gradients by hand.

### The Training Loop

One complete iteration of the training algorithm:

```
  FOR each mini-batch of training examples:
    1. Forward pass:      compute predictions ŷ
    2. Loss computation:  L = cross_entropy(y, ŷ)
    3. Backward pass:     compute ∂L/∂W for all weights W
    4. Weight update:     W = W - α × ∂L/∂W
  END FOR

  One complete pass through the training set = one EPOCH
```

Modern training uses **mini-batch gradient descent**: instead of updating weights after every single example (too noisy) or after seeing the entire dataset (too slow and memory-intensive), we update after each small batch (typically 32–256 examples). This balances noise and efficiency.

---

## Key Concepts

| Term | Definition | Why It Matters |
|---|---|---|
| **Artificial Neuron** | A mathematical function that computes a weighted sum of inputs, adds a bias, and applies an activation function | The basic building block of all neural networks |
| **Weight** | A learnable parameter that scales the contribution of one neuron's output to the next neuron's input | Weights encode the "knowledge" learned during training |
| **Bias** | A learnable constant added to the weighted sum before activation | Allows neurons to fire even when all inputs are zero; shifts the decision boundary |
| **ReLU** | Rectified Linear Unit: f(z) = max(0, z) | Most widely used hidden-layer activation; computationally cheap and effective |
| **Softmax** | Activation function that converts logits into a probability distribution summing to 1 | Standard output activation for multi-class classification |
| **Loss Function** | A scalar measure of the difference between predictions and true labels | The quantity the training algorithm minimises |
| **Gradient Descent** | Iterative optimisation algorithm that adjusts weights by moving opposite the loss gradient | The universal training algorithm for neural networks |
| **Backpropagation** | Algorithm for computing gradients of the loss with respect to every weight using the chain rule | Makes training deep networks computationally feasible |
| **Epoch** | One full pass through the entire training dataset | A unit for measuring training progress |
| **Mini-batch** | A small subset of training data used to compute one weight update | Balances the noise of single-sample updates against the cost of full-dataset updates |

---

## Code Reference

See `code/lesson_03.py` for runnable demos: building a dense network with Keras, inspecting weight shapes, training on MNIST, and plotting training and validation curves.

---

## Activities

1. **Parameter Counting:** In `lesson_03.py`, call `model.summary()` and manually verify the parameter counts for each layer using the formula: `params = (input_size × output_size) + output_size`. Confirm your calculations match Keras's reported counts.

2. **Activation Function Comparison:** Modify the code to train three identical networks on MNIST — one with ReLU activations, one with sigmoid, one with tanh. Plot all three validation accuracy curves on the same graph. Which converges fastest? Which achieves the highest final accuracy?

3. **Learning Rate Sensitivity:** Train the same network with learning rates of 0.1, 0.01, 0.001, and 0.0001. Plot training loss curves for all four. Describe what happens at very high and very low learning rates.

4. **Manual Forward Pass:** Extract the weights and biases from a trained Keras model using `model.layers[1].get_weights()`. Take one MNIST image and manually compute the output of the first hidden layer using numpy matrix multiplication. Compare your result to `model.predict()` to verify they match.

5. **Visualise the Loss Landscape:** After training, record the final loss and accuracy. Then deliberately corrupt 10% of the training labels (assign random labels) and retrain. How does corrupted data affect convergence? What does this reveal about what the network is actually learning?

---

## Review Questions

1. Explain the forward pass in a neural network. Starting from a raw MNIST image, describe every mathematical operation that takes place before the network produces a class probability.

2. What is the purpose of activation functions, and why must they be non-linear? What would happen if every layer used only a linear activation (or no activation)?

3. Explain gradient descent in plain language without using the word "gradient." Use an analogy to describe how the algorithm finds better weight values.

4. A single dense layer has 256 input neurons and 128 output neurons. How many learnable parameters does it have (include biases)? Show your calculation.

5. What is the difference between an epoch and a mini-batch? If a training set has 60,000 examples and the batch size is 128, how many weight updates occur in one epoch? Show your calculation.

---

## Further Reading

- **"Neural Networks and Deep Learning" by Michael Nielsen (neuralnetworksanddeeplearning.com)** — Free online book with beautiful visual explanations of backpropagation and gradient descent, highly recommended for visual learners.
- **3Blue1Brown: "Neural Networks" series (YouTube)** — The most visually intuitive introduction to neural networks; Chapter 2 on gradient descent is particularly excellent.
- **"Understanding Backpropagation" — Chris Olah's blog (colah.github.io)** — Deep intuitive explanation of backpropagation with clear diagrams.
- **Keras documentation: Model training APIs** — Official guide to `model.compile()`, `model.fit()`, and all training-related parameters used in this lesson.
- **"Deep Learning" by Goodfellow et al. (Chapters 6–8)** — Rigorous mathematical treatment of feedforward networks, regularisation, and optimisation algorithms.

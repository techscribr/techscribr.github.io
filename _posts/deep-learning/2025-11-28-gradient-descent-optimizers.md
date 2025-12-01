---
title: "Deep Learning Primer: A concise introduction to Backprop and Optimizers"
date: 2025-11-28 10:00:00 +0530
categories: [Deep Learning]
tags: [Deep-Learning]
math: true
---

## 1. High-Level Overview: How Deep Learning Models Learn

At its core, a deep learning model is a function approximation machine. It tries to map inputs to outputs by adjusting its internal parameters (weights and biases) to minimize error.

### Key Concepts

  * **The Epoch:** One complete pass through the entire training dataset. A single epoch is rarely enough to learn complex patterns; we usually require many epochs.
  * **The Batch:** Since datasets can be massive (millions of images), we cannot feed them all into memory at once. We divide the data into smaller chunks called **batches**.
  * **The Forward Pass:** The input data flows through the network layers. At each layer, matrix multiplications and activation functions are applied. The network produces a prediction ($\hat{y}$).
  * **The Backward Pass (Backpropagation):** The model compares its prediction ($\hat{y}$) to the actual target ($y$) using a **Loss Function** (e.g., Mean Squared Error). It then calculates the gradient of this loss with respect to every weight in the network, moving from the output layer back to the input layer.
  * **Gradient Descent (GD):** This is the optimization algorithm. Once we know the gradients (which tell us the direction of the steepest ascent of the error), we move the weights in the opposite direction to reduce the error.

### Weight Update Formulation

The fundamental equation for updating a weight parameter $w$ is:

$$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w} =  w_{old} - \eta \cdot \nabla{L(w_t)}$$

Where:

  * $w_{old}$ is the current weight.
  * $\eta$ (eta) is the **learning rate** (step size).
  * $\frac{\partial L}{\partial w}$ is the gradient (derivative of the Loss $L$ with respect to weight $w$ at time-step t).

-----

## 2. Deep Dive into Back-Propagation

Here is the step-by-step derivation fo backprop equations for a neural network with 2 inputs and only 1 hidden layer.

### a. The Network Setup & Notation

Let's define the architecture clearly to match our requirements.

**Architecture:**

  * **Input Layer (Layer 0):** Two nodes, $x_1$ and $x_2$.
  * **Hidden Layer (Layer 1):** Two nodes, $h_1$ and $h_2$.
  * **Output Layer (Layer 2):** One node, $o_1$.
  * **Activation Function:** We will use Sigmoid ($\sigma$) for the hidden layer and Linear (Identity) for the output layer (standard for regression).
  * **Loss Function:** Mean Squared Error (MSE).

**Notation Rule:**
$w^{[l]}_{ji}$ denotes the weight in layer $l$ connecting **source node $i$** to **destination node $j$**.

  * Example: $w^{[1]}_{21}$ connects Input Node 1 ($x_1$) to Hidden Node 2 ($h_2$).

#### Diagram

![Image of Deep Neural Net](assets/img/deep-learning/neural-net.svg)

### b. Forward Pass Equations

Before deriving gradients, we must define how the signal flows forward.

**Hidden Layer Calculation:**

  * **Pre-activation:** $$z^{[1]}_1 = w^{[1]}_{11}x_1 + w^{[1]}_{12}x_2 + b^{[1]}_1$$
  * **Pre-activation:** $$z^{[1]}_2 = w^{[1]}_{21}x_1 + w^{[1]}_{22}x_2 + b^{[1]}_2$$
  * **Activation:** $$a^{[1]}_1 = \sigma(z^{[1]}_1)$$ and $$a^{[1]}_2 = \sigma(z^{[1]}_2)$$

**Output Layer Calculation:**

  * **Pre-activation:** $$z^{[2]}_1 = w^{[2]}_{11}a^{[1]}_1 + w^{[2]}_{12}a^{[1]}_2 + b^{[2]}_1$$
  * **Activation (Linear):** $$\hat{y} = a^{[2]}_1 = z^{[2]}_1$$

**Loss Calculation (MSE):**

$$L = \frac{1}{2}(y - \hat{y})^2$$

*(Note: We use $\frac{1}{2}$ to cancel out the exponent during differentiation).*

### c. Back-Propagation Derivation

We want to update the weights to minimize Loss ($L$). We use the Chain Rule to find $\frac{\partial L}{\partial w}$.

#### Part A: Gradient for Output Layer Weight ($w^{[2]}_{11}$)

*This is the weight connecting Hidden Node 1 ($h_1$) to the Output Node.*

We decompose the derivative using the chain rule:
$$\frac{\partial L}{\partial w^{[2]}_{11}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{[2]}_1} \cdot \frac{\partial z^{[2]}_1}{\partial w^{[2]}_{11}}$$

**1. Derivative of Loss w.r.t Prediction:**
$$\frac{\partial L}{\partial \hat{y}} = \frac{\partial}{\partial \hat{y}} (\frac{1}{2}(y - \hat{y})^2) = -(y - \hat{y})$$

**2. Derivative of Prediction w.r.t Pre-activation (Linear activation):**
Since $\hat{y} = z^{[2]}_1$, the derivative is 1.
$$\frac{\partial \hat{y}}{\partial z^{[2]}_1} = 1$$

**3. Derivative of Pre-activation w.r.t Weight:**

$$z^{[2]}_1 = w^{[2]}_{11}a^{[1]}_1 + \dots$$

$$\frac{\partial z^{[2]}_1}{\partial w^{[2]}_{11}} = a^{[1]}_1 \quad (\text{The input from the previous layer})$$

**Final Equation for Output Layer:** Combine the terms:

$$\frac{\partial L}{\partial w^{[2]}_{11}} = -(y - \hat{y}) \cdot a^{[1]}_1$$

#### Part B: Gradient for Hidden Layer Weight ($w^{[1]}_{21}$)

*This is the weight connecting Input Node 1 ($x_1$) to Hidden Node 2 ($h_2$).*

This path is longer. The error must propagate from the Loss $\rightarrow$ Output $\rightarrow$ Hidden Node 2 $\rightarrow$ Weight.

$$\frac{\partial L}{\partial w^{[1]}_{21}} = \frac{\partial L}{\partial z^{[2]}_1} \cdot \frac{\partial z^{[2]}_1}{\partial a^{[1]}_2} \cdot \frac{\partial a^{[1]}_2}{\partial z^{[1]}_2} \cdot \frac{\partial z^{[1]}_2}{\partial w^{[1]}_{21}}$$

Let's calculate each term:

**1. The Output Error ($\frac{\partial L}{\partial z^{[2]}_1}$):**
From Part A, we know this is the error term $\delta^{[2]}$.

$$\frac{\partial L}{\partial z^{[2]}_1} = -(y - \hat{y})$$

**2. Backpropagate through Hidden-to-Output Weight ($\frac{\partial z^{[2]}_1}{\partial a^{[1]}_2}$):**
How much does Hidden Node 2 affect the Output Sum?

$$z^{[2]}_1 = w^{[2]}_{11}a^{[1]}_1 + w^{[2]}_{12}a^{[1]}_2 + \dots$$

$$\frac{\partial z^{[2]}_1}{\partial a^{[1]}_2} = w^{[2]}_{12}$$

**3. Derivative of Hidden Activation ($\frac{\partial a^{[1]}_2}{\partial z^{[1]}_2}$):**
We used the Sigmoid function $\sigma$. The derivative of sigmoid is $\sigma(z)(1-\sigma(z))$.

$$\frac{\partial a^{[1]}_2}{\partial z^{[1]}_2} = \sigma'(z^{[1]}_2)$$

**4. Derivative of Hidden Pre-activation w.r.t Weight ($$\frac{\partial z^{[1]}_2}{\partial w^{[1]}_{21}}$$):**

$$z^{[1]}_2 = w^{[1]}_{21}x_1 + w^{[1]}_{22}x_2 + \dots$$

$$\frac{\partial z^{[1]}_2}{\partial w^{[1]}_{21}} = x_1$$

**Final Equation for Hidden Layer:**
Combine all terms:

$$\frac{\partial L}{\partial w^{[1]}_{21}} = [-(y - \hat{y}) \cdot w^{[2]}_{12} \cdot \sigma'(z^{[1]}_2)] \cdot x_1$$

### Summary of Updates

To update the weights using Gradient Descent with learning rate $\eta$:

1.  **Output Layer:** $$w^{[2]}_{11(new)} = w^{[2]}_{11(old)} - \eta \cdot [-(y-\hat{y}) \cdot a^{[1]}_1]$$
2.  **Hidden Layer:** $$w^{[1]}_{21(new)} = w^{[1]}_{21(old)} - \eta \cdot [-(y-\hat{y}) \cdot w^{[2]}_{12} \cdot \sigma'(z^{[1]}_2) \cdot x_1]$$

-----

## 3. Why Move Opposite to the Gradient?

Why do we perform $w - \nabla L_w$? Why not plus? Or divide?
The Taylor Series approximation states that for a function $L(w)$, the value at a nearby point $w + \Delta(w)$ can be approximated using its value at $w$ as follows:

$$L(w + \Delta(w)) \approx L(w) + \Delta(w) \cdot \nabla{L(w)}$$

We want the new loss $L(w + \Delta(w))$ to be **lower** than the current loss $L(w)$, because only then it makes sense to move in the direction of $\Delta(w)$.

$$L(w + \Delta(w)) < L(w)$$

This implies that the term $\Delta(w) \cdot \nabla{L(w)}$ must be **negative**, and ideally we'll want this to be as negative as possible to achieve the largest gain. Decomposing this dot product using standard definition we get:

$$\Delta(w) \cdot \nabla{L(w)} = \lVert \Delta(w) \rVert \lVert \nabla{L(w)} \rVert \cos(\alpha) $$

* $\alpha$ is the angle between these two vectors. 

We'll need $\cos(\alpha) = -1 \Rightarrow \alpha = 180^{\circ}$. This indeed implies the optimal direction of $\Delta(w)$ vector is opposite to that of $\nabla{L(w)}$. Thus, moving opposite to the gradient guarantees the largest reduction in the loss function for a sufficiently small step size.

-----

## 4. Deep Dive into Optimizers

### a. Batch vs. Stochastic vs. Mini-Batch GD

**Batch Gradient Descent:**

  * **Mechanism:** Computes the gradient using the **entire dataset** to calculate a single update.
  * **Update Freq:** Once per epoch.
  * **Problem:** Extremely slow for large datasets; requires high memory; can get stuck in saddle points.

**Stochastic Gradient Descent (SGD):**

  * **Mechanism:** Computes the gradient using a **single training example**.
  * **Update Freq:** Number of updates = Number of training examples per epoch.
  * **Solution:** Solves the speed and memory issue of Batch GD.
  * **New Problem:** The gradient is very noisy (high variance). The path to the minimum oscillates wildly, making convergence difficult.

**Mini-Batch Gradient Descent (The Middle Ground):**

  * **Mechanism:** Uses a small batch (e.g., 32, 64, 128 samples).
  * **Why:** It leverages matrix algebra (vectorization) for speed but provides a more stable estimate of the gradient than pure SGD.
  * **Accumulation:** Gradients are averaged over the batch, and the weights are updated once per batch.

### b. Momentum-Based GD

  * **Motivation:** In vanilla SGD, if the surface is like a ravine (steep in one direction, flat in another), the optimizer oscillates across the steep slopes and makes slow progress along the valley floor.
  * **Formulation:** We introduce a "velocity" term ($v$). We add a fraction ($\gamma$) of the previous update vector to the current one.

    $$v_t = \gamma v_{t-1} + \eta \nabla L(w_t)$$
      
    $$w_{t+1} = w_t - v_t$$
  
  * **How the velocity changes over time**
    * $v_0 = 0$
    * $v_1 = \gamma.0 + \eta \nabla L(w_1) = \eta \nabla L(w_1)$
    * $v_2 = \gamma.(\eta \nabla L(w_1)) + \eta \nabla L(w_2) = \gamma.\eta \nabla L(w_1) + \eta \nabla L(w_2)$
    * $v_3 = \gamma^2.\eta \nabla L(w_1) + \gamma.\eta \nabla L(w_2) + \eta \nabla L(w_3)$
    * ...
    * $v_t = \gamma^{t-1}.\eta \nabla L(w_1) + \gamma^{t-2}.\eta \nabla L(w_2) + ... +  \eta \nabla L(w_t)$

  * **Improvement:** It builds up speed in directions with consistent gradients and dampens oscillations in noisy directions. It converges faster than vanilla GD.
  * **Challenge:** It acts like a heavy ball rolling down a hill; it gains so much momentum it can overshoot the minimum.
    * It oscillates in and out of the minima valley as the momentum carries it out of the valley.
    * It takes a lot of u-turns before converging.

### c. Nesterov Accelerated Gradient (NAG)

  * **Motivation:** Momentum blindly accelerates. Nesterov tries to be smarter by "looking ahead."
  * **Formulation:** 
    * Recall the formulation: $$v_t = \gamma v_{t-1} + \eta \nabla L(w_t)$$
    * We know we're going to move by at-least $\gamma v_{t-1}$ and then a bit more by $\eta \nabla L(w_t)$.
    * Instead of calculating the gradient at the current position $w_t$ at time-step $t$, why not calculate the gradient at $w_{look-ahead}$ where $w_{look-ahead} = w_t - \gamma v_{t-1}$?

    $$w_{look-ahead} = w_t - \gamma v_{t-1}$$
    
    $$v_t = \gamma v_{t-1} + \eta \nabla L(w_{look-ahead})$$
    
    $$w_{t+1} = w_{t} - v_t$$
  
  * **Improvement:** If the momentum is about to carry us up the other side of the valley (overshooting), the "look-ahead" gradient will point back, correcting the velocity sooner.
  * **Challenge:** Still requires manual tuning of the learning rate.

### d. Adagrad (Adaptive Gradient)

  * **Motivation:** A single learning rate may not be the best fit for all surfaces. Different parameters need different learning rates. Infrequent parameters (sparse data) require larger updates, while frequent ones need smaller updates. So, it would be great to have a learning rate which can adapt to the gradient.
  * **Usual Tips for Annealing LR**
    * Halve the LR every $n$ epochs.
    * Halve the LR after an epoch if the validation error is more than its value at the end of previous epoch.
    * **Exponential decay**: $\eta = {\eta_{0}}^{-kt}$, where $\eta_0$ and $k$ are hyper-parameters, and $t$ is the time-step. 
    * **$1/t$ decay**: $\eta = \frac{\eta_0}{1 + kt}$, where $\eta_0$ and $k$ are hyper-parameters, and $t$ is the time-step. 
  * **Intuition**: For a sparse feature, the gradient update would be very small. This is a problem if the feature is important (e.g. *is_director = Nolan*). This would mean $\nabla w = 0$ for most dataset. So, it'll be good to have a different LR for each parameter which takes care of frequency of features. So, our intuition is to decay the LR for parameters in proportion to their update history.
  * **Formulation:**

    $$v_t = v_{t-1} + (\nabla L(w_t))^2$$
    
    $$w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \cdot \nabla L_{w_t}$$
  
  * **How the velocity changes over time**
    * $v_0 = 0$
    * $v_1 = (\nabla L(w_1))^2$
    * $v_2 = (\nabla L(w_1))^2 + (\nabla L(w_2))^2$
    * ...
    * $v_t = (\nabla L(w_1))^2 + (\nabla L(w_2))^2 + ... + (\nabla L(w_t))^2$

  * **Improvement:** Eliminates the need to manually tune the learning rate for every parameter.
    * **Sparse Features**
      * Appears rarely $\Rightarrow$ Gradient is often zero $\Rightarrow$ Accumulator $v_t$ grows slowly.
      * Denominator stays small $\Rightarrow$ Effective LR stays large, preventing them from dying out.
    * **Dense Features**
      * Appears in many batches $\Rightarrow$ Gradient is mostly non-zero $\Rightarrow$ Accumulator $v_t$ grows very quickly (sum of non-zero squares).
      * Denominator becomes large $\Rightarrow$ Effective LR shrinks, forcing it to take smaller, more conservative steps.
  * **Challenge:** $v_t$ accumulates positive terms forever. Eventually, the denominator becomes infinite, and the learning rate effectively shrinks to zero, stopping learning prematurely.

### e. RMSProp (Root Mean Square Propagation)

  * **Motivation:** Fix Adagrad’s "decaying learning rate" problem by decaying the denominator.
  * **Formulation:** Instead of a cumulative sum, it uses an **Exponential Moving Average** of squared gradients. It "forgets" old history.
    
    $$v_t = \beta v_{t-1} + (1-\beta)(\nabla L(w_t))^2$$ 
    
    * Usually $\beta \ge 0.9$
    
    $$w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \cdot \nabla L(w_t)$$

  * **How the velocity changes over time**
    * $v_0 = 0$
    * $v_1 = (1 - \beta)(\nabla L(w_1))^2$
    * $v_2 = \beta(1 - \beta)(\nabla L(w_1))^2 + (1 - \beta)(\nabla L(w_2))^2$
    * ...
    * $v_t = \beta^{t-1}(1 - \beta)(\nabla L(w_1))^2 + \beta^{t-2}(1 - \beta)(\nabla L(w_2))^2 + ... + (1 - \beta)(\nabla L(w_t))^2$
    * **In general**: $v_t = (1 - \beta)\sum_{k=1}^{t}\beta^{t - k}(\nabla L(w_k))^2$

  * **Improvement:** Learning rate adapts but doesn't vanish.
    * Because $0 < \beta < 1$, $\beta^{t - k} \rightarrow 0$ as $t \rightarrow \infty$. So, very old gradients almost vanish.
    * If gradients have roughly constant magnitude, RMSProp stabilizes. 
      * Assume $(\nabla L(w_t))^2 \approx G$
      * Our velocity equation becomes: $$v_t = \beta v_{t-1} + (1-\beta)G$$ 
      * If training goes on for a long time, $t \rightarrow \infty$.
        * $v_\infty = \beta v_\infty + (1 - \beta)G \Rightarrow v_\infty = G$
  * **Challenge:** It relies solely on the second moment (variance) of the gradient, lacking the momentum aspect.
    * RMSProp adapts per-parameter scales (good), but it **doesn’t keep a momentum-like running average of raw gradients** (unless you add momentum separately). Momentum helps accelerate convergence and reduce oscillation in ravines.
    * Also, RMSProp’s $v_t$ term initialized at 0 is **biased toward zero** at the beginning (especially with large $\beta$ like 0.99). This makes early updates smaller than they should be and slows initial progress.

### f. Adam (Adaptive Moment Estimation)

  * **Motivation:** Combine the best of both worlds: Momentum (First moment) and RMSProp (Second moment).
  * **Formulation:**
    1.  Calculate Momentum (Mean): $m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L(w_t)$
    2.  Calculate Variance (RMSProp): $v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L(w_t))^2$
    3.  **Bias Correction:** Since $m$ and $v$ start at 0, they are biased toward 0 initially. We correct this:

        $$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
    
    4.  **Update:**

        $$w_{t+1} = w_{t} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$
  
  * **Rationale behind Bias Correction**
    * Note that we're taking a running average of the gradients ($m_t$) in the weight update procedure instead of the true gradient $\nabla L(w_t)$ [ignoring the bias correction part]. Reason being, we don't want to rely too much on the current gradient and utilize the overall behaviour of the past gradients over many time steps.
    * So, we're more interested in $E(\nabla L(w_t))$ and not on a single point-estimate $\nabla L(w_t)$ computed at time $t$.
    * However, instead of computing $E(\nabla L(w_t))$, we are computing $m_t$ as the EMA.
    * Ideally we'd want $E(m_t) = E(\nabla L(w_t))$

  * **Bias Correction Formula Derivation**      
      $$m_t = (1 - \beta)\sum_{i=1}^t\beta^{t - i}g_i$$ where $g_i = \nabla L(w_i)$  
      $$\Rightarrow E(m_t) = E[(1 - \beta)\sum_{i=1}^t\beta^{t - i}g_i]$$  
      $$\Rightarrow E(m_t) = (1 - \beta)\sum_{i=1}^t E[\beta^{t - i} g_i]$$  
      $$\Rightarrow E(m_t) = (1 - \beta)\sum_{i=1}^t \beta^{t - i} E(g_i)$$   
      **Assumption**: All $g_i$s are from the same distribution, i.e. $E(g_i) = E(g), \forall i$  
      $$\Rightarrow E(m_t) = (1 - \beta) \cdot E(g) \cdot \sum_{i=1}^t \beta^{t - i}$$  
      $$\Rightarrow E(m_t) = (1 - \beta) \cdot E(g) \cdot \frac{1 - \beta^t}{1 - \beta}$$  
      $$\Rightarrow E(m_t) = E(g) \cdot (1 - \beta^t)$$  
      That's why we divide $m_t$ with $(1 - \beta^t)$ to get $\hat{m}_t$. Same goes for $v_t$ and $\hat{v}_t$.
  
  * **Improvement:** Generally the default optimizer for most deep learning tasks. It is fast, stable, and handles sparse gradients well.
    * Momentum (first moment) reduces noise and speeds up along consistent directions.
    * Adaptive scaling (second moment) normalizes coordinates with different gradient magnitudes (helps with ill-conditioning and sparse/dense feature imbalance).
    * Bias-correction fixes the "startup paralysis" where early steps are too small.
    * Empirically, Adam converges faster and more stably across many problems with default betas (
$\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{−8}$).
  * **Challenge:**
    * **Generalization gap**: Adam can converge faster but sometimes yields worse generalization compared to SGD with momentum; not a silver bullet.
    * **Non-convergence pathologies**: Under some circumstances Adam’s adaptive steps can cause non-convergent behavior. Variants (AMSGrad, AdamW, AdaBound, RAdam, etc.) address some of these issues.
    * **Weight decay handling**: naive L2 regularization + Adam interacts poorly. Decoupled weight decay (AdamW) is better.

---

## 5. What came after ADAM?

Since the release of **Adam** (2014), the deep learning community has developed numerous optimizers to address its three main weaknesses: **poor generalization** (often worse than SGD), **instability** (convergence issues), and **memory inefficiency**.

Here is a high-level summary of the major post-Adam optimizers, categorized by the problem they solve.

### a. AdamW (Adam with Decoupled Weight Decay)
**Summary:**
Standard Adam implements weight decay essentially as L2 regularization (adding it to the loss function). AdamW decouples the weight decay from the gradient update. It applies weight decay directly to the weights *after* the primary gradient update. This simple fix made Adam usable for state-of-the-art transfer learning (like BERT, ViT) where weight decay is critical.

* **Pros:**
    * **Significantly better generalization** than standard Adam.
    * Fixes the hidden bug in Adam's original implementation regarding L2 regularization.
    * The de-facto standard for training Transformers (BERT, GPT, etc.).
* **Cons:**
    * Requires careful tuning of the weight decay hyperparameter (it is no longer "set and forget").

### b. RAdam (Rectified Adam)
**Summary:**
Adam often suffers from high variance in the very early stages of training (bad starts) because the variance estimate is based on too few samples. This usually forces users to implement a "warm-up" period manually. RAdam internally uses a "rectifier" to dynamically turn on/off the adaptive momentum based on the variance, effectively automating the warm-up process.

* **Pros:**
    * **Removes the need for a manual warm-up period.**
    * Stabilizes training in the critical first few epochs.
    * Robust to changes in learning rate.
* **Cons:**
    * Eventually behaves exactly like Adam, so it inherits Adam's later-stage generalization issues if not combined with other techniques (like Lookahead).

### c. AMSGrad
**Summary:**
Researchers discovered a theoretical flaw in Adam where it could fail to converge for certain simple functions because the exponential moving average of past squared gradients could decrease. AMSGrad fixes this by using the **maximum** of past squared gradients rather than the exponential average.

* **Pros:**
    * **Theoretically guaranteed convergence** (unlike Adam).
    * More stable learning rates that never increase.
* **Cons:**
    * **Practical performance is often worse** than Adam.
    * Because it accumulates the "maximum" history, the learning rate can decay too quickly and "stall" training early.

### d. Nadam (Nesterov-accelerated Adam)
**Summary:**
Nadam combines Adam with **Nesterov Momentum** (NAG). Instead of updating parameters based on the current gradient, it effectively "looks ahead" by updating based on where the parameters are *going* to be, applying the momentum step before calculating the gradient.

* **Pros:**
    * Converges faster than Adam in many cases.
    * Often achieves a slightly lower final loss.
* **Cons:**
    * Slightly higher computational cost per step.
    * Often doesn't fix the generalization gap (test accuracy) compared to SGD.

### e. Lookahead
**Summary:**
Lookahead is a "wrapper" or "meta-optimizer" that can be used *with* any other optimizer (like Adam or SGD). It maintains two sets of weights: "Fast weights" (updated by the inner optimizer) and "Slow weights." Every $k$ steps, the slow weights take a step toward the fast weights, and the fast weights are reset to the slow weights.

* **Pros:**
    * **Drastically improves stability.**
    * Reduces the variance of the inner optimizer.
    * Often helps escape saddle points and local minima.
* **Cons:**
    * Introduces two new hyperparameters (sync period $k$ and slow-step size $\alpha$).
    * Slight memory overhead (storing a copy of slow weights).

### f. LAMB (Layer-wise Adaptive Moments for Batch training)
**Summary:**
Designed specifically for **large-batch training** (e.g., training BERT with batch size 32k+). Standard Adam diverges at such huge batch sizes. LAMB normalizes the update layer-wise, allowing the learning rate to scale linearly with batch size without instability.

* **Pros:**
    * Enables **massive parallel training**, reducing training time from days to minutes.
    * Standard for training Large Language Models (LLMs) from scratch.
* **Cons:**
    * Overkill and potentially unstable for small batch sizes.
    * Computationally heavier due to layer-wise norm calculations.

### g. Lion (Evolved Sign Momentum)
**Summary:**
A very recent optimizer (Google, 2023) discovered via symbolic search (AI designing AI). It drops the complex adaptive variance term of Adam entirely and uses the **sign** operation on the momentum. It is simpler and lighter than Adam.

* **Pros:**
    * **Memory Efficient:** Saves ~30-50% memory compared to Adam (tracks only momentum, not variance).
    * **Faster:** Fewer calculations per step.
    * Often outperforms AdamW on vision and language tasks.
* **Cons:**
    * **Sensitive to batch size:** The "sign" operation depends heavily on the batch noise.
    * Requires a much smaller learning rate (typically 3x-10x smaller than Adam), so old hyperparameters cannot be reused.

### Summary Table

| Optimizer | Best Use Case | Key "Win" | Key "Loss" |
| :--- | :--- | :--- | :--- |
| **AdamW** | **Default** for most Deep Learning | Fixes Weight Decay / Generalization | Need to tune `weight_decay` |
| **AMSGrad** |	Theoretical constraints	| Guaranteed Convergence (Theory) |	Often performs worse than Adam |
| **RAdam** | Unstable initial training | **No Warm-up** needed | Same late-stage issues as Adam |
| **Nadam** | If Adam is too slow | Faster convergence | Computes slightly more |
| **Lookahead** | If loss is oscillating wildly | High Stability | More Hyperparameters |
| **LAMB** | Massive Batch Sizes (LLMs) | Extremely fast training | Overkill for small projects |
| **Lion** | Memory constrained models | **Memory Efficiency** & Speed | Finicky tuning (Batch size/LR) |

---
---

## Resources
* [Deep Learning Playlist - NPTEl/IITM](https://www.youtube.com/playlist?list=PLEAYkSg4uSQ1r-2XrJ_GBzzS6I-f8yfRU)
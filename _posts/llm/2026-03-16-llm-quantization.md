---
title: "A Practical Introduction to LLM Quantization and Linear Mapping"
date: 2026-03-16 18:00:00 +0530
categories: [Deep Learning]
tags: [LLM]
math: true
---

## Why Quantization Matters for LLMs

Modern LLMs are enormous, not just in parameter count, but in the **memory and compute they demand at inference time**. A model with billions of parameters stored in FP16 can easily require **tens of gigabytes of memory**, making it impractical to run on consumer GPUs, edge devices, or mobile hardware.

Quantization tackles this problem by **compressing high-precision weights into lower-precision representations** (e.g., INT8 or INT4). The key idea is simple: most neural network weights don’t need 16-32 bits of precision to preserve the model’s behavior. By representing them with fewer bits, we can dramatically reduce memory usage and bandwidth while keeping accuracy nearly intact.

The result is transformative:

* **Smaller models** that fit on limited hardware
* **Faster inference** due to cheaper integer operations
* **Lower energy consumption**, enabling on-device AI

In short, quantization is one of the main techniques that makes **large language models practical to deploy in the real world**.

## The Landscape of LLM Quantization
To navigate the zoo of quantization acronyms (GPTQ, AWQ, QLoRA, SmoothQuant), we must map them across three distinct architectural axes: The Math, The Target, and The Timing.

### AXIS 1: The Math (How do we map the numbers?)
This defines the physical shape of the "buckets" we use to squash floating-point numbers into smaller bit representations.

#### 1. Linear (Uniform) Quantization
* **The Concept**: The integer buckets are spaced equally apart.
* **Pros**: Hardware loves it. GPUs have dedicated silicon (INT8 Tensor Cores) specifically designed to multiply evenly-spaced integers at blinding speeds.
* **Cons**: LLM weights are usually normally distributed (a Bell Curve). Most weights are near zero, and very few are at the extremes. Linear quantization wastes precious buckets on the empty "tails" of the distribution.
Examples: Standard INT8, GGUF (often uses linear block-wise INT4).

#### 2. Non-Linear (Distribution-based) Quantization
* **The Concept**: The buckets are not evenly spaced. Instead, they are crowded tightly together around 0.0 (where most the weights live) and spread further apart at the tails.
* **The Star Player - NormalFloat (NF4)**: Invented for QLoRA, NF4 assumes the weights form a perfect Gaussian bell curve. It mathematically guarantees that every 4-bit bucket will contain the exact same number of weights.
* **Pros**: Incredible accuracy preservation at extremely low bit-rates (4-bit).
* **Cons**: Hardware hates it. We cannot do matrix multiplication on NF4 numbers. We must de-quantize them back to FP16 in the GPU's SRAM right before multiplication, meaning it saves memory but doesn't speed up compute.

### AXIS 2: The Target (What are we quantizing?)
Are we quantizing just the model itself, the data flowing through it, or the memory it uses?

#### 1. Weight-Only Quantization (WOQ)
* **Standard formats**: W8A16, W4A16 (Weight in 4-bit, Activations stay in 16-bit Float).
* **Why do it?** In autoregressive generation (batch size = 1), compute is fast, but moving weights from VRAM is slow (The Memory Bandwidth Wall). By only shrinking the weights, we solve the bottleneck.
* **Algorithms**: 
  * **GPTQ**: Looks at the Second-Order derivative (the Hessian matrix). If it quantizes a weight and causes a rounding error, it mathematically updates the unquantized weights next to it to compensate for the error.
  * **AWQ**: Looks at a calibration dataset and says, "1% of these weights are critical for the activations. Let's keep those 1% in high precision and aggressively quantize the other 99%."
  
#### 2. Weight + Activation Quantization (W8A8)
* **The Goal**: Squash both the weights and the incoming tokens/activations into INT8.
* **Why do it?** If we have high batch sizes (serving 100 users at once), we are no longer memory-bound, we are compute-bound. W8A8 allows us to use INT8 Tensor Cores, doubling our raw mathematical throughput.
* **The Solution (SmoothQuant)**: Remember the "Outlier Crisis"? Activations have massive outliers. SmoothQuant uses a genius math trick: it mathematically divides the activation outliers by a scale factor, and multiplies the corresponding weights by that exact same scale factor. It "migrates" the difficulty from the activations into the weights, making both smooth enough to safely quantize to INT8.

#### 3. KV-Cache Quantization
* **The Problem**: In long-context models (e.g., Gemini's 1M context or Claude's 200K), the weights aren't the problem. The KV Cache (the memory storing the Keys and Values of the past conversation) grows so massively it crashes the GPU.
* **The Solution**: Algorithms like KIVI or FlashInfer quantize the KV cache down to 2-bit or 4-bit on the fly.

### AXIS 3: The Timing (When do we quantize?)

#### 1. Post-Training Quantization (PTQ)
* **The Concept**: The model is fully trained in FP16. We download it, run a script for an hour (using GPTQ, AWQ, or EXL2), and squash it.
* **Pros**: Fast, cheap, requires no massive compute clusters. This is the industry standard for deployment.

#### 2. Quantization-Aware Training (QAT)
* **The Concept**: We squash the model, but the accuracy drops. So, we leave it squashed, and run a Training phase (Backpropagation) to let the model "re-learn" how to be smart despite its limited precision.
* **The Star Player (QLoRA)**: we freeze the massive base model in 4-bit (NF4), but attach tiny 16-bit "Adapter" layers (LoRA) on the side. We train the 16-bit adapters. This allows we to fine-tune a massive 70B model on a single consumer GPU.

#### 3. Extreme QAT (The 1-Bit Era)
* **The Future (BitNet / 1.58b)**: Researchers are now training LLMs from scratch where every weight is strictly -1, 0, or 1.
* **The Paradigm Shift**: If weights are only 1, 0, or -1, we no longer need Matrix Multiplication. We only need Matrix Addition. This threatens to completely upend NVIDIA's dominance, as it moves LLM execution away from specialized floating-point Tensor Cores and toward standard CPU/ASIC integer addition.

## A Primer on the Affine Linear Quantization
Affine linear quantization converts high-precision floating-point values into low-precision integers using a simple linear mapping. Instead of storing each value directly, we represent it using two parameters: a scale factor ($S$) and a zero-point ($Z$), that map the floating-point range to a discrete integer range.

The general formula to quantize a floating-point value ($W_{fp}$) into an integer ($W_{int}$) is:

$$W_{int} = \text{clamp}\left( \text{round}\left( \frac{W_{fp}}{S} \right) + Z, \ Q_{min}, \ Q_{max} \right)$$

Where:
* $Q_{min}$ and $Q_{max}$ are the absolute minimum and maximum boundaries of our target discrete integer space. For instance, if we are quantizing down to standard Signed INT8, $Q_{min} = -128$ and $Q_{max} = 127$. If we are quantizing to Unsigned UINT8, $Q_{min} = 0$ and $Q_{max} = 255$.
* The $\text{clamp}()$ function acts as a safety mechanism to prevent integer overflow. If a massive floating-point outlier causes our calculation to exceed our target range, the clamp function strictly bounds it. Anything higher than $Q_{max}$ simply becomes $Q_{max}$, and anything lower than $Q_{min}$ becomes $Q_{min}$. For reference, $\text{clamp}(x, y, z)$ mathematically restricts a value $x$ between a lower bound $y$ and an upper bound $z$, defined as: $\text{clamp}(x, y, z) = \max(y, \min(x, z))$.

To de-quantize it back during inference (so the GPU can accumulate the results):

$$W_{fp\_approx} = S \times (W_{int} - Z)$$

This mapping can be implemented in two distinct flavors, depending on how we choose to align the floating-point values with the integer range: Symmetric and Asymmetric quantization.

### Flavor 1: Symmetric Quantization (The Math)
In symmetric quantization, we force the floating-point range to be perfectly mirrored around zero. We do this by finding the maximum absolute value in the tensor and using it as the symmetric boundary $[-W_{max}, +W_{max}]$.

#### The Formulation
* Scale Factor ($S$): We divide our symmetric floating-point maximum by our maximum integer bucket.

$$S = \frac{\max(|W_{actual\_min}|, |W_{actual\_max}|)}{Q_{max}}$$

* Zero-Point ($Z$): Because the range is perfectly centered on zero, the zero-point is strictly defined as $0$.

$$Z = 0$$

#### The Justification
By mathematically guaranteeing that $Z = 0$, symmetric quantization eliminates an entire term during matrix multiplication. Computing $X \times W$ on the GPU is vastly faster because the Tensor Cores don't have to constantly add and subtract a zero-point offset during the millions of dot-product operations.

#### The Example
Let's say a layer's weights range from $-3.0$ to $+5.0$. We want to quantize to Signed INT8 (range $[-127, 127]$).
1. Find Maximum: $\max(\lvert -3.0 \rvert, \lvert 5.0 \rvert) = 5.0$. Our symmetric range is artificially forced to $[-5.0, +5.0]$.
2. Calculate Scale ($S$): $S = \frac{5.0}{127} \approx 0.03937$.
3. Calculate Zero-Point ($Z$): $Z = 0$.
4. Quantize a value ($3.5$): $W_{int} = \text{round}\left(\frac{3.5}{0.03937}\right) + 0 = 89$.

*Notice the drawback: The real data only went down to $-3.0$, but we reserved integer buckets all the way down to $-5.0$. We wasted a portion of the negative INT8 buckets.*

### Flavor 2: Asymmetric Quantization (The Math)
In asymmetric quantization, we map the exact, unmanipulated minimum and maximum of the floating-point data to the absolute minimum and maximum of the integer range.

#### The Formulation
* Scale Factor ($S$): We divide the entire continuous range by the entire discrete range.

$$S = \frac{W_{actual\_max} - W_{actual\_min}}{Q_{max} - Q_{min}}$$

* Zero-Point ($Z$): We must ensure that a floating-point $0.0$ maps to a precise, whole integer.

$$Z = \text{round}\left( Q_{min} - \frac{W_{actual\_min}}{S} \right)$$

#### The Justification
Asymmetric quantization doesn't waste any integer buckets. It tightly hugs the data distribution. This is absolutely critical for data that is inherently skewed or one-sided (like activation functions such as ReLU, which output only positive numbers).

#### The Example
Let's use the exact same data range: $-3.0$ to $+5.0$. We want to quantize to Unsigned UINT8 (range $[0, 255]$).
1. Calculate Scale ($S$): $S = \frac{5.0 - (-3.0)}{255 - 0} = \frac{8.0}{255} \approx 0.03137$. (Notice this scale is smaller than the symmetric one, meaning higher precision).
2. Calculate Zero-Point ($Z$): $Z = \text{round}\left( 0 - \frac{-3.0}{0.03137} \right) = \text{round}(95.6) = 96$.
3. Quantize a value ($0.0$): $W_{int} = \text{round}\left(\frac{0.0}{0.03137}\right) + 96 = 96$.
4. Quantize a value ($5.0$): $W_{int} = \text{round}\left(\frac{5.0}{0.03137}\right) + 96 = 159 + 96 = 255$. (Maps perfectly to the top of the bucket).

### Demystifying Symmetric and Asymmetric Quantization

#### Question 1: Weights vs. Activations

*Is symmetric quantization more useful for weights while asymmetric quantization is more useful for activations?*

Yes, absolutely. This is the industry-standard architecture for LLM inference (often called W8A8 - Weight 8-bit, Activation 8-bit).
* **Why Weights use Symmetric**: During training, L2 Regularization (Weight Decay) forces the model's weights to form a beautiful, symmetrical Bell Curve centered exactly at 0.0. Because the real data is naturally symmetric (e.g., -3.1 to +3.1), Symmetric Quantization fits it perfectly without wasting integer buckets. Furthermore, it forces the Zero-Point ($Z$) to be exactly 0, which makes the GPU's matrix multiplication hardware run at maximum speed.
* **Why Activations use Asymmetric**: Activations are the outputs of non-linear functions (like ReLU, GeLU, SiLU). ReLU, for example, forces all negative numbers to 0. If our activation data is strictly positive (e.g., 0.0 to 15.5), using Symmetric Quantization would force the algorithm to pretend the range is $[-15.5, +15.5]$. We would completely waste all the negative integer buckets (half of our precision!). Asymmetric Quantization tightly hugs the exact $[0.0, 15.5]$ range, preserving all our precision.

#### Question 2: Clarifying Symmetric Quantization

*Does the float range surround 0.0? Can we use both signed/unsigned integers here? Why is it called symmetric?*

**The Float Range Concept**
In Symmetric Quantization, the real float values in our tensor might be skewed (e.g., -3.5 to +4.7). However, the algorithm ignores the minimum. It simply looks for the absolute maximum value: $\max(|-3.5|, |4.7|) = 4.7$. Then, it artificially forces the range to become perfectly mirrored around zero: $[-4.7, +4.7]$.

**Signed vs. Unsigned Integers**
**We MUST use Signed Integers** (e.g., INT8: -127 to +127). We cannot use Unsigned Integers (UINT8: 0 to 255). Why? Because the defining mathematical rule of Symmetric Quantization is that the floating-point 0.0 MUST map exactly to the integer 0. If we used UINT8, the middle of the integer range is 128, meaning 0.0 would map to 128. That breaks the rule!

*Note: Standard INT8 goes from -128 to 127. But to be perfectly symmetrical around 0, Symmetric algorithms often throw away the -128 bucket and only use -127 to 127.*

**Why is it called "Symmetric"?**
It is called Symmetric because both the float boundary and the integer boundary are mathematically mirrored around zero.
* Float boundary: $[-max, +max]$
* Integer boundary: $[-127, +127]$

Because both boundaries are perfectly centered on zero, $0.0$ float is guaranteed to equal $0$ integer. No offset ($Z=0$) is required.

#### Question 3: Clarifying Asymmetric Quantization
Does the float range start at 0.0? Can we use both signed/unsigned? Why is it called asymmetric?

**The Float Range Concept**
The float range does not have to start at 0.0. It can be $[0.0, 9.37]$, but it can just as easily be $[-2.1, 5.8]$ or $[10.5, 20.0]$. In Asymmetric Quantization, the algorithm does not artificially manipulate the boundaries. It takes the exact min and max of the actual data and maps them exactly as they are.

**Signed vs. Unsigned Integers**
We can use either, but Unsigned Integers (UINT8: 0 to 255) are heavily preferred. Since we are tightly mapping an arbitrary range (like $[-2.1, 5.8]$), it makes logical sense to map the absolute bottom (-2.1) to the absolute bottom integer bucket (0). If we used signed INT8, we would map -2.1 to -128. The math works exactly the same, but using 0 to 255 is easier for software engineers to debug.

**Why is it called "Asymmetric"?**
It is called Asymmetric because the real data boundaries are not perfectly mirrored around zero, and therefore, the mapping is offset. If our range is $[-2.1, 5.8]$, 0.0 is not in the exact center of the data. Therefore, when we map it to integers, $0.0$-float will NOT map to $0$-int. It will map to some arbitrary integer offset like 68. Because $Z \neq 0$, the mapping has an asymmetric offset.

### The Ultimate Summary Table

| Feature | Symmetric Quantization | Asymmetric Quantization |
| --- | --- | --- |
| Mathematical Rule | $Z$ must equal $0$ | $Z$ can be any integer |
| Float Range Used | Artificially forced to [-max, +max] | Exact actual [min, max] |
| Integer Type | Strictly Signed (e.g., INT8) | Usually Unsigned (e.g., UINT8) |
| Best Used For | Weights (Naturally centered on 0) | Activations (Often skewed / positive) |
| Compute Speed | Maximum (No offset math required) | Slower (Requires offset cross-terms) |
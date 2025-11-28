---
title: "Connection between KL Divergence & Cross Entropy"
date: 2025-11-17 15:00:00 +0530
categories: [Machine Learning]
tags: [ML-Theory, Classification]
math: true
---

## What's the connection between KL Divergence and Cross Entropy?

This question connects the mathematical definitions of information theory directly to the concept of a classification loss function. Let's understand it with a coin toss experiment.

### The Connection: KL Divergence and Cross-Entropy

The key relationship is:

$$
\text{Cross-Entropy}(P, Q) = \text{KL Divergence}(P || Q) + \text{Entropy}(P)
$$

Where:
* **$P$ (True Distribution):** The probability distribution of the **True Coin** (the ground truth, which we observe).
* **$Q$ (Simulating Distribution):** The probability distribution of the **Simulating Coin** (our model's prediction).
* **$\text{Entropy}(P)$:** The inherent uncertainty of the True Coin, which is a **constant** value that does not depend on our model $Q$.

Since Entropy(P) is a constant, **minimizing Cross-Entropy is mathematically equivalent to minimizing KL Divergence.**

$$
\text{Minimizing Cross-Entropy} \propto \text{Minimizing KL Divergence}
$$

This is the core reason why Cross-Entropy is used as a loss function: it forces the model $Q$ to become as close to the true distribution $P$ as possible.

---

### Step-by-Step Derivation (The Coin Flip Example)

Let's define our coin flip scenario (a Bernoulli trial).

* **Heads ($H$):** Outcome is 1.
* **Tails ($T$):** Outcome is 0.

#### Definitions

1.  **True Coin ($P$):** The true, observed distribution (the ground truth, which we observe in our training data).
    * $P(H) = p$ (The true probability of heads)
    * $P(T) = 1 - p$

2.  **Simulating Coin ($Q$):** Our model's predicted distribution (the output of our neural network).
    * $Q(H) = q$ (The model's predicted probability of heads)
    * $Q(T) = 1 - q$

---

#### A. Deriving KL Divergence

KL Divergence measures the dissimilarity between $P$ and $Q$.

$$
KL(P || Q) = \sum_{i \in \{H, T\}} P(i) \cdot \log\left(\frac{P(i)}{Q(i)}\right)
$$

$$
KL(P || Q) = P(H) \log\left(\frac{P(H)}{Q(H)}\right) + P(T) \log\left(\frac{P(T)}{Q(T)}\right)
$$

Substitute the probabilities ($p$ and $q$):

$$
KL(P || Q) = p \log\left(\frac{p}{q}\right) + (1-p) \log\left(\frac{1-p}{1-q}\right)
$$

Using the log rule $\log(a/b) = \log(a) - \log(b)$:

$$
KL(P || Q) = p [\log(p) - \log(q)] + (1-p) [\log(1-p) - \log(1-q)]
$$

Rearrange the terms:

$$
KL(P || Q) = \underbrace{\left[ -p \log(q) - (1-p) \log(1-q) \right]}_{\text{Term 1}} + \underbrace{\left[ p \log(p) + (1-p) \log(1-p) \right]}_{\text{Term 2}}
$$

---

#### B. Deriving Cross-Entropy and Entropy

**1. Cross-Entropy ($H(P, Q)$):**

Cross-Entropy measures the average number of bits needed to encode events from the true distribution $P$ when using the simulating distribution $Q$:

$$
H(P, Q) = \sum_{i \in \{H, T\}} -P(i) \log(Q(i))
$$

$$
H(P, Q) = -P(H) \log(Q(H)) - P(T) \log(Q(T))
$$

Substitute the probabilities:

$$
H(P, Q) = -p \log(q) - (1-p) \log(1-q)
$$

**Notice this is exactly Term 1 from the KL derivation.**

**2. Entropy ($H(P)$):**

Entropy measures the average number of bits needed to encode events from the true distribution $P$ using the true distribution $P$:

$$
H(P) = \sum_{i \in \{H, T\}} -P(i) \log(P(i))
$$

$$
H(P) = -p \log(p) - (1-p) \log(1-p)
$$

If we look back at Term 2 from the KL derivation, Term 2 is simply **$-H(P)$**.

---

### C. The Final Connection

Substituting the definitions back into the expanded KL formula:

$$
KL(P || Q) = \underbrace{\left[ -p \log(q) - (1-p) \log(1-q) \right]}_{\text{Cross-Entropy (H(P, Q))}} - \underbrace{\left[ -p \log(p) - (1-p) \log(1-p) \right]}_{\text{Entropy (H(P))}}
$$

Therefore:
$$
KL(P || Q) = H(P, Q) - H(P)
$$

Rearranging this gives us the fundamental relationship:
$$
H(P, Q) = KL(P || Q) + H(P)
$$

Since $H(P)$ (the true coin's inherent uncertainty) is fixed by the data, minimizing the Cross-Entropy $H(P, Q)$ achieves the goal of minimizing the dissimilarity: $KL(P \Vert Q)$.
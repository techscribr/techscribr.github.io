---
title: "Probability Distribution: Binomial"
date: 2025-10-20 12:00:00 +0530
categories: [Probability]
tags: [Probability-Distribution]
math: true
---

The **Binomial distribution** is a cornerstone of probability theory that models the number of successes in a fixed number of independent trials. This distribution focuses on "yes/no" outcomes in a set experiment.

The distribution requires four specific conditions:
1.  **Fixed Trials:** The number of trials, **n**, is fixed in advance.
2.  **Binary Outcomes:** Each trial has only two outcomes (Success/Failure).
3.  **Independence:** Trials are independent of one another.
4.  **Constant Probability:** The probability of success, **p**, remains the same for every trial.

## Probability Mass Function (PMF)
The PMF calculates the probability of getting exactly **k** successes in **n** trials.

$$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$$

Where:
* **k** is the number of successes ($0, 1, 2, ..., n$).
* **n** is the total number of trials.
* **p** is the probability of success in a single trial.
* **$\binom{n}{k}$** is the binomial coefficient ("n choose k"), calculated as $\frac{n!}{k!(n-k)!}$.

![Image of Binomial PMF](assets/img/probability-distributions/binomial.png)

## Derivations

The mean of a Binomial distribution is $np$ and the variance is $np(1-p)$. It is intuitive to derive them by viewing the Binomial distribution as a sum of **Bernoulli trials**.

Let $X = I_1 + I_2 + ... + I_n$, where $I_i$ is an indicator variable for the $i$-th trial.
* $I_i = 1$ if success (probability $p$)
* $I_i = 0$ if failure (probability $1-p$)

### Derivation of the Mean (E[X])
Since the expectation operator is linear ($E[X+Y] = E[X] + E[Y]$):

1.  **Expectation of a Single Trial:**

    $$E[I_i] = 1 \cdot p + 0 \cdot (1-p) = p$$

2.  **Summing them up:**

    $$E[X] = E[\sum_{i=1}^{n} I_i] = \sum_{i=1}^{n} E[I_i] = \sum_{i=1}^{n} p$$

3.  **Final Result:**

    $$E[X] = np$$

### Derivation of the Variance (Var(X))
Because the trials are **independent**, the variance of the sum is equal to the sum of the variances.

1.  **Variance of a Single Trial:**

    $$Var(I_i) = E[I_i^2] - (E[I_i])^2$$

    Since $1^2 = 1$ and $0^2=0$, $E[I_i^2] = p$.

    $$Var(I_i) = p - p^2 = p(1-p)$$

2.  **Summing them up:**

    $$Var(X) = \sum_{i=1}^{n} Var(I_i) = \sum_{i=1}^{n} p(1-p)$$

3.  **Final Result:**

    $$Var(X) = np(1-p)$$

## Practical Scenarios

### Scenario 1: Quality Control in Manufacturing
**Situation:** A factory produces light bulbs where **5% are defective** ($p = 0.05$). A quality control inspector randomly samples a batch of **10 bulbs** ($n = 10$). What is the probability that exactly **2 bulbs** are defective?

**Derivation:**
We use the PMF with $n=10$, $k=2$, and $p=0.05$.

$$P(X=2) = \binom{10}{2} (0.05)^2 (0.95)^{8}$$

First, calculate the combination:

$$\binom{10}{2} = \frac{10 \times 9}{2 \times 1} = 45$$

Then plug in the values:

$$P(X=2) = 45 \cdot 0.0025 \cdot 0.6634 \approx 0.0746$$

There is approximately a **7.5%** chance of finding exactly 2 defective bulbs in the batch.

### Scenario 2: Basketball Free Throws
**Situation:** A basketball player has a specific free-throw success rate of **80%** ($p = 0.8$). During a game, they take **5 shots** ($n=5$). What is the probability they make exactly **4 shots**?

![Image of Basketball player shooting free throw](assets/img/probability-distributions/binomial-example.jpg)

**Derivation:**
We use the PMF with $n=5$, $k=4$, and $p=0.8$.

$$P(X=4) = \binom{5}{4} (0.8)^4 (0.2)^{1}$$

Since $\binom{5}{4} = 5$:

$$P(X=4) = 5 \cdot 0.4096 \cdot 0.2 = 0.4096$$

There is roughly a **41%** chance the player will make exactly 4 out of 5 shots.

## Connection with Bernoulli and Poisson

The Binomial distribution serves as a bridge between the simplest and one of the most common distributions in statistics.

### 1. The Building Block: Bernoulli Distribution
The relationship is straightforward: **A Binomial distribution is the sum of $n$ independent Bernoulli trials.**
* If you set $n=1$ in a Binomial distribution, it becomes exactly a Bernoulli distribution.
* Bernoulli deals with a *single* coin toss; Binomial deals with *counting heads* in multiple tosses.

### 2. The Limit: Poisson Distribution
The Poisson distribution is actually a special case of the Binomial distribution.
* If you increase the number of trials to infinity ($n \to \infty$) while simultaneously decreasing the probability of success to zero ($p \to 0$) in such a way that the average number of successes remains constant ($np = \lambda$), the Binomial distribution transforms mathematically into the Poisson distribution. Essentially, if you have a Binomial scenario with a huge number of opportunities for an event to occur, but the chance of it occurring on any single opportunity is tiny, it can be simplified and modeled as a Poisson distribution.

#### The Conditions for the Approximation

A Binomial distribution can be well-approximated by a Poisson distribution if:
* The number of trials **n** is large (e.g., n > 50).
* The probability of success **p** is small (e.g., p < 0.1).
* The mean of the Binomial distribution, **$\lambda = n * p$**, is a moderate, finite number. This $\lambda$ becomes the rate parameter for the Poisson distribution.

#### Practical Example: Manufacturing Defects

Let's see how this connection is useful.

* **Binomial Scenario:** A factory produces **10,000** microchips ($n=10,000$). The probability that any single chip is defective is **0.01%** ($p=0.0001$). What's the probability of finding exactly 3 defective chips?
    * Calculating this with the Binomial formula is very difficult because it involves huge numbers like $10,000!$.

* **Poisson Approximation:**
    1.  **Check the conditions:** $n=10,000$ is very large and $p=0.0001$ is very small. The conditions are met.
    2.  **Calculate the rate parameter ($\lambda$):** $\lambda = n * p = 10,000 * 0.0001 = 1$.
    3.  **Use the Poisson formula:** Now we can model this as a Poisson distribution with $\lambda=1$. The probability of finding exactly 3 defective chips ($k=3$) is:

        $$P(X=3) = \frac{1^3 e^{-1}}{3!} \approx \frac{0.3679}{6} \approx 0.0613$$

The Poisson distribution simplifies the problem by focusing on the average rate of defects (1 per 10,000 chips) rather than the outcome of thousands of individual trials.

### Summary of the Connection

| Feature | Binomial Distribution | Poisson Distribution |
| :--- | :--- | :--- |
| **Focus** | Number of successes in a *fixed number of trials*. | Number of events in a *fixed interval* (time/space). |
| **Parameters** | $n$ (trials), $p$ (probability) | $\lambda$ (average rate) |
| **Number of Trials** | Finite | Effectively infinite |
| **Probability** | Can be large | Must be very small |
| **Key Relationship**| As $n \rightarrow \infty$ and $p \rightarrow 0$, Bin($n, p$) $\rightarrow$ Poisson($\lambda = n*p$) | |
   
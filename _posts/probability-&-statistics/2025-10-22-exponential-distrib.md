---
title: "Probability Distribution: Exponential"
date: 2025-10-22 10:00:00 +0530
categories: [Probability]
tags: [Probability-Distribution]
math: true
---

The exponential distribution is a continuous probability distribution used to model the time between events in a Poisson process (i.e., events that occur independently and at a constant average rate).

## Probability Density Function (PDF)

The **Probability Density Function (PDF)** gives the probability of the time between events.

$$f(x) = \lambda e^{-\lambda x}$$

Where:
* **x** is the random variable representing the time between events ($x \ge 0$).
* **$\lambda$** (lambda) is the rate parameter, representing the average number of events per unit of time.
* **e** is Euler's number.

![Image of Exponential PDF](assets/img/probability-distributions/exponential.png)

## Derivations

### Derivation of the Mean (E[X])

The mean, or expected time between events, is calculated by integrating $x \cdot f(x)$ over its entire range. This requires integration by parts ($\int u \,dv = uv - \int v \,du$).

1.  **Set up the integral:**
    
    $$E[X] = \int_{0}^{\infty} x \cdot \lambda e^{-\lambda x} \,dx$$

2.  **Apply integration by parts:**
    Let $u = x$ (so $du = dx$) and $dv = \lambda e^{-\lambda x} dx$ (so $v = -e^{-\lambda x}$).
    
    $$E[X] = \left[-x e^{-\lambda x}\right]_{0}^{\infty} - \int_{0}^{\infty} (-e^{-\lambda x}) \,dx$$

3.  **Evaluate the terms:**
    The first term, $[-x e^{-\lambda x}]$, evaluates to 0 at both infinity (as $e^{-\lambda x}$ goes to zero faster than $x$ goes to infinity) and at 0. The second term becomes:
    
    $$E[X] = \int_{0}^{\infty} e^{-\lambda x} \,dx = \left[-\frac{1}{\lambda} e^{-\lambda x}\right]_{0}^{\infty}$$
    
    This evaluates to $0 - (-\frac{1}{\lambda}) = \frac{1}{\lambda}$.

4.  **Final Result:**
    
    $$E[X] = \frac{1}{\lambda}$$

### Derivation of the Variance (Var(X))

The variance is calculated using the formula $Var(X) = E[X^2] - (E[X])^2$.

1.  **Calculate E[X²]:**
    
    $$E[X^2] = \int_{0}^{\infty} x^2 \cdot \lambda e^{-\lambda x} \,dx$$
    
    We use integration by parts again. Let $u = x^2$ ($du = 2x \,dx$) and $dv = \lambda e^{-\lambda x} dx$ ($v = -e^{-\lambda x}$).
    
    $$E[X^2] = \left[-x^2 e^{-\lambda x}\right]_{0}^{\infty} - \int_{0}^{\infty} (-e^{-\lambda x}) \cdot 2x \,dx$$
    
    The first term is 0. The second term simplifies to:
    
    $$E[X^2] = \int_{0}^{\infty} 2x e^{-\lambda x} \,dx = \frac{2}{\lambda} \int_{0}^{\infty} x \lambda e^{-\lambda x} \,dx$$
    
    We recognize the integral as the definition of $E[X]$, which we already found to be $\frac{1}{\lambda}$.
    
    $$E[X^2] = \frac{2}{\lambda} \cdot E[X] = \frac{2}{\lambda} \cdot \frac{1}{\lambda} = \frac{2}{\lambda^2}$$

2.  **Calculate the Variance:**
    
    $$Var(X) = E[X^2] - (E[X])^2 = \frac{2}{\lambda^2} - \left(\frac{1}{\lambda}\right)^2 = \frac{2}{\lambda^2} - \frac{1}{\lambda^2}$$

3.  **Final Result:**
    
    $$Var(X) = \frac{1}{\lambda^2}$$

## Practical Examples

1.  **Customer Service:** The exponential distribution can model the time between consecutive customer arrivals at a call center. If a center receives an average of 10 calls per hour ($\lambda=10$), the average time between calls would be $1/\lambda = 1/10$ of an hour, or 6 minutes. This helps in staffing and resource allocation.

2.  **Reliability Engineering:** It's used to model the lifetime of electronic components, like an LED light bulb, that fail at a constant rate rather than wearing out. If a type of bulb has a failure rate of 1 per 50,000 hours ($\lambda = 1/50000$), its mean time to failure (MTTF) is $1/\lambda = 50,000$ hours. This is crucial for setting warranties and maintenance schedules.

For the exponential distribution, **P(a ≤ X ≤ b)** indicates the probability that the **time until the next event** falls between time **a** and time **b**.

### Parallel to the Poisson Distribution

The relationship is about **timing vs. counting**.

* **Exponential Distribution (Timing):** $P(a ≤ X ≤ b)$ answers the question, "What is the chance the next phone call arrives between 2 and 5 minutes from now?" It's about the timing of a *single* event.

* **Poisson Distribution (Counting):** The equivalent concept would be finding the probability that the *number* of events falls within a certain range. For example, "What is the chance we get between 2 and 5 phone calls in the next hour?" You would calculate this by summing the individual probabilities:
    
    $$P(2 \le \text{Count} \le 5) = P(X=2) + P(X=3) + P(X=4) + P(X=5)$$

In short, the exponential distribution gives a probability over a **continuous interval of time**, while the Poisson distribution gives a probability over a **discrete range of counts**.
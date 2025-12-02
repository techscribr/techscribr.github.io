---
title: "Stats Primer: A Detailed Look at z & t-statistic"
date: 2025-10-29 11:00:00 +0530
categories: [Statistics]
tags: [Statistics]
math: true
---

## 1. What is Hypothesis Testing?

Hypothesis testing is a statistical framework used to determine if observed data provides enough evidence to support a specific belief about a population. It involves comparing a sample's metric (like a mean or proportion) against a known value or another sample to see if the difference is "real" or just due to random chance.

To perform a test, we establish two opposing hypotheses:

* **Null Hypothesis ($H_0$):** This is the default position or the "status quo." It assumes there is no relationship, no difference, or that an effect does not exist.
    * For a one-proportion test, it assumes the population proportion equals a specific value ($p = p_0$).
    * For a two-proportion test, it assumes the proportions of two groups are equal ($p_1 = p_2$).
* **Alternative Hypothesis ($H_a$):** This is the claim you are trying to validate. It assumes there is a significant difference or effect.
    * For example, in a two-proportion test, it assumes $p_1 \neq p_2$.

## 2. Z-Statistic and T-Statistic: Bird's Eye View

### **Z-statistic (Z-score)**

A **z-statistic** measures how many standard errors your sample estimate (e.g. sample mean) is away from the hypothesized population value (e.g. population mean) **when the population standard deviation is known** (or the sample is large enough for CLT to apply).

$$z = \frac{\hat{\theta} - \theta_0}{\sigma / \sqrt{n}}$$

Where:

* ($\theta_0$) = hypothesized value
* ($\sigma$) = **population** standard deviation
* ($n$) = sample size

#### **Intuition:**

It tells you how “extreme” your sample is under a **normal distribution** with known variance.

### **T-statistic**

A **t-statistic** measures how many estimated standard errors your sample is away from the hypothesized value **when the population standard deviation is unknown** and must be estimated from the sample.

$$t = \frac{\hat{\theta} - \theta_0}{s / \sqrt{n}}$$

Where:

* ($s$) = **sample** standard deviation
* Degrees of freedom = ($n - 1$) (or a variant for two samples)

#### **Intuition:**

It accounts for **extra uncertainty** from estimating the variance, especially for small samples.
It uses the **t-distribution**, which has heavier tails than the normal distribution.

### **In one sentence**

* **Z-statistic:** use when variance is known; standardized difference measured using a normal distribution.
* **T-statistic:** use when variance is unknown; standardized difference measured using a t-distribution with heavier tails.

## 3. Hypothesis Testing: P-Value vs. Critical Value (Confidence Interval)

There are two interchangeable ways to make a decision in hypothesis testing. Both methods rely on a **Significance Level ($\alpha$)**, typically set at 0.05, which is the probability of a Type I error (False Positive).

**1. The P-Value Approach:**
The p-value is the probability of observing a result as extreme as yours if the Null Hypothesis were true (i.e. just by sheer luck).
* **Decision Rule:** If the **p-value is less than $\alpha$**, you reject the Null Hypothesis.
* If the p-value is high (e.g., 0.35), you conclude that the difference is likely due to random noise.

**2. The Critical Value (Confidence Interval) Approach:**
The critical value is the "line in the sand" on the distribution curve.
* For a standard Z-test with $\alpha = 0.05$, the critical values are $\pm 1.96$ ($Z_{\alpha/2}$).
* **Decision Rule:** If your calculated Z-statistic or T-statistic exceeds the critical value (e.g., $Z_{score} > 1.96$), the result falls into the rejection region, and you reject $H_0$.

**Inter-operability:**
For parametric tests like z-test or t-test we'll use the 2nd approach for uniformity and for non-parametric tests like permutation test or bootstrapping, we'll use the 1st approach.

## 4. Use Cases: When to Use Which Test?

* **1-Sample Test (z, t-stat):** Used when you have **one group** and want to compare its statistic to a **known, stable, or fixed population value**.
    * *Examples:* Manufacturing quality control (comparing bolts to a spec) or comparing student scores to a national average.
* **2-Sample Test (z, t-stat):** Used when you have **two independent groups** and want to compare them to each other to see if they are different.
    * *Examples:* A/B testing (Control vs. Variant) or comparing medical drug efficacy vs. placebo.
* **1-Proportion Test (z-stat only):** Used for categorical data (success/failure) for a single group against a benchmark.
    * *Example:* Validating a vendor's claim that their defect rate is 2%.
* **2-Proportion Test (z-stat only):** Used for categorical data comparing two independent groups.
    * *Example:* Comparing conversion rates of two website designs.

## 5. Hypothesis Testing via Z-Test (Means)

### 1-Sample Z-Test Example
* **Formulation:** Used when comparing a sample mean ($\bar{x}$) to a known population mean ($\mu_0$).

    $$Z = \frac{\bar{x} - \mu_0}{\frac{\sigma}{\sqrt{n}}}$$

    * $\bar{x}$: Sample mean
    * $\mu_0$: Hypothesized population mean
    * $\sigma$: Population standard deviation
    * $n$: Sample size
* **Scenario:** A factory produces bolts with a **known true diameter** ($\bar{x}$) of 15 mm and a **known population standard deviation** ($\sigma$ = 0.2) mm. You collect a sample of (n = 100) bolts to check whether the machine has drifted.
* **Hypothesis:** $H_0: \mu = 15$; $H_a: \mu \neq 15$. This is a **two-tailed test**.
* **Critical Value:** At ($\alpha$ = 0.05), the two-tailed critical value is:

    $$Z_{\text{crit}} = \pm 1.96$$

    If the absolute Z-score exceeds 1.96, we reject ($H_0$).
* **Calculation:** Suppose the sample mean you observed is: $\bar{x} = 15.05 \text{mm}$

    Then:

    $$Z = \frac{15.05 - 15}{0.2 / \sqrt{100}} = \frac{0.05}{0.02} = 2.5$$

* **Decision Rule:** Since ($Z = 2.5$) and ($2.5 > 1.96$) we **reject** the null hypothesis.
* **Conclusion:** Because the Z-statistic exceeds the critical threshold, there is statistically significant evidence at the 5% level that the machine’s mean bolt diameter is no longer 15 mm.
* **Action:** The machine should be recalibrated.

### 2-Sample Z-Test Example
* **Formulation:** Used when comparing **two independent sample means**, assuming both populations have **known standard deviations**.

  $$Z = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}}$$

  * ($\bar{x}_1$, $\bar{x}_2$): Sample means
  * ($\sigma_1$, $\sigma_2$): **Population** standard deviations
  * ($n_1$, $n_2$): Sample sizes for the two groups
* **Scenario:** A company uses **two different packing machines** (Machine A and Machine B) to fill cereal boxes.
  Historically, both machines have very stable fill weights with **known population standard deviations**
  ($\sigma_1 = \sigma_2 = 5$) grams.
  You want to test whether the machines fill differently.
  * Sample from Machine A: ($n_1 = 64$), ($\bar{x}_1 = 502\text{g}$)
  * Sample from Machine B: ($n_2 = 64$), ($\bar{x}_2 = 498\text{g}$)
* **Hypothesis:** $H_0 : \mu_1 = \mu_2; H_a : \mu_1 \neq \mu_2$. This is a **two-tailed test**.
* **Critical Value:** At ($\alpha = 0.05$), the two-tailed critical value is:
  
    $$Z_{\text{crit}} = \pm 1.96$$

    If the absolute Z-score exceeds 1.96, we reject ($H_0$).
* **Calculation:**

    $$Z = \frac{502 - 498}{\sqrt{5^2/64 + 5^2/64}}
    = \frac{4}{\sqrt{0.390625 + 0.390625}}
    = \frac{4}{\sqrt{0.78125}}
    = \frac{4}{0.884} \approx 4.52
    $$

* **Decision Rule:** Since ($Z = 4.52$) and (4.52 > 1.96), we **reject** the null hypothesis.
* **Conclusion:** There is statistically significant evidence at the 5% level that the two packing machines produce **different mean fill weights**.
* **Action:** The company should investigate calibration differences between Machine A and Machine B.

### 1-Proportion Z-Test Example
* **Formulation:** Used when comparing a **sample proportion** ($\hat{p}$) to a **hypothesized population proportion** ($p_0$).

    $$Z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1 - p_0)}{n}}}$$

    * ($\hat{p}$): Sample proportion
    * ($p_0$): Hypothesized proportion
    * ($n$): Sample size
* **Scenario:** A website historically has a **signup conversion rate of 12%**. A new landing page design is tested on a sample of (n = 500) visitors. You want to check whether the conversion rate has changed. Suppose you observe **70 signups**, so:

  $$\hat{p} = \frac{70}{500} = 0.14$$

* **Hypothesis:** $H_0 : p = 0.12$; $H_a : p \neq 0.12$. This is a **two-tailed test**.
* **Critical Value:** At ($\alpha = 0.05$), the two-tailed critical value is:

    $$Z_{\text{crit}} = \pm 1.96$$

    Reject $H_0$ if ($\lvert Z \rvert > 1.96$).
* **Calculation:**

    $$Z = \frac{0.14 - 0.12}{\sqrt{\frac{0.12(0.88)}{500}}}
    = \frac{0.02}{\sqrt{0.0002112}}
    = \frac{0.02}{0.01453}
    \approx 1.38$$

* **Decision Rule:** Since (Z = 1.38) and (1.38 < 1.96), we **fail to reject** the null hypothesis.
* **Conclusion:** There is **not enough evidence** to conclude that the new landing page changed the true conversion rate.
* **Action:** The current design does not show a statistically significant uplift; consider testing additional variations or increasing sample size.

#### Derivation of the Standard Error (SE)
In the 1-proportion z-test, the standard error (SE) formula is:

$$SE = \sqrt{\frac{p_0(1-p_0)}{n}}$$

a. **Building Block: The Bernoulli Trial**  
Every single data point in a proportion test is a binary outcome (Success/Failure, Click/No-Click). This is modeled as a **Bernoulli random variable**, let's call it $X$.
* If Success ($X=1$): Probability is $p_0$.
* If Failure ($X=0$): Probability is $1 - p_0$.

The **Variance** of a single Bernoulli trial is a known property:

$$Var(X) = p_0(1 - p_0)$$

b. **The Sample Proportion**   
When we run a test, we don't look at just one data-point, we calculate the **sample proportion** ($\hat{p}$) from a sample of size $n$.

$$\hat{p} = \frac{\sum X_i}{n}$$

(This is essentially the average of all the 1s and 0s).

c. **Variance of the Sample Proportion**  
To find the variance of this new variable $\hat{p}$, we apply the rules of variance to the formula above. Since the observations are independent, their variances add up.

$$Var(\hat{p}) = Var\left( \frac{\sum X_i}{n} \right)$$

When you pull a constant (like $1/n$) out of a variance calculation, it gets squared:

$$Var(\hat{p}) = \frac{1}{n^2} \sum Var(X_i)$$

Since we have $n$ identical trials, the sum of their variances is just $n \times Var(X)$:

$$Var(\hat{p}) = \frac{1}{n^2} \cdot [n \cdot p_0(1-p_0)]$$

Simplify the $n$ terms:

$$Var(\hat{p}) = \frac{p_0(1-p_0)}{n}$$

d. **From Variance to Standard Error**  
The Standard Error is simply the **standard deviation of the sampling distribution**. To get standard deviation from variance, we take the square root.

$$SE = \sqrt{Var(\hat{p})} = \sqrt{\frac{p_0(1-p_0)}{n}}$$

e. **Why use $p_0$ instead of $\hat{p}$?**  
In a 1-sample hypothesis test, we calculate the standard error assuming the **Null Hypothesis is true**. The Null Hypothesis claims the true population rate is $p_0$. Therefore, the sampling distribution must be centered and spread according to that hypothesized value ($p_0$), not the data we just collected ($\hat{p}$).

### 2-Proportion Z-Test Example
* **Formulation:** Used when comparing **two independent proportions** (e.g., conversion rate A vs. conversion rate B).

    $$Z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1 - \hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}$$

    Where:
    * ($\hat{p}_1$, $\hat{p}_2$): sample proportions
    * ($n_1$, $n_2$): sample sizes
    * ($\hat{p}$): **pooled** proportion
* **Scenario:** A company wants to compare two email subject lines (A vs. B) to see which produces a higher open rate.
  * Version A:
    ($n_1 = 1000$) emails sent, ($x_1 = 180$) opens $\rightarrow$ ($\hat{p}_1 = 0.18$)
  * Version B:
    ($n_2 = 1200$) emails sent, ($x_2 = 264$) opens $\rightarrow$ ($\hat{p}_2 = 0.22$)
* **Hypothesis:** $H_0 : p_1 = p_2$; $H_a : p_1 \neq p_2$. This is a **two-tailed** test.
* **Critical Value:** At ($\alpha = 0.05$):

    $$Z_{\text{crit}} = \pm 1.96$$

    Reject $H_0$ if ($\lvert Z \rvert > 1.96$).
* **Calculation of the Pooled Variance:** Under the null hypothesis ($p_1 = p_2 = p$), the best estimate of the common proportion is the **pooled estimator**:

    * Combine successes: $x = x_1 + x_2 = 180 + 264 = 444$
    * Combine sample sizes: $n = n_1 + n_2 = 1000 + 1200 = 2200$
    * Compute pooled proportion:

    $$\hat{p} = \frac{x}{n} = \frac{444}{2200} = 0.2018$$

    * Use pooled variance in Z-statistic:

    $$\text{Var}(\hat{p}_1 - \hat{p}_2)
    = \hat{p}(1-\hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)$$

    This variance estimate is used because under ($H_0$), both samples share a common true proportion.
* **Calculation**

    $$Z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1 - \hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}
     = \frac{0.18 - 0.22}{\sqrt{0.2018 (1 - 0.2018)\left(\frac{1}{1000} + \frac{1}{1200}\right)}}
     \approx -2.33$$

* **Decision Rule:** Since ($Z = -2.33$) and ($\lvert -2.33 \rvert > 1.96$), we **reject** the null hypothesis.
* **Conclusion:** There is statistically significant evidence at the 5% level that **email subject line B produces a different (in this case, higher) open rate than A**.
* **Action:** Subject line B shows a significant improvement and should be used or further optimized.

#### Derivation of the Pooled Standard Error (SE)

The **Pooled Standard Error** is used specifically when calculating the **Z-statistic** for hypothesis testing. The original Z-statistic equation is given below.

$$Z = \frac{\hat{p}_1 - \hat{p}_2}
{\sqrt{\frac{\hat{p}_1 (1 - \hat{p}_1)}{n_1} + \frac{\hat{p}_2 (1 - \hat{p}_2)}{n_2}}}$$

**Step 1: The Null Hypothesis Assumption**
The Z-test starts with the Null Hypothesis ($H_0$) that there is **no difference** between the groups.

$$H_0: p_1 = p_2 = p$$

If $H_0$ is true, the two samples actually come from the exact same population. Therefore, it is more accurate to combine (pool) them to estimate a single common proportion, $\hat{p}_{pool}$.

**Step 2: Calculate the Pooled Proportion**

$$\hat{p}_{pool} = \frac{\text{Total Successes}}{\text{Total Observations}} = \frac{x_1 + x_2}{n_1 + n_2}$$

**Step 3: Substitute into the Variance Equation**  
Because we assume $p_1$ and $p_2$ are the same: ($$p_1 = p_2 = \hat{p}_{pool}$$), we substitute $\hat{p}_{pool}$ into the 'un-pooled' variance equation below.

$$Var_{un-pooled} = \frac{\hat{p}_1 (1 - \hat{p}_1)}{n_1} + \frac{\hat{p}_2 (1 - \hat{p}_2)}{n_2}$$

$$Var_{pooled} = \frac{\hat{p}_{pool}(1-\hat{p}_{pool})}{n_1} + \frac{\hat{p}_{pool}(1-\hat{p}_{pool})}{n_2}$$

Factor out the common term $$\hat{p}_{pool}(1-\hat{p}_{pool})$$:

$$Var_{pooled} = \hat{p}_{pool}(1-\hat{p}_{pool}) \left( \frac{1}{n_1} + \frac{1}{n_2} \right)$$

**Step 4: Calculate Standard Error**
The Standard Error is simply the square root of the variance.

$$SE_{pooled} = \sqrt{ \hat{p}_{pool}(1-\hat{p}_{pool}) \left( \frac{1}{n_1} + \frac{1}{n_2} \right) }$$

## 6. Hypothesis Testing via T-Test (Means)

### 1-Sample T-Test Example
* **Formulation:** Used when comparing a sample mean ($\bar{x}$) to a known population mean ($\mu_0$), **but the population standard deviation is unknown**.
    We estimate the standard deviation using the **sample standard deviation (s)**.

    $$t = \frac{\bar{x} - \mu_0}{\frac{s}{\sqrt{n}}}$$

    * ($\bar{x}$): Sample mean
    * ($\mu_0$): Hypothesized population mean
    * ($s$): Sample standard deviation
    * ($n$): Sample size
* **Scenario:**
    A nutritionist claims that a protein bar contains **20g of protein on average**.  
    You suspect the actual mean is different.  
    Since the population standard deviation is unknown, you collect a sample of **25 bars** and measure their protein content.  

    Suppose your sample results are: ($\bar{x}$ = 19.2)g, ($s$ = 2.5)g
* **Hypothesis:** $H_0 : \mu = 20$; $H_a : \mu \neq 20$. This is a **two-tailed test**.
* **Degrees of Freedom:** $df = n - 1 = 25 - 1 = 24$
* **Critical Value:** At ($\alpha = 0.05$) with ($df = 24$), the two-sided critical value is:

    $$t_{\text{crit}} = \pm 2.064$$

    Reject $H_0$ if ($\lvert t \rvert > 2.064$).
* **Calculation:**
    $$t = \frac{19.2 - 20}{2.5 / \sqrt{25}}
    = \frac{-0.8}{0.5}
    = -1.6$$

* **Decision Rule:** Since ($\lvert t \rvert = 1.6$) and (1.6 < 2.064), we **fail to reject** the null hypothesis.
* **Conclusion:** There is **not enough evidence** to conclude that the true mean protein content differs from 20g.
* **Action:** No immediate concern; more data or a larger sample may be needed for a definitive conclusion.

### 2-Sample T-Test Example
* **Formulation:** Used when comparing **two independent sample means** when **population standard deviations are unknown**. The test statistic is:

  $$t = \frac{\bar{x}_1 - \bar{x}_2}{
  \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

  * ($\bar{x}_1$, $\bar{x}_2$): Sample means
  * ($s_1$, $s_2$): Sample standard deviations
  * ($n_1$, $n_2$): Sample sizes

  This is the **Welch’s t-test**, which does not assume equal variances.
* **Scenario:**
  A school wants to know whether **two different teaching methods** lead to different average test scores.

  * Method A: ($n_1 = 30$), ($\bar{x}_1 = 78$), ($s_1 = 10$)
  * Method B: ($n_2 = 28$), ($\bar{x}_2 = 72$), ($s_2 = 12$)
* **Hypothesis:** $H_0: \mu_1 = \mu_2$; $H_a: \mu_1 \neq \mu_2$. This is a **two-tailed test**.
* **Degrees of Freedom:** Welch’s formula:

    $$df =
    \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}
    {\frac{(s_1^2/n_1)^2}{n_1 - 1} +
    \frac{(s_2^2/n_2)^2}{n_2 - 1}}$$

    Compute:

    $$\frac{s_1^2}{n_1} = \frac{100}{30} = 3.333$$

    $$\frac{s_2^2}{n_2} = \frac{144}{28} = 5.143$$

    $$df \approx 53$$

* **Critical Value:** At ($\alpha = 0.05$), two-tailed:

    $$t_{\text{crit}} \approx \pm 2.006$$ 
    
    * using df $\approx$ 53
* **Calculation:**

$$t = \frac{78 - 72}{\sqrt{3.333 + 5.143}}
= \frac{6}{\sqrt{8.476}}
= \frac{6}{2.912}
\approx 2.06$$

* **Decision Rule:** Since ($t$ = 2.06) and (2.06 > 2.006), we **reject** the null hypothesis.
* **Conclusion:** There is statistically significant evidence at the 5% level that the two teaching methods lead to **different average test scores**.
* **Action:** The school should consider adopting Method A more broadly or conducting further controlled experiments.

### Proportion T-Test Example
**Why it doesn't exist:**
There is no "proportion t-test" because proportion data is based on the Binomial distribution. The variance of a proportion is directly tied to the mean ($Variance = p(1-p)$). Unlike continuous data (where you need a T-test to account for estimating *both* the mean and the unknown variance separately), with proportions, once you estimate the mean ($p$), you have automatically estimated the variance. As $n$ increases, the Binomial distribution is approximated by the Normal distribution, making the Z-test the correct tool.
---
title: "Statistics: P-values & Confidence Intervals"
date: 2025-10-28 12:00:00 +0530
categories: [Statistics]
tags: [Statistics]
math: true
---

In the realm of statistics, p-values and confidence intervals are fundamental tools that help us make sense of data and draw conclusions, particularly when dealing with samples and trying to understand a larger population. Let's break down these concepts with some illustrative examples.

## Understanding P-Values

**What is a p-value?**

The p-value, or probability value, is a measure that helps you determine the statistical significance of your results when conducting a hypothesis test. In essence, it tells you the probability of observing your data (or something more extreme) if the null hypothesis were true. The null hypothesis ($H_0$) generally represents the "status quo" or a statement of no effect or no difference.

**How to interpret a p-value?**

* **Low p-value (typically ≤ 0.05):** This indicates that your observed data is unlikely to have occurred by random chance if the null hypothesis were true. Therefore, you *reject* the null hypothesis in favor of the alternative hypothesis ($H_1$), which suggests there is a real effect or difference.
* **High p-value (> 0.05):** This means that your observed data is quite likely to have occurred even if the null hypothesis were true. Therefore, you *fail to reject* the null hypothesis. It's important to note that this doesn't *prove* the null hypothesis is true, only that you don't have enough evidence to reject it.

The 0.05 threshold is a common convention, but it can be adjusted depending on the field and the desired level of certainty.

**Example of a p-value:**

Imagine a pharmaceutical company develops a new drug to lower blood pressure. They want to test if this new drug is more effective than an existing standard drug.

* **Null Hypothesis ($H_0$):** There is no difference in blood pressure reduction between the new drug and the standard drug.
* **Alternative Hypothesis ($H_1$):** The new drug leads to a greater reduction in blood pressure than the standard drug.

They conduct a clinical trial with two groups of patients. After the trial, they analyze the data and find that the group taking the new drug had an average blood pressure reduction of 15 mmHg, while the group on the standard drug had an average reduction of 10 mmHg.

They perform a statistical test, and the resulting **p-value is 0.03**.

**Interpretation:** Since the p-value (0.03) is less than the conventional threshold (0.05), they would **reject the null hypothesis**. This suggests that it's unlikely they would have observed such a difference in blood pressure reduction (or an even larger one) if the new drug was actually no better than the standard drug. Therefore, they have statistical evidence to conclude that the new drug is indeed more effective.

## Understanding Confidence Intervals

**What is a confidence interval?**

A confidence interval (CI) provides a range of plausible values for an unknown population parameter (like the mean or proportion) based on your sample data. It gives you an idea of how precise your estimate is and how much uncertainty there is.

A confidence interval is typically expressed as: **Point Estimate ± Margin of Error**.

The "confidence level" (commonly 95%, but can be 90%, 99%, etc.) indicates how confident you can be that the true population parameter lies within this interval.

**How to interpret a confidence interval?**

A 95% confidence interval means that if you were to repeat your study or experiment many times, 95% of the calculated confidence intervals would contain the true population parameter. It **doesn't** mean there's a 95% chance that *your specific* interval contains the true value (it either does or it doesn't).

**Example of a confidence interval:**

Let's say a market research firm wants to estimate the average amount of money a household in a particular city spends on groceries per week. They survey 200 households and find the average weekly spending is $250.

They calculate a **95% confidence interval** for the average weekly grocery spending to be **$235** to **$265**.

**Interpretation:** They are 95% confident that the *true* average weekly grocery spending for *all* households in that city falls somewhere between **$235** and **$265**.

This interval gives them a range of plausible values. A narrower interval would indicate a more precise estimate, while a wider interval would suggest more uncertainty.

## The Relationship Between P-Values and Confidence Intervals

P-values and confidence intervals are closely related and often provide complementary information:

* **Hypothesis Testing with CIs:** If you are testing a hypothesis about a specific value, you can often use a confidence interval. For instance, in our drug example, if the 95% confidence interval for the *difference* in blood pressure reduction between the new and standard drug was, say, 2 mmHg to 8 mmHg, since this interval does not include zero (which would represent no difference), it would lead to the same conclusion as the p-value being less than 0.05 – we would reject the null hypothesis.
* **Information Provided:** A p-value gives you a single number representing the strength of evidence against the null hypothesis. A confidence interval, on the other hand, provides a range of plausible values for the effect size, which can be more informative in many practical situations.

In essence, both p-values and confidence intervals are crucial tools in statistics that allow researchers and data analysts to draw meaningful conclusions from data while acknowledging and quantifying the inherent uncertainty involved in working with samples.

### Step-by-Step Procedure to Calculate the 95% Confidence Interval
Below, we’ll outline a step-by-step procedure to calculate the 95% confidence interval boundaries (\\$235 and \\$265) for the given example, assuming the sample mean is \\$250, population standard deviation is 108.24 and the sample size is 200 households.

#### Step 1: Identify the Given Information
- **Sample Mean $(\bar{x})$**: $250 (average weekly grocery spending from the survey).
- **Population SD $(\sigma)$**: 108.24
- **Sample Size $(n)$**: 200 households.
- **Confidence Level $(CI)$**: 95% (implies a significance level $\alpha = 0.05$, so the z-score for a two-tailed test is based on 95% of the area under the normal curve).
- **Goal**: Calculate the confidence interval boundaries (\\$235, \\$265).
- **Assumptions**:
  - The sample size `n = 200` is large, so the Central Limit Theorem applies, allowing us to use the normal distribution (z-distribution) rather than the t-distribution.
  - The data is approximately normally distributed or the sample is large enough for the sample mean to be normally distributed.

#### Step 2: Understand the Confidence Interval Formula
For a 95% confidence interval for the population mean $\mu$ when the sample size is large $n \geq 30$, we use:

$$\text{CI} = \bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

Where:
- $\bar{x}$: Sample mean ($250).
- $z_{\alpha/2}$: Critical z-score for a 95% confidence level (two-tailed, so $\alpha/2 = 0.025$, and $z_{0.025} \approx 1.96$).
- \($\sigma$\): Population mean (108.24).
- \(n\): Sample size (200).
- $\frac{\sigma}{\sqrt{n}}$: Standard error (SE), which measures the variability of the sample mean.

The interval is:

$$\text{Lower bound} = \bar{x} - z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}, \quad \text{Upper bound} = \bar{x} + z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

#### Step 3: Determine the Critical Z-Score
For a 95% confidence interval:
- The confidence level is 95%, so $\alpha$ = 1 - 0.95 = 0.05.
- For a two-tailed test, we need the z-score that leaves 2.5% in each tail, so $\alpha/2$ = 0.025.
- From standard normal tables, the z-score for $P(Z < z) = 0.975$ is approximately $z = 1.96$.

Thus, $z_{\alpha/2} = 1.96$.

#### Step 4: Calculate the Standard Error
The standard error (SE) is:

$$\text{SE} = \frac{\sigma}{\sqrt{n}} = \frac{108.24}{\sqrt{200}} \approx \frac{108.24}{14.142} \approx 7.653$$

#### Step 5: Compute the Confidence Interval
Using the formula:

$$\text{CI} = \bar{x} \pm z_{\alpha/2} \cdot \text{SE}$$

$$\text{CI} = 250 \pm 1.96 \cdot 7.653$$

Calculate the margin of error:

$$1.96 \cdot 7.653 \approx 15$$

So:

$$\text{Lower bound} = 250 - 15 = 235$$

$$\text{Upper bound} = 250 + 15 = 265$$

The 95% confidence interval is \\$235 to \\$265, matching the given example.

#### Step 6: Verify Assumptions
- **Large Sample Size**: $n = 200$ is large, so the Central Limit Theorem ensures the sample mean is approximately normally distributed, justifying the use of the z-distribution.
- **Normality**: If the data is heavily skewed, a t-distribution might be considered, but with $n = 200$, the z-distribution is robust.

#### Summary of the Procedure
1. **Collect Sample Data**: Sample mean $\bar{x} = 250$, Population SD $\sigma = 108.24$, sample size $(n = 200)$.
2. **Determine Confidence Level**: 95% implies $z_{\alpha/2} = 1.96$.
3. **Calculate Standard Error**: $\text{SE} = \frac{\sigma}{\sqrt{n}} \approx 7.653$.
4. **Compute Interval**: $\bar{x} \pm z_{\alpha/2} \cdot \text{SE} = 250 \pm 15 = (235, 265)$.

**Common Misconception**: The statement “95% confident” does not mean there’s a 95% chance the true mean is between \\$235 and \\$265. Instead, it means the method used to construct the interval is 95% reliable. The true mean is a fixed (unknown) value, not a random variable with a probability distribution.

## An Example to Illustrate the Concept

Let’s extend the grocery spending scenario to make the concept concrete, using a simplified simulation to mimic the sampling process. Suppose the true (unknown) average weekly grocery spending for all households in the city is \\$260 (we’ll use this for illustration, but in reality, we wouldn’t know it).

### Scenario Setup
- **Population**: All households in the city.
- **True Mean $(\mu)$**: \\$260 (unknown to the researchers).
- **True Standard Deviation $(\sigma)$**: Assume \\$100 (based on the previous calculation’s population standard deviation of ~\\$108.24, which is reasonable for grocery spending).
- **Sample**: The market research firm surveys 200 households, finding a sample mean $(\bar{x})$ of \\$250.
- **Confidence Interval**: They calculate a 95% CI of \\$235 to \\$265.

### Simulation of Repeated Sampling
Imagine the firm conducts the survey multiple times, each with a new random sample of 200 households. Due to the Central Limit Theorem (since $n = 200$ is large), the sample mean $(\bar{x})$ follows an approximately normal distribution with:
- Mean = $(\mu = 260)$.
- Standard error = $(\frac{\sigma}{\sqrt{n}} = \frac{100}{\sqrt{200}} \approx 7.07)$.

For each sample, they calculate a 95% CI using:
$$\text{CI} = \bar{x} \pm 1.96 \cdot \frac{\sigma}{\sqrt{n}} = \bar{x} \pm 1.96 \cdot 7.07 \approx \bar{x} \pm 13.86$$

Here’s what might happen in five hypothetical samples:
1. **Sample 1**: $(\bar{x} = 250)$, CI = $(250 \pm 13.86 = (236.14, 263.86))$. Contains true mean ($260).
2. **Sample 2**: $(\bar{x} = 255)$, CI = $(255 \pm 13.86 = (241.14, 268.86))$. Contains true mean ($260).
3. **Sample 3**: $(\bar{x} = 248)$, CI = $(248 \pm 13.86 = (234.14, 261.86))$. Contains true mean ($260).
4. **Sample 4**: $(\bar{x} = 270)$, CI = $(270 \pm 13.86 = (256.14, 283.86))$. Contains true mean ($260).
5. **Sample 5**: $(\bar{x} = 280)$, CI = $(280 \pm 13.86 = (266.14, 293.86))$. Does **not** contain true mean ($260).

If we repeat this process 100 times, about 95 of the CIs will include the true mean \\$260. The given CI \\$235 to \\$265 from your sample  $(\bar{x} = 250)$ is one such interval, and we’re 95% confident it’s one of the “successful” intervals that captures the true mean.

#### Visualizing the Concept
Imagine a dartboard where the bullseye is the true mean (\\$260). Each sample’s CI is like a dart throw—most (95%) hit the bullseye (contain \\$260), but some miss. The \\$235 to \\$265 interval is one throw, and we’re 95% confident it’s a hit.
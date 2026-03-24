---
title: "Epilogue: The Grand Unifying Theory of A/B Testing (Enter the GLM)"
date: 2026-03-23 21:00:00 +0530
categories: [Statistics]
tags: [Statistics]
math: true
---

Welcome to the epilogue of our six-part series on experimentation and A/B testing! Over the past few months, we’ve covered a massive amount of ground. We started with the foundational statistics of p-values and z-tests, navigated the tricky waters of Sample Ratio Mismatch (SRM), ventured into the complex realm of causal inference (DiD, PSM, IV), and finally arrived at the cutting-edge world of Multi-Armed and Contextual Bandits.

Looking back, it might feel like we’ve handed you a massive toolbox containing dozens of completely different, highly specialized tools. 

But recently, a perceptive reader on LinkedIn left a fantastic comment that inspired this final post. He pointed out the missing link. What if we told you that almost everything we’ve learned in this series is actually just one tool wearing different disguises? 

That single, underlying engine is the **Generalized Linear Model (GLM)**.

Often taught in a completely separate mental silo as a "predictive machine learning algorithm", the GLM is actually the Rosetta Stone of experimentation. Let’s pull back the curtain and see how the entire A/B testing landscape maps perfectly to GLMs. 

BTW, feel free to go through the [section](https://techscribr.github.io/posts/linear-regression/#10-how-are-generalized-linear-models-glms-different-from-vanilla-linear-regression-explain-with-an-example) on GLM in our Linear Regression article as a quick refresher before proceding through this article.

### 1. The Basics: T-Tests and ANOVA are just Linear Regression
In [Part 2](https://techscribr.github.io/posts/z-and-t-statistic/) and [Part 3](https://techscribr.github.io/posts/ab-testing/), we learned about 2-sample t-tests and ANOVA for comparing continuous metrics like avg-test-score. 

#### Re-interpretations of T-test and ANOVA
Mathematically, running a 2-sample t-test is identical to running a standard Linear Regression (a GLM with a Gaussian family and Identity link). If we assign our Control group as $T = 0$ and our Treatment group as $T = 1$, the equation looks like this:

$$Y_i = \beta_0 + \beta_1 T_i + \epsilon_i$$

* **$\beta_0$** is the average of our Control group.
* **$\beta_1$** is the exact absolute lift caused by the Treatment.
* The p-value the model spits out for $\beta_1$ is *numerically ideantical to the classic t-test p-value* under these assumptions. 

ANOVA is simply this exact same equation expanded to include $T_2, T_3$, etc.

#### ANOVA with 3 Variants
Imagine you are running an A/B/C test. You have a **Control**, **Treatment 1**, and **Treatment 2**. 

To feed this into a linear regression model, the math requires us to use "dummy encoding" (1s and 0s). However, to avoid a mathematical error called the *dummy variable trap* (perfect multicollinearity), we don't create a variable for all three groups. We drop one group to serve as our **Reference Group** (the baseline). Naturally, we choose the Control group.

Our model creates two variables:
* $T_{1i} = 1$ if the user is in Treatment 1, otherwise $0$.
* $T_{2i} = 1$ if the user is in Treatment 2, otherwise $0$.

The ANOVA linear regression equation becomes:

$$Y_i = \beta_0 + \beta_1 T_{1i} + \beta_2 T_{2i} + \epsilon_i$$

##### Interpreting the Coefficients ($\beta$)
Because of how we set up the 1s and 0s, the coefficients map perfectly to our variant means and the absolute lift. Let's look at the math for each specific group:

**1. The Control Group (The Baseline)**
For a user in the Control group, both $T_{1i}$ and $T_{2i}$ are $0$.

$$Y_i = \beta_0 + \beta_1(0) + \beta_2(0)$$

$$Y_i = \beta_0$$

* **Interpretation of $\beta_0$:** The Intercept ($\beta_0$) is simply the **average score of the Control group**.

**2. Treatment 1**
For a user in Treatment 1, $T_{1i} = 1$ and $T_{2i} = 0$.

$$Y_i = \beta_0 + \beta_1(1) + \beta_2(0)$$

$$Y_i = \beta_0 + \beta_1$$

* **Interpretation of $\beta_1$:** Since $\beta_0$ is the Control average, $\beta_1$ represents the **exact absolute lift** caused by Treatment 1. (i.e., $\text{Treatment 1 Mean} - \text{Control Mean}$).

**3. Treatment 2**
For a user in Treatment 2, $T_{1i} = 0$ and $T_{2i} = 1$.

$$Y_i = \beta_0 + \beta_1(0) + \beta_2(1)$$

$$Y_i = \beta_0 + \beta_2$$

* **Interpretation of $\beta_2$:** Similarly, $\beta_2$ is the **exact absolute lift** caused by Treatment 2. (i.e., $\text{Treatment 2 Mean} - \text{Control Mean}$).

##### The P-Values: Global vs. Individual
When you run a traditional ANOVA, it spits out a single **Global p-value** (from an F-test) that tells you: *"Is there a significant difference anywhere between these groups?"* When you run this as a Linear Regression, you get a richer output:
1.  **The F-Statistic P-Value:** The regression summary will have an overall F-statistic at the bottom. This is the *exact same p-value* you get from the traditional ANOVA. It tests if *any* of the $\beta$ coefficients ($\beta_1$ or $\beta_2$) are different from zero.
2.  **The Coefficient P-Values:** The model also gives individual p-values for $\beta_1$ and $\beta_2$. These act as built-in post-hoc tests! The p-value for $\beta_1$ tells you if Treatment 1 is significantly different from the Control, and the p-value for $\beta_2$ tells you if Treatment 2 is significantly different from the Control. 

*(Note: You still need to be mindful of multiple comparison corrections like Bonferroni when looking at these individual p-values, but the math is all there in one simple GLM output!)*

#### 2-Proportion Z-Tests
What about our 2-proportion Z-tests for Conversion Rates? That is simply a GLM using a Binomial family and a Logit link: otherwise known as **Logistic Regression**. 

While both approaches test the same underlying hypothesis, the exact p-values may differ slightly in practice due to different estimation methods (e.g., Wald test vs pooled variance approximation), especially in small samples.

Our Chi-Square tests for count data? That is a GLM using a Poisson family and a Log link.

### 2. Variance Reduction: Supercharging Sample Sizes
In [Part 4](https://techscribr.github.io/posts/sample-size-for-ab-testing/), we talked about the pain of calculating sample sizes. What if we need to detect a tiny effect, but we simply don't have enough daily traffic? 

By framing our A/B test as a GLM, we unlock a superpower: **Covariate Adjustment** (often referred to in the industry as CUPED - Controlled-experiment Using Pre-Existing Data). 

Instead of just looking at the treatment indicator ($T_i$), we can add pre-experiment knowledge about our users into the model:

$$Y_i = \beta_0 + \beta_1 T_i + \beta_2 X_{i, \text{history}} + \epsilon_i$$

**What exactly is $X_{i, \text{history}}$**? This is our **Covariate** - a measurable piece of data about the user that was recorded *before* the experiment started. It must be a variable that strongly predicts their future behavior but is completely untouched by the A/B test. In e-commerce, the absolute best covariate is the user's historical spend. 

**Let's look at a practical toy example to see the math in action:**
Imagine we are testing a new checkout flow designed to increase Average Order Value (AOV). We have two distinct users in our Treatment group:
* **Shopper A (The Whale):** Their historical average order value is $500. So, for Shopper A, **$X_{i, \text{history}} = 500$**.
* **Shopper B (The Bargain Hunter):** Their historical average order value is $20. So, for Shopper B, **$X_{i, \text{history}} = 20$**.

Let's assume our new checkout flow is genuinely successful and adds exactly \\$10 to everyone's basket. During the test, Shopper A spends \\$510, and Shopper B spends \\$30. 

#### The Math: Why the \\$480 Gap Drowns the \\$10 Signal
If we run a standard t-test (which ignores the $X$ covariate), the statistical engine only sees the final numbers: 510 and 30. 

To determine if our $10 lift is statistically significant, the t-test calculates the **Standard Error (SE)** of our treatment effect. The formula for standard error relies entirely on the overall variance ($\sigma^2$) of the raw metric $Y$:

$$SE(\text{Lift}) \approx \sqrt{\frac{\sigma^2}{N}}$$

Because our raw data bounces wildly between 30 and 510, the overall variance ($\sigma^2$) is absolutely massive. When $\sigma^2$ is a massive number, our Standard Error becomes a massive number. 

To get statistical significance, our observed lift (\\$10) needs to be roughly twice as large as the Standard Error. If the natural \\$480 gap between our Whale and our Bargain Hunter pumps the Standard Error up to \\$50, our tiny \\$10 signal is completely drowned out. The math assumes the \\$10 difference is just random noise. The only way to shrink the SE in a standard t-test is to blindly increase $N$ (the sample size) to thousands or millions of users.

#### The GLM Solution: Conditioning on Predictable Variation
Now, we use our GLM and include $X_{i, \text{history}}$ as a covariate. 

The key idea is not that the model simply "ignores" raw variance, but that it **conditions on predictable variation**. It mathematically separates the data into two buckets:
1.  **Predictable variation** (explained by $X$)
2.  **Unpredictable variation** (residual noise)

Formally, the CUPED mechanism performs an adjustment of the form:

$$Y_i' = Y_i - \theta (X_i - E[X])$$

where:

$$\theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$

This transformation removes the component of $Y$ that is linearly predictable from $X$, leaving behind a purified, lower-variance signal ($Y_i'$).

You can interpret this directly through the lens of our GLM regression: the model explains a massive portion of the variance in $Y$ using $X$, leaving only a much smaller **residual variance**:

$$\sigma_{\text{residual}}^2 = \text{Var}(Y \mid X)$$

Because historical spend ($X$) is highly correlated with current spend ($Y$), most of the wild variation (like that $480 gap between our Whale and Bargain Hunter) is explained away *before* we evaluate the treatment effect.

So, rather than “changing the formula,” we are simply reducing the unexplained variance that feeds into the exact same statistical machinery. Our new Standard Error plummets because it relies on this reduced residual variance:

$$SE(\text{Lift})_{\text{adjusted}} \approx \sqrt{\frac{\sigma_{\text{residual}}^2}{N}}$$

Because the numerator is now tiny, our \\$10 treatment signal becomes crystal clear. We can confidently declare our \\$10 lift as statistically significant with a fraction of the original sample size required!

### 3. Causal Inference: GLMs Under the Hood
In [Part 5](https://techscribr.github.io/posts/ab-testing-did-psm-iv/), we tackled situations where we couldn't perfectly randomize our traffic. GLMs were secretly doing all the heavy lifting:
* **Propensity Score Matching (PSM):** How did we calculate the "statistical twin" probability score to fix selection bias? We ran a Logistic Regression (GLM) where the target was the treatment assignment and the features were the confounders.
* **Difference-in-Differences (DiD):** We calculated the true causal uplift over time by running a linear model with an interaction term ($\beta_3(\text{Group}_i \times \text{Time}_t)$). If our outcome was a count metric, standard DiD would break, and we’d simply swap our engine to a Poisson GLM.

The GLM provides a flexible estimation framework, but **causal validity comes from assumptions - not the model itself**.

### 4. Contextual Bandits: The Evolution of GLMs
Finally, in [Part 6](https://techscribr.github.io/posts/ab-testing-multi-arm-bandits/), we explored Contextual Bandits. We specifically looked at **LinUCB** (Linear Upper Confidence Bound). As the name implies, LinUCB assumes the expected reward is a linear combination of the user's context features. It uses a form of regularized linear regression (similar to **Ridge Regression**) to estimate rewards, but crucially augments this with an uncertainty term to drive exploration.

But what if our bandit is trying to optimize Ad Clicks (a binary 0 or 1)? Linear models are bad at predicting bounded probabilities. The industry solution is **GLM-UCB** (Generalized Linear Bandits). We wrap our bandit algorithm in a logistic link function, allowing it to learn a Logistic Regression model on the fly while balancing exploration and exploitation. In practice, these models are updated iteratively and often rely on approximations, making them more complex than a standard "fit-once" GLM, but the underlying statistical structure remains the same.

### The Final Takeaway
When we study A/B testing and experimentation, we are usually taught Frequentist statistics. When we study predictive modeling, we are taught Machine Learning. 

Understanding that Generalized Linear Models bridge this gap is a massive leap forward in any Data Scientist's career. When we realize that a t-test, a variance reduction technique, and a Contextual Bandit are all built on the same underlying statistical foundation, we stop memorizing isolated tools and start thinking in terms of a unified modeling framework.

Thank you for joining us on this six-part journey through the world of experimentation! 

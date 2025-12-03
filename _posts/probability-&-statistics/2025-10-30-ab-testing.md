---
title: "Stats Primer: Connecting the dots ... A/B Tests"
date: 2025-10-30 15:00:00 +0530
categories: [Statistics]
tags: [Statistics]
math: true
---

In the previous article, we’ve looked at hypothesis tests as tools to decide whether a difference is real or just random noise. But in product experimentation, we care about something deeper: **Did the new feature *cause* the improvement?**

Surprisingly, the foundation of modern causal inference in product analytics comes straight from the z-tests and t-tests we just explored. A/B testing is simply hypothesis testing embedded into a randomized experimental design.

In this article, we’ll connect these dots and explore how the statistical tests we've mastered become the core engines powering parametric and non-parametric A/B testing.

## 1. What is an A/B Test?
An A/B test is a method used to compare two versions of a variable (like a webpage, email, or feature) to determine which one performs better. In a typical setup, you test a new version (Version B) against a current version (Version A) to see if the new version produces a statistically significant improvement.

**The Motivation:**
Making business decisions based on small differences without rigorous testing is essentially acting on random noise. The goal is to move away from guessing and towards data-driven decisions. By isolating variables and testing them, you ensure that observed improvements are real and not just driven by randomness or novelty effects.

### 1.1 The Holy Grail: A/B Testing and Causal Inference
A/B testing is fundamentally an exercise in **Causal Inference**. In data science, we often struggle to distinguish correlation from causation. If users who use a "Wishlist" feature spend more money, did the feature cause the spending, or do high-spending users simply like using wishlists?

A/B testing (Randomized Controlled Trials) resolves this through **Randomization**.

By randomly assigning users to Group A (Control) or Group B (Treatment), you ensure that both groups are statistically identical in every observable and unobservable characteristic (age, income, motivation, etc.) *before* the intervention. Because the only difference between the two universes is the feature you changed, any statistical difference in the outcome can be causally attributed to your change.

In mathematical terms, randomization allows us to equate the **Simple Difference in Means** with the **Average Treatment Effect (ATE)** without needing complex statistical adjustments.

$$
\text{ATE} = \mathbb{E}[Y \mid T=1] - \mathbb{E}[Y \mid T=0]
$$

### 1.2 When You Can't Randomize: A/B Testing vs. Quasi-Experiments

Sometimes, A/B testing is impossible due to ethical concerns, technical limitations, or the need to analyze historical data. In these "Observational Studies," we cannot assume the treated and untreated groups are identical.

To infer causation without randomization, we rely on **Quasi-Experimental methods**. Unlike A/B testing, these methods require strict (and often unverifiable) assumptions to simulate a randomized experiment.

* **Difference-in-Differences (DiD):**
    This method compares the *change* in outcomes over time between a treated group and a control group. Instead of comparing the groups directly (which might be different), you compare their trend lines.
    * *Example:* Comparing traffic in a city that launched a marketing campaign vs. a similar city that didn't, before and after the launch.
    * *The Catch:* It relies on the "Parallel Trends Assumption"—that without the treatment, both groups would have moved in the same way.
* **Regression Discontinuity Design (RDD):**
    This is used when treatment is assigned based on a strict cutoff (e.g., a scholarship given to students with a GPA above 3.5). RDD compares students just above the cutoff (3.51) to those just below (3.49).
    * *The Logic:* Students right at the margin are likely very similar, so the cutoff acts as a "local" A/B test.
    * *The Catch:* Results are only valid for users near the cutoff, not the entire population.
* **Instrumental Variables (IV):**
    This technique uses a third variable (the Instrument) that affects the treatment but has no direct effect on the outcome. It acts as a "natural experiment" that forces some variation in the treatment assignment.
    * *Example:* Using a "winning a lottery" as an instrument to study the effect of wealth on health. The lottery determines wealth randomly, breaking the link between wealth and unobserved factors like ambition.    

## 2. Causal Inference Landscape
We can find a concise (but not exhaustive) representation of different well-known causal inference frameworks below.

        Causal Inference
        │
        ├── 1. Randomized Experiments (True Experiments)
        │       └── A/B Testing (Online Controlled Experiments)
        │             │
        │             ├── Parametric A/B Tests
        │             │       ├── 1 and 2-Sample Z-Tests
        │             │       ├── 1 and 2-Proportion Z-Tests
        │             │       ├── 1-Sample t-Test
        │             │       ├── 2-Sample t-Test (Welch)
        │             │       └── ANOVA / Chi-Square tests
        │             │
        │             └── Non-Parametric A/B Tests
        │                     ├── Permutation Test
        │                     ├── Bootstrap Test (resampling-based)
        │                     └── Kolmogorov–Smirnov (KS) Test
        │
        └── 2. Quasi-Experiments (When Randomization Is Not Possible)
                ├── Difference-in-Differences (DiD)
                │      └── Uses time variation + treated/untreated groups
                ├── Regression Discontinuity Design (RDD)
                │      └── Uses sharp or fuzzy cutoffs as treatment assignment
                └── Instrumental Variables (IV)
                    └── Uses external “instrument” that influences treatment but not outcome directly

**What Lies Ahead:** We've looked at Z-tests and t-tests in detail in our previous post. Let's focus on other A/B testing mechanisms in this post including ANOVA / Chi-Square tests (parametric) and Permutation, Bootstrap Test (Non-Parametric).

## 3. ANOVA
Analysis of Variance (ANOVA) is a statistical test used to determine if there are any statistically significant differences between the means of **three or more** independent groups. It essentially checks if the variation *between* the groups is larger than the variation *within* the groups.

### The Concept Explained with an Example
Imagine a researcher wants to test if three different teaching methods (A, B, and C) have a different effect on student exam scores.
* **Group A:** Taught with Method A.
* **Group B:** Taught with Method B.
* **Group C:** Taught with Method C.

The researcher collects the final exam scores from students in each group.

**The question ANOVA answers is:** Are the differences in the average scores of these three groups large enough to conclude that the teaching methods have a real effect, or are the differences just due to random chance?
* **Null Hypothesis ($H_0$):** There is no difference between the teaching methods. The mean scores for all three groups are equal (${\mu_A = \mu_B = \mu_C}$).
* **Alternative Hypothesis ($H_a$):** At least one teaching method has a different mean score.

ANOVA works by comparing the **variance between the groups' average scores** to the **variance within each group's individual scores**. If the variance between the groups is significantly larger than the variance within the groups, we reject the null hypothesis.

### The Mathematical Background
ANOVA's math is centered around partitioning the total variation in the data into different components. This is done using "Sum of Squares."

#### 1. Partitioning the Variance (Sum of Squares)
* **Sum of Squares Total (SST):** Measures the total variation of every student's score from the overall average score of all students combined. It represents the total variation in the data.
* **Sum of Squares Between (SSB):** Measures the variation of each group's average score from the overall average score. This represents the variation **caused by the different teaching methods**.
* **Sum of Squares Within (SSW):** Measures the variation of each student's score from their own group's average. This represents the **random, unexplained variation or "noise"**.

The core relationship is: **SST = SSB + SSW**

#### 2. Formulation Details
Here are the formulas for the Sum of Squares in a One-Way ANOVA (Analysis of Variance), broken down by component.
To ensure clarity, let's first define the notation used in the equations:

**Notation Key**
* $k$: Number of groups.
* $n$: Total number of observations (items) across all groups.
* $n_i$: The number of observations in the $i^{th}$ group.
* $x_{ij}$: The $j^{th}$ observation in the $i^{th}$ group.
* $\bar{x}_i$: The sample mean of the $i^{th}$ group.
* $\bar{x}$: The **Grand Mean** (the mean of all $n$ observations).

**1. SST (Sum of Squares Total)**
This represents the total variation in the data. It compares every individual observation to the Grand Mean.

$$SST = \sum_{i=1}^{k} \sum_{j=1}^{n_i} (x_{ij} - \bar{x})^2$$

* **Degrees of Freedom ($df_T$):** $n - 1$

**2. SSB (Sum of Squares Between)**
This represents the variation **between** the group means. It measures how much the groups differ from the Grand Mean (the "treatment" effect).

$$SSB = \sum_{i=1}^{k} n_i (\bar{x}_i - \bar{x})^2$$

* **Degrees of Freedom ($df_B$):** $k - 1$
* *Note:* If the sample sizes for all groups are equal, $n_i$ is a constant.

**3. SSW (Sum of Squares Within)**
This represents the variation **within** each group. It measures the error or random variation (how far individual items are from their own group mean). This is often called $SSE$ (Sum of Squares Error).

$$SSW = \sum_{i=1}^{k} \sum_{j=1}^{n_i} (x_{ij} - \bar{x}_i)^2$$

**Degrees of Freedom ($df_W$):** $n - k$

#### 3. Calculating Mean Squares
To get an average measure of variance (called a **Mean Square**), we divide the Sum of Squares by the degrees of freedom (df).

* **Mean Square Between (MSB):** This is the average variance between the groups.
    $$MSB = \frac{SSB}{k - 1}$$
    (where `k` is the number of groups)

* **Mean Square Within (MSW):** This is the average variance within the groups (the random noise).
    $$MSW = \frac{SSW}{N - k}$$
    (where `N` is the total number of students)

#### 4. The F-statistic
The F-statistic is the final test statistic. It's the ratio of the variance between the groups to the variance within the groups.

$$F = \frac{\text{Mean Square Between}}{\text{Mean Square Within}} = \frac{MSB}{MSW}$$

**Interpretation:**
* If the F-statistic is close to 1, it means the variation between the groups is about the same as the random variation within the groups, so there's likely no treatment effect.
* If the F-statistic is large, it means the variation between the groups is much larger than the random variation, suggesting the treatment (the teaching method) has a real effect.

This calculated F-statistic is then compared to a critical value from an F-distribution to determine a p-value. If the p-value is below our significance level (e.g., 0.05), we reject the null hypothesis and conclude that at least one teaching method leads to a different average score.

#### Range of the F-statistic
The F-statistic can take any value from **0 to positive infinity**.

Since the F-statistic is a ratio of two variances (Mean Squares), which are always non-negative, the F-statistic itself can never be negative. There is no theoretical upper limit.

#### How to Find the Critical F-statistic
The critical F-statistic is the threshold value used to determine statistical significance. It's the "line in the sand" on the F-distribution. If your experimental F-statistic exceeds this critical value, your result is significant.

You find this value using three key pieces of information:
1.  **Significance Level (α):** The probability of a Type I error you're willing to accept (commonly 0.05).
2.  **Numerator Degrees of Freedom (df₁):** The degrees of freedom for the variance *between* groups. `df₁ = k - 1`, where `k` is the number of groups.
3.  **Denominator Degrees of Freedom (df₂):** The degrees of freedom for the variance *within* groups. `df₂ = N - k`, where `N` is the total number of observations.

In practice, you use statistical software or an F-distribution table to find the critical value corresponding to these three parameters.

#### Example
Let's use our previous example of testing three different teaching methods.

* **Number of groups (k):** 3
* **Total students (N):** Assume 10 students per group, so N = 30.
* **Significance level (α):** 0.05

**1. Calculate the Degrees of Freedom:**
  * **Numerator df₁** = k - 1 = 3 - 1 = **2**
  * **Denominator df₂** = N - k = 30 - 3 = **27**

**2. Find the Critical Value:**
  * We need to find the critical F-value for an F-distribution with (2, 27) degrees of freedom at an alpha of 0.05.
  * Using statistical software or an F-table, we find that the critical F-value is approximately **3.35**.

**Conclusion:**
This means if our experiment's calculated F-statistic is **greater than 3.35**, the p-value will be less than 0.05, and we would **reject the null hypothesis**. We would conclude that at least one teaching method has a significantly different effect on student scores.

## 4. Chi-squared Test
The Chi-squared (${\chi^2}$) test is a statistical tool used to determine if there is a significant difference between what you expected to find and what you actually observed in a dataset.

There are two primary types:
* **Test for Independence:** Checks if two variables are related or independent of each other.
* **Goodness of Fit Test:** Checks if a single variable's distribution matches what is expected.

### Test for Independence
#### Step 1: State the Hypotheses
First, you state your null and alternative hypotheses. The null hypothesis always assumes that there is no relationship between the variables (e.g. snack choice and movie genre).

* **Null Hypothesis ($H_0$):** The two variables are independent. In our example, snack choice and movie genre are not related.
* **Alternative Hypothesis ($H_a$):** The two variables are dependent. Snack choice and movie genre are related.

#### Step 2: Create a Contingency Table
You organize your collected data into a table of observed frequencies.

**Example Scenario:** You survey 200 people at a movie theater to see if there's a link between their favorite movie genre (Comedy, Action, Romance) and their preferred snack (Popcorn, Nachos).

**Observed Data Table:**

|                 | **Popcorn** | **Nachos** | **Row Total** |
| :-------------- | :---------- | :--------- | :------------ |
| **Comedy** | 70          | 20         | **90** |
| **Action** | 45          | 25         | **70** |
| **Romance** | 25          | 15         | **40** |
| **Column Total**| **140** | **60** | **200** |

#### Step 3: Calculate Expected Frequencies
Next, you calculate what the frequencies would be in each cell if the null hypothesis were true (i.e., if the variables were perfectly independent).

The formula for each cell is:
**Expected Value = (Row Total × Column Total) / Grand Total**

* **Expected (Comedy, Popcorn):** (90 × 140) / 200 = **63**
* **Expected (Comedy, Nachos):** (90 × 60) / 200 = **27**
* **Expected (Action, Popcorn):** (70 × 140) / 200 = **49**
* And so on for every cell...

#### Step 4: Calculate the Chi-Squared Statistic
This step measures the difference between your observed data and your expected data.

The formula is:

$${\chi^2 = \sum \frac{(O-E)^2}{E}}$$

Where:
* **O** = Observed frequency
* **E** = Expected frequency

You calculate this value for each cell and sum them up. For the "Comedy, Popcorn" cell:
* ${(70 - 63)^2 / 63 = 49 / 63 \approx 0.78}$

Let's assume after doing this for all cells, the final Chi-squared value is **${\chi^2 = 5.25}$**.

#### Step 5: Make a Conclusion
Finally, you compare your calculated Chi-squared value to a critical value from a Chi-squared distribution table. This requires a significance level (usually α = 0.05) and the degrees of freedom.

* **Degrees of Freedom** = (Number of Rows - 1) × (Number of Columns - 1) = (3 - 1) × (2 - 1) = **2**
* The critical Chi-squared value for 2 degrees of freedom at α = 0.05 is **5.99**.

**Conclusion:**
* Since our calculated value (5.25) is **less than** the critical value (5.99), we **fail to reject the null hypothesis**.
* This means there is not enough statistical evidence to conclude that a relationship exists between movie genre preference and snack choice.

### Goodness of Fit
The Chi-squared Goodness of Fit test determines if the observed frequency distribution of a single categorical variable matches a claimed or expected distribution. In essence, it tests how "well" your sample data "fits" a pre-existing theory.

Here is a step-by-step guide with an example.

#### The Scenario: M&M Colors
Let's say the M&M company claims the following color distribution for their plain candies:
* **Blue:** 24%
* **Green:** 16%
* **Orange:** 20%
* **Red:** 13%
* **Yellow:** 14%
* **Brown:** 13%

You buy a large bag containing **400** M&Ms and want to test if your bag's color distribution matches the company's claim.

#### Step 1: State the Hypotheses
* **Null Hypothesis ($H_0$):** The color distribution of M&Ms in your bag is the same as the distribution claimed by the company.
* **Alternative Hypothesis ($H_a$):** The color distribution is different from the company's claim.

#### Step 2: Organize Observed and Expected Frequencies
First, you count the M&Ms of each color in your bag (**Observed**). Then, you calculate how many of each color you would **expect** to see in a bag of 400 if the company's claim is true.

**Expected Count = Total Sample Size × Claimed Percentage**
For example, Expected Blue = 400 × 0.24 = 96.

| Color  | Observed (O) | Expected (E) |
| :----- | :------------- | :------------- |
| Blue   | 90             | 96             |
| Green  | 70             | 64             |
| Orange | 82             | 80             |
| Red    | 48             | 52             |
| Yellow | 55             | 56             |
| Brown  | 55             | 52             |
| **Total**| **400** | **400** |

#### Step 3: Calculate the Chi-Squared Statistic
This step measures the total difference between your observed and expected counts.

The formula is:

$${\chi^2 = \sum \frac{(O-E)^2}{E}}$$

You calculate this for each color and add them up:
* **Blue:** ${(90 - 96)^2 / 96 = 0.375}$
* **Green:** ${(70 - 64)^2 / 64 = 0.563}$
* **Orange:** ${(82 - 80)^2 / 80 = 0.050}$
* **Red:** ${(48 - 52)^2 / 52 = 0.308}$
* **Yellow:** ${(55 - 56)^2 / 56 = 0.018}$
* **Brown:** ${(55 - 52)^2 / 52 = 0.173}$

**Total Chi-Squared Value (${\chi^2}$)** = 0.375 + 0.563 + 0.050 + 0.308 + 0.018 + 0.173 = **1.487**

#### Step 4: Make a Conclusion
Finally, you compare your calculated ${\chi^2}$ value to a critical value from a Chi-squared distribution table.

* **Degrees of Freedom (df)** = Number of Categories - 1 = 6 - 1 = **5**
* For a significance level of **α = 0.05** and df = 5, the critical value is **11.07**.

**Conclusion:**
* Since our calculated ${\chi^2}$ value of **1.487** is much **less than** the critical value of **11.07**, we **fail to reject the null hypothesis**.
* This means there is no statistically significant difference between the color distribution in your bag and the distribution claimed by the M&M company. Your sample data "fits" the expected distribution well.

### Fundamental Differences with z and t-tests
The fundamental difference is the type of data they analyze: **z-tests and t-tests are used for continuous data** to compare means or proportions, while **Chi-squared tests are used for categorical data** to compare observed frequencies against expected frequencies.

This table summarizes the core distinctions between the tests.

| Feature         | Z-test / T-test                                       | Chi-squared Test                                           |
| :-------------- | :---------------------------------------------------- | :--------------------------------------------------------- |
| **Type of Data**| **Continuous** (e.g., height, revenue, time) or **Proportions** | **Categorical** (e.g., colors, genres, yes/no) |
| **Purpose** | To compare the **means** or **proportions** of one or two groups. | To compare **observed counts** against **expected counts**. |
| **Hypothesis About** | A population parameter like the mean (${\mu}$) or proportion (${p}$). | A distribution (**Goodness of Fit**) or the relationship between two variables (**Independence**). |

### Scenarios for Usage
Here are practical scenarios where you would choose one test over the other.

#### When to Use a Z-test or T-test
You use these tests when your question is about averages or percentages.
* **A/B Testing Conversion Rates:** Comparing the **conversion rate** (a proportion) of two different website layouts to see which is more effective.
* **Medical Research:** Testing if a new drug lowers blood pressure more than a placebo by comparing the **average blood pressure** in the two groups of patients.
* **Quality Control:** A factory wants to know if the **average weight** of a product from the assembly line is equal to the target weight of 500g.
* **Finance:** Determining if the **average daily return** of a stock is significantly different from zero.

#### When to Use a Chi-squared Test
You use this test when your question is about counts, frequencies, or relationships between categories.
* **Market Research:** Determining if there is a **relationship between a customer's age group** (18-25, 26-40, 41+) and their **preferred social media platform** (Instagram, Facebook, TikTok). This is a **Test for Independence**.
* **Genetics:** Testing if the **observed counts of offspring with different traits** (e.g., purple vs. white flowers) match the expected ratio predicted by Mendelian genetics. This is a **Goodness of Fit Test**.
* **Polling Analysis:** Analyzing survey results to see if **voting preference** (Party A, Party B, Undecided) is independent of the voter's geographical region (North, South, East, West). This is a **Test for Independence**.
* **Manufacturing:** A company claims its bags of candy contain an equal number of red, green, and blue pieces. You count the pieces in a random sample to see if the **observed frequencies** match the **expected frequencies**. This is a **Goodness of Fit Test**.

## 5. Bonferroni's Correction
The Bonferroni correction is a method used to counteract the increased risk of making a Type I error (a false positive) when you run multiple simultaneous hypothesis tests. It works by making the significance threshold for each individual test much stricter.

### The Problem: Multiple Comparisons Inflate Error Rates
When you run a single A/B test with a significance level (alpha) of 0.05, you accept a 5% chance of finding a significant result when there isn't one. However, when you run multiple tests at once, the probability of getting at least one false positive across the entire "family" of tests increases dramatically.

* **1 Test:** The probability of a false positive is **5%**.
* **3 Tests:** The probability of at least one false positive increases to about **14%**.
* **10 Tests:** The probability of at least one false positive jumps to about **40%**.

The Bonferroni correction is a simple, if conservative, way to control this **Family-Wise Error Rate (FWER)**.

### How the Bonferroni Correction Works
The method is straightforward: you divide your original alpha level by the number of tests you are performing.

$$\text{Corrected Alpha} = \frac{\text{Original Alpha}}{\text{Number of Tests } (n)}$$

This new, lower alpha becomes the p-value threshold you must beat for any individual test to be considered statistically significant.

### Example: A/B/C/D Test on a Homepage
Imagine you want to test three new headlines (B, C, and D) against your current headline (A, the control) to see which one increases the sign-up rate.

This involves running three separate comparisons:
1.  A vs. B
2.  A vs. C
3.  A vs. D

Your original significance level is $\alpha = 0.05$, and the number of tests is $n = 3$.

#### Without Correction (The Wrong Way)
You run the tests and get the following p-values:
* **A vs. B:** $p = 0.04$
* **A vs. C:** $p = 0.21$
* **A vs. D:** $p = 0.45$

Without correction, you would look at the A vs. B result ($p=0.04$) and, seeing it's less than 0.05, declare headline B the winner.

#### With Bonferroni Correction (The Right Way)
You first calculate your new, stricter significance threshold:

$$\text{Corrected Alpha} = 0.05 / 3 \approx 0.0167$$

Now, you compare your p-values to this new threshold:
* **A vs. B:** $p = 0.04$ (This is **greater than** 0.0167) $\rightarrow$ **Not Significant**
* **A vs. C:** $p = 0.21$ (This is **greater than** 0.0167) $\rightarrow$ **Not Significant**
* **A vs. D:** $p = 0.45$ (This is **greater than** 0.0167) $\rightarrow$ **Not Significant**

#### Conclusion
After applying the Bonferroni correction, you correctly conclude that none of the new headlines produced a statistically significant improvement. The result for headline B was likely a false positive caused by running multiple tests, and the correction helped you avoid making a poor business decision based on random noise.

### The Math Behind the Exploding Error Rate
The math behind the exploding chance of false positives relies on the basic rules of compound probability. The easiest way to see this is to calculate the probability of the opposite event: making **zero** false positives, and then subtracting that from 1.

Let's assume:
* Significance level ($\alpha$) = 0.05 (a 5% chance of a false positive on any single test).
* Probability of **NOT** getting a false positive on a single test = $1 - 0.05 = 0.95$.

### The Math with Examples
**1 Test:**
  * The probability of getting **at least one** false positive is simply **0.05** (or 5%).

**2 Tests:** The probability of having **no** false positive on the first test AND **no** false positive on the second test is: $0.95 \times 0.95 = 0.95^2 = 0.9025$
  * So, the probability of having **at least one** false positive is the complement: $1 - 0.9025 = 0.0975 \text{ (or 9.75%)}$

**5 Tests:** The probability of having **no** false positives across all five tests is: $0.95^5 \approx 0.774$
  * The probability of having **at least one** false positive is: $1 - 0.774 = 0.226 \text{ (or 22.6%)}$

**20 Tests:** The probability of having **no** false positives across all twenty tests is: $0.95^{20} \approx 0.358$
  * The probability of having **at least one** false positive is: $1 - 0.358 = 0.642 \text{ (or 64.2%)}$

As you can see, after just 20 tests, it's more likely than not that you've been fooled by random chance at least once.

### The General Formula
This demonstrates the formula for the **Family-Wise Error Rate (FWER)**, which is the probability of making at least one Type I error across $n$ tests:

$$FWER = 1 - (1 - \alpha)^n$$

## 6. Permutation Test
A permutation test works by simulating the null hypothesis—that there is no difference between versions A and B—by repeatedly shuffling the observed data and seeing how often you get a result as extreme as the one you actually saw.

The fundamental difference is that **z-tests and t-tests are *parametric***, meaning they assume your data follows a specific theoretical distribution (like a normal distribution). A **permutation test is *non-parametric***; it makes no assumptions about the data's distribution because it builds the null distribution directly from your collected data.

### How a Permutation Test Works for an A/B Test
The core idea is to test the null hypothesis: **"The labels 'A' and 'B' are meaningless."** If this were true, then randomly shuffling these labels among your data points shouldn't matter.

Here's the step-by-step process:

1.  **Calculate the Observed Difference:** First, you run your A/B test and calculate the actual difference between Group A and Group B for your chosen metric (e.g., the difference in conversion rates). Let's call this the **observed statistic**.

2.  **Pool the Data:** Combine the results from both Group A and Group B into a single large dataset. By pooling the data, you are effectively assuming the null hypothesis is true (that the labels A and B don't matter).

3.  **Shuffle and Re-assign:** Randomly shuffle the pooled data. Then, create new "A" and "B" groups of the same size as your original groups.

4.  **Calculate a Permuted Statistic:** Calculate the difference in the metric between your new, shuffled "A" and "B" groups. Record this value.

5.  **Repeat:** Repeat steps 3 and 4 thousands of times (e.g., 10,000 times). Each time, you get a new "permuted statistic."

6.  **Build the Null Distribution:** The collection of all your permuted statistics forms a distribution centered around zero. This is your empirically-derived null distribution—it shows the range of differences you could expect to see if the treatment had no real effect.

7.  **Calculate the P-value:** Finally, you count how many of the permuted statistics are as extreme or more extreme than your original **observed statistic**. This proportion is your p-value.

### Example: A/B Test for Conversion Rate
Let's test if changing a "Buy Now" button from blue (A) to green (B) increases the conversion rate.

#### 1. Observed Data
* **Group A (Blue Button):** 1,000 users, 80 converted. **Conversion Rate = 8%**
* **Group B (Green Button):** 1,000 users, 105 converted. **Conversion Rate = 10.5%**
* **Observed Statistic (Difference):** 10.5% - 8% = **2.5%**

#### 2. Permutation Test
* **Pool:** Combine all 2,000 users. This pool has a total of 185 conversions (80 + 105) and 1,815 non-conversions.

* **Shuffle & Recalculate (One Iteration):**
    * Randomly assign 1,000 users from the pool to a "new A" and the other 1,000 to a "new B."
    * Let's say in this shuffle, "new A" gets 90 conversions (9% rate) and "new B" gets 95 conversions (9.5% rate).
    * The first **permuted statistic** is 9.5% - 9% = **0.5%**.

* **Repeat 10,000 Times:** You now have 10,000 simulated differences. Most will be close to zero.

* **Calculate the P-value:** You check how many of those 10,000 simulated differences were 2.5% or greater. Let's say you find that 210 of them were.
    * **P-value** = 210 / 10,000 = **0.021**

#### Conclusion
Since the p-value of **0.021** is less than our typical significance level of 0.05, we **reject the null hypothesis**. The permutation test shows that the observed 2.5% difference is unlikely to be a result of random chance, suggesting the green button is genuinely more effective.

Here is the addition to your text. I have maintained the exact structure and example data to ensure continuity, followed by the comparison section.

## 7. Bootstrap Test
A bootstrap test works by using our data to estimate the "true" population distribution. Instead of shuffling labels to simulate a world where there is no difference (like the permutation test), the bootstrap simulates conducting the experiment over and over again to see how much the results fluctuate.

While the permutation test asks **"Is this result just random noise?"**, the bootstrap test asks **"How confident are we in the specific magnitude of this result?"**

### How a Bootstrap Test Works for an A/B Test
The core idea is **resampling with replacement**. We treat our sample groups (A and B) as if they *are* the entire population, and we draw samples from them to estimate the variability of the metric.

Here's the step-by-step process:

1.  **Calculate the Observed Difference:** Just like before, calculate the actual difference between Group A and Group B. (e.g., Conversion Rate B - Conversion Rate A).

2.  **Keep Groups Separate:** Unlike the permutation test, **do not pool the data**. Keep Group A and Group B distinct.

3.  **Resample with Replacement:**
    * Create a "Bootstrap Sample A" by drawing $N_A$ observations from the original Group A, putting each observation back after drawing it (sampling with replacement).
    * Create a "Bootstrap Sample B" by drawing $N_B$ observations from the original Group B in the same way.

4.  **Calculate a Bootstrap Statistic:** Calculate the conversion rate for "Bootstrap Sample A" and "Bootstrap Sample B," then find the difference. Record this value.

5.  **Repeat:** Repeat steps 3 and 4 thousands of times (e.g., 10,000 times).

6.  **Build the Sampling Distribution:** The collection of these 10,000 differences forms a distribution centered around your *observed* difference. This shows the spread of possible results you might see if you repeated the experiment many times.

7.  **Calculate Confidence Interval:** Instead of a p-value, you typically calculate a Confidence Interval (CI). For a 95% CI, you cut off the bottom 2.5% and top 2.5% of your bootstrap statistics. If this interval does not include zero, the result is statistically significant.

### Example: A/B Test for Conversion Rate (Revisited)
We use the same data: Blue Button (A) vs. Green Button (B).

#### 1. Observed Data
* **Group A:** 1,000 users, 80 converted (8%).
* **Group B:** 1,000 users, 105 converted (10.5%).
* **Observed Statistic (Difference):** **2.5%**

#### 2. Bootstrap Test
* **Resample (One Iteration):**
    * We draw 1,000 users from Group A (with replacement). Some users might be picked twice, some not at all. Let's say this "Bootstrap A" has 75 conversions (7.5%).
    * We draw 1,000 users from Group B (with replacement). Let's say "Bootstrap B" has 110 conversions (11.0%).
    * The first **bootstrap statistic** is 11.0% - 7.5% = **3.5%**.

* **Repeat 10,000 Times:** We generate 10,000 variations of the difference. Some will be 2.5%, some 1%, some 4%, etc.

* **Calculate the 95% Confidence Interval:**
    * We sort all 10,000 differences from lowest to highest.
    * We look at the value at the 2.5th percentile (the 250th value) and the 97.5th percentile (the 9,750th value).
    * Let's assume our interval comes out to **[0.4%, 4.6%]**.

#### Conclusion
Because the 95% Confidence Interval **[0.4%, 4.6%]** does not contain **0.0%**, we conclude that the difference is statistically significant. Moreover, we can say with 95% confidence that the Green Button increases conversion by somewhere between 0.4% and 4.6%.

### Comparison: Permutation vs. Bootstrap
While both are powerful non-parametric methods that rely on simulation rather than complex formulas, they serve slightly different purposes.

| Feature | Permutation Test | Bootstrap Test |
| :--- | :--- | :--- |
| **Primary Goal** | **Hypothesis Testing:** Testing if an effect exists. | **Estimation:** Estimating the accuracy/variance of an effect. |
| **Mechanism** | **Shuffling:** Destroys the relationship between the group and the outcome to simulate "random chance." | **Resampling:** Preserves the relationship to simulate "repeating the experiment." |
| **Null Hypothesis** | Assumes $H_0$ is true (Groups are identical). | Does not assume $H_0$ is true; assumes sample $\approx$ population. |
| **Distribution Center** | Centered around **Zero** (Null Distribution). | Centered around the **Observed Difference**. |
| **Output** | Produces a precise **P-value**. | Produces a **Confidence Interval** (and Standard Error). |

### When to Use Which?
**Use a Permutation Test when:**
* You strictly want a **"Yes/No" answer** on significance (P-value).
* You have a very small sample size where assumptions of normality (for Z/T-tests) definitely do not hold, and you want the most robust test against False Positives.
* You are testing a complex metric where the standard error is hard to derive mathematically, but you only care if Group B is different from Group A.

**Use a Bootstrap Test when:**
* You care about **magnitude and uncertainty** (Confidence Intervals). Business stakeholders usually prefer "The uplift is between 2% and 5%" over "P-value is 0.02."
* You are analyzing a metric that is **not a mean or proportion**, such as a **Median** or **99th Percentile** (e.g., "Does the new feature reduce the 99th percentile latency?"). There are no simple formulas for the standard error of a median, making Bootstrap the gold standard here.

### Practical Differences
1.  **Hypothesis Testing (p-value):** For testing the null hypothesis of no difference, both methods are excellent choices and will lead to the same conclusion almost all the time.

2.  **Confidence Intervals:** This is the main difference. Because bootstrapping estimates the sampling distribution of your metric, it can be used to construct a **confidence interval** for the effect size (e.g., the true difference in P90 APE between the models). A permutation test cannot do this because its entire structure is based on the assumption of the null hypothesis (no difference).

### How do We Decide on the Simulation Count
Determining the number of iterations (often denoted as $B$) involves balancing **precision** against **computational time**.

Here is the breakdown of how to decide the number of simulations for both Permutation Tests and Bootstrap Tests.

#### **1. The Quick "Rules of Thumb"**
If you want a standard answer that satisfies most reviewers and statistical requirements without doing math:

| Goal | Recommended $B$ | Why? |
| :--- | :--- | :--- |
| **Initial / Debugging** | **100** | Fast enough to check if code works; not accurate enough for results. |
| **P-value ($\alpha = 0.05$)** | **1,000** | Sufficient to distinguish if $p < 0.05$ reliably. |
| **P-value ($\alpha = 0.01$)** | **5,000 - 10,000** | You need more resolution to see "rare" events in the tail. |
| **Confidence Intervals** | **2,000+** | Estimating tails (2.5th and 97.5th percentiles) is harder than estimating the mean. |
| **Publication Quality** | **10,000+** | Computers are fast now; running 10k usually removes any doubt about simulation error. |

#### **2. The Technical Approach (Calculating $B$)**
If you need to justify your choice or have a specific precision target, use these formulas.

##### **A. For P-values (Permutation & Bootstrap Hypothesis Tests)**
A simulation-based p-value ($\hat{p}$) is an estimate of the true p-value ($p$). The uncertainty of this estimate follows a binomial distribution. You can calculate the standard error ($SE$) of your estimated p-value:

$$SE(\hat{p}) = \sqrt{\frac{p(1-p)}{B}}$$

**How to use this:**
1.  **Guess the p-value:** Assume the p-value is near your significance threshold (e.g., $p \approx 0.05$).
2.  **Set desired precision:** Decide how much "noise" you can tolerate. For example, you want a standard error of 0.005 (so a true $0.05$ doesn't accidentally look like $0.06$).
3.  **Solve for $B$:**
    $$B = \frac{p(1-p)}{SE^2}$$

**Example:**
To distinguish $p=0.05$ with a precision ($SE$) of $0.005$:
$$B = \frac{0.05(1-0.05)}{0.005^2} = \frac{0.0475}{0.000025} = 1,900 \text{ iterations}$$

##### **B. For Confidence Intervals (Bootstrap)**
Confidence intervals (CIs) require estimating the *tails* of the distribution (e.g., the 2.5th and 97.5th percentiles). The tails are much more volatile than the center (mean).
* **Recommendation:** Use at least **2,000** bootstraps.
* **Pre-calculation:** There is no simple formula like the one above, but studies show that for 95% CIs, $B=2,000$ stabilizes the endpoints sufficiently for most data.

#### **3. The "Exact" Limit (When to Stop Simulating)**
Sometimes your sample size is small enough that you don't need to simulate—you can calculate **every possible permutation**. This is called an "Exact Test."

Check the total number of permutations ($N_{perm}$) before you start:

* **Two-Sample Test:** If you have Group A ($n_a$) and Group B ($n_b$), the distinct combinations are:
    $$N_{perm} = \binom{n_a + n_b}{n_a}$$
* **Paired/Correlation Test:** For $n$ pairs, the permutations are often $2^n$ or $n!$ depending on the specific test.

**Rule:** If $N_{perm} < 10,000$, do not simulate. Just run every single permutation. It is more accurate and often faster.

* *Example:* Comparing 2 groups of 5 items each.
    $\binom{10}{5} = 252$.
    *Don't do 1,000 simulations. Just calculate all 252 exact combinations.*

#### **Summary Checklist**
1.  **Check Exactness:** Is your sample size small ($n < 10-15$)? If yes, calculate the exact theoretical limit. If it's small, run the exact test.
2.  **Check Goal:**
    * **Hypothesis Test:** Use $B = 1,000$ to $5,000$.
    * **Confidence Interval:** Use $B \geq 2,000$.
3.  **Check Resources:** If your code takes 1 second to run 1 iteration, 10,000 iterations = ~3 hours. If that's too long, use the precision formula (Section 2A) to find the minimum viable $B$.
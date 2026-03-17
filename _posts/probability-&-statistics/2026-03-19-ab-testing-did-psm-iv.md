---
title: "Beyond A/B Testing: Causal Inference in the Wild (DiD, PSM, and IV)"
date: 2026-03-19 18:00:00 +0530
categories: [Statistics]
tags: [Statistics]
math: true
---

If you've been following my series on statistical testing, you're already comfortable with 2-sample t-tests, ANOVA, and Chi-Square tests. Those tools are fantastic for randomized, perfectly controlled A/B tests. But what happens when you can't perfectly randomize? What happens when the real world gets messy?

In this article, we are stepping into the realm of Causal Inference. We will explore three robust methods to measure true uplift when standard A/B testing fails: Difference-in-Differences (DiD), Propensity Score Matching (PSM), and Instrumental Variables (IV).

To anchor these concepts, let's use a continuous backdrop: A Cricket Coaching Academy.

## 1. The Setup: The Cricket Academy Problem
Imagine you run a cricket academy with 100 students. You've been tracking their Batting Averages. Suddenly, a prominent International Player offers to guest-coach a batch of your students for one month.

Your objective as a Data Scientist: Measure the exact uplift (in runs) caused by the International Player's coaching.

### Why the Standard 2-Sample T-Test Fails
Your first instinct might be to split the kids, let the month pass, and run a 2-sample t-test comparing the final Batting Averages of the Control Group vs. the Treatment (Pro Coach) Group.

**The fatal flaw here is the "Snapshot Problem"**. A standard t-test is cross-sectional, it only looks at the final score.
* What if the Control group started with an average of 40 and ended at 42?
* What if the Treatment group started at 20 and ended at 30?

A t-test would simply compare 42 vs. 30 and conclude the Control group is "better," completely missing the massive +10 uplift achieved by the Pro coach. It implicitly assumes both groups started at the exact same baseline, which is rarely true even with randomization.

## 2. Difference-in-Differences (DiD): The Time-Aware Approach
If we randomized the 100 kids into two groups of 50, DiD is our most robust approach. DiD doesn't compare final scores; it compares **trajectories**.

### The Intuition
DiD isolates the Treatment Effect from the Time Effect. Even without the special coach, kids naturally improve over a month due to regular practice. DiD calculates the natural growth of the Control group and subtracts it from the growth of the Treatment group.

### The DiD Equation:

$$\text{Uplift} = (\bar{Y}_{Treatment\_Post} - \bar{Y}_{Treatment\_Pre}) - (\bar{Y}_{Control\_Post} - \bar{Y}_{Control\_Pre})$$

Let's plug in some hypothetical numbers. Suppose your regular students (Control) improved their batting average from 20.0 to 22.0 over the month (a +2.0 natural gain). The students training with the Pro (Treatment) improved from 20.0 to 28.0 (a +8.0 gain).

Uplift = (28.0 - 20.0) - (22.0 - 20.0) = 8.0 - 2.0 = +6.0 runs.

### The "So What?" Problem: Proving the ROI
We found an average uplift of +6.0 runs. But is this a reliable result, or just a lucky month for a few kids in the Treatment group? A single point estimate (the number 6.0) doesn't tell us anything about the variance or spread of the underlying data.

From a practical business perspective, this distinction matters immensely. The International Player isn't coaching out of the goodness of their heart—they charge a premium fee. Before you sign a lucrative contract to retain them for the rest of the year, you need to prove this +6.0 is **statistically significant** (highly unlikely to be random noise). If the students' scores are wildly scattered and the result is just a statistical mirage, paying for the coach is a terrible investment.

### The Mathematical Formulation (OLS Regression)
To get those crucial confidence intervals and p-values, we can't just do napkin math. We must formulate DiD as a robust statistical model. We achieve this using an Ordinary Least Squares (OLS) regression with an interaction term:

$$Y_{it} = \beta_0 + \beta_1(\text{Group}_i) + \beta_2(\text{Time}_t) + \beta_3(\text{Group}_i \times \text{Time}_t) + \epsilon_{it}$$

Here is the rationale behind the components of the equation:
* $Y_{it}$ (Outcome): The batting average for a specific student $i$ at a specific time period $t$.
* $\beta_0$ (Intercept): The baseline batting average of the Control group at Time 0.
* $\beta_1$ (Group Effect): Controls for the baseline difference between the groups (the starting skill gap).
* $\beta_2$ (Time Effect): Controls for the natural time trend (the parallel improvement everyone experiences).
* $\beta_3$ (The DiD Estimator / True Uplift): By isolating the interaction between being in the Treatment group and being in the Post-training time period, it extracts the exact value added by the coach. Because it's a regression coefficient, it comes complete with standard errors and a p-value to guide our financial decision.
* $\epsilon_{it}$ (Error Term): The random noise or unobserved individual factors for that specific student.

### Interpreting a specific instance ($Y_{it}$):
To see how this works in practice, let's calculate the expected batting average for a student in the **Treatment Group** after the 1-month training (Post-test), using our previous toy numbers ($\beta_0 = 20.0$, $\beta_1 = 0.0$ assuming no baseline difference, $\beta_2 = 2.0$, and the uplift $\beta_3 = 6.0$):
1. For this specific scenario, we plug in $Group_i = 1$ (Treatment) and $Time_t = 1$ (Post-test).
2. The equation becomes: $Y_{it} = 20.0 + 0.0(1) + 2.0(1) + 6.0(1 \times 1) + \epsilon_{it}$
3. Solving this gives us: $Y_{it} = 28.0 + \epsilon_{it}$

This tells us that we expect this specific student to score a $28.0$, plus or minus some individual random variance ($\epsilon_{it}$). The OLS equation elegantly builds the final score by systematically layering the baseline, the natural time trend, and the coach's specific uplift!

### DiD Python Simulation

```import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

np.random.seed(42)

# Simulate 100 students (2 time periods = 200 rows)
n_students = 100
records = []

for student_id in range(1, n_students + 1):
    is_treatment = 1 if student_id > 50 else 0
    base_skill = np.random.normal(20, 3) # Starting average around 20
    
    # Pre-test (Time = 0)
    records.append({
        "student_id": student_id, "time": 0, "group": is_treatment,
        "batting_avg": base_skill + np.random.normal(0, 2)
    })
    
    # Post-test (Time = 1) - Natural time effect is +2. Coach effect is +6.
    coach_effect = 6.0 if is_treatment else 0
    time_effect = 2.0
    records.append({
        "student_id": student_id, "time": 1, "group": is_treatment,
        "batting_avg": base_skill + time_effect + coach_effect + np.random.normal(0, 2)
    })

df_did = pd.DataFrame(records)

# Run DiD Regression
model_did = smf.ols('batting_avg ~ group * time', data=df_did).fit()
print(model_did.summary().tables[1])
print(f"\nEstimated Uplift (group:time): {model_did.params['group:time']:.2f} runs")
print(f"P-Value: {model_did.pvalues['group:time']:.4f}")
```

### Bonus: The Power of Covariates (e.g., Age)
Here is arguably the biggest reason Data Scientists prefer regression (OLS) over the vanilla arithmetic method for DiD: Adding Covariates.

Imagine you realize that older kids naturally hit the ball further, regardless of the coach. If you try to calculate the DiD by hand, you'd have to slice your data into tiny, messy buckets (a DiD for 12-year-olds, a DiD for 13-year-olds, etc.) to ensure a fair comparison.

With OLS, you simply add Age to your equation:

$$Y_{it} = \beta_0 + \beta_1(\text{Group}_i) + \beta_2(\text{Time}_t) + \beta_3(\text{Group}_i \times \text{Time}_t) + \beta_4(\text{Age}_i) + \epsilon_{it}$$

* $\beta_4$ (Covariate Effect): This new term captures the isolated effect of a student's age on their batting average.

By adding this, OLS mathematically "holds Age constant." It figures out exactly how much advantage age provides and subtracts that background noise from everyone's score. The result? Your data points cluster much tighter around the trend line, your Standard Error plummets, and your p-value for the true uplift ($\beta_3$) becomes much sharper. In Python, this is as easy as changing your formula to `batting_avg ~ group * time + age`. You've effectively noise-canceled your experiment!

## 3. Propensity Score Matching (PSM): Fixing Selection Bias
**The Plot Twist**: What if we didn't randomize? What if, out of our 100 total students, the academy owner hand-picked the 25 "most promising" students to train with the Pro?

Now DiD is broken. The Parallel Trends Assumption is dead. Promising students don't just have higher baseline scores; they learn faster. Their natural time trend is steeper than the average student's.

### The Intuition
To make a fair comparison, we must retrospectively create a "quasi-randomized" control group. We look at the 75 kids left behind and find the statistical "twins" of the 25 who were selected.

### The Mechanism
1. **Calculate Propensity**: Run a Logistic Regression using observable traits (Skill, Motivation, Age) to predict the probability (0.0 to 1.0) of a student being selected for the camp.
2. **Match (Sequential Matching without Replacement)**: Now we find the "twins." We use a greedy, sequential algorithm to pair students up.
    * We take the first Treated student and look at their Propensity Score (e.g., 0.85).
    * We search the Control group for the student with the closest score (e.g., 0.84) and pair them up.
    * Crucially, we then remove that Control student from the available pool. This is called matching without replacement. It ensures that if we have two elite Treated students, they don't accidentally get matched to the exact same elite Control student.
    * We repeat this process until all 25 Treated students have a unique Control partner.
3. **Discard**: Throw away the remaining 50 unmatched control students. They were never promising to begin with, so comparing them to the elite group would introduce selection bias.
4. **Compare**: Calculate the difference in outcomes only among the 25 matched pairs.

**Are we using the DiD OLS equation here?**

No. DiD requires panel data (a Pre-test and a Post-test) to calculate slopes over time. PSM is typically used on cross-sectional data (just the final Post-test snapshot). Because the matching process has already artificially balanced the baselines of the two groups, we don't need the complex time-interaction math anymore.

Instead, we simply calculate the **Average Treatment Effect on the Treated (ATT)**:

$$\text{ATT} = \bar{Y}_{Treated\_Matched} - \bar{Y}_{Control\_Matched}$$

To get the crucial p-value that tells us if this uplift is mathematically significant, you run a standard, simple OLS regression exclusively on your new, matched 50-student dataset (`outcome ~ treatment`). The regression equation looks like this:

$$Y_{i} = \beta_0 + \beta_1(\text{Treatment}_i) + \epsilon_{i}$$

* $Y_i$ (Outcome): The final batting average for student $i$.
* $\beta_0$ (Intercept): The baseline average score of the matched control group.
* $\beta_1$ (The ATT / True Uplift): This coefficient represents the exact value added by the coach. This is the $\beta$ value we look for in our model output, and its associated p-value tells us if the coach's impact is statistically significant.
* $\epsilon_{i}$ (Error Term): The remaining random variance per student.

### PSM Python Simulation
```
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
n_total = 100

# Confounders: Highly skilled and motivated kids are more likely to be picked
skill = np.random.normal(50, 15, n_total)
motivation = np.random.normal(5, 2, n_total)

# Biased Assignment Logic
# We tune the intercept (-11.8) so that roughly 25 out of 100 students get picked
z = -11.8 + 0.1 * skill + 1.0 * motivation
prob_treatment = 1 / (1 + np.exp(-z))
treatment = np.random.binomial(1, prob_treatment)

# True uplift is 5.0. 
outcome = (0.5 * skill) + (2.0 * motivation) + (5.0 * treatment) + np.random.normal(0, 2, n_total)

df_psm = pd.DataFrame({'skill': skill, 'motivation': motivation, 'treatment': treatment, 'outcome': outcome})

print(f"Total Selected for Camp: {df_psm['treatment'].sum()}")

# 1. Naive Comparison (Severely Biased!)
naive_diff = df_psm[df_psm.treatment==1]['outcome'].mean() - df_psm[df_psm.treatment==0]['outcome'].mean()
# INTERPRETATION OF NAIVE UPLIFT: 
# This will print roughly 16.20 runs. Why is it so ridiculously high? 
# Because it conflates the true value of the coach (5.0) WITH the fact that the 
# treated kids already possessed much higher 'skill' and 'motivation' baselines.
print(f"Naive Uplift (Biased): {naive_diff:.2f} runs (Artificially high!)") 

# 2. Calculate Propensity Scores
lr = LogisticRegression()
lr.fit(df_psm[['skill', 'motivation']], df_psm['treatment'])
df_psm['propensity_score'] = lr.predict_proba(df_psm[['skill', 'motivation']])[:, 1]

# 3. Match using a Sequential Greedy algorithm (Without Replacement)
treated = df_psm[df_psm.treatment == 1].copy()
control = df_psm[df_psm.treatment == 0].copy()

matched_control_indices = []
available_controls = control.copy()

for _, treated_row in treated.iterrows():
    if len(available_controls) == 0:
        break # Ran out of controls!
    
    # Calculate absolute difference in propensity scores
    distances = np.abs(available_controls['propensity_score'] - treated_row['propensity_score'])
    
    # Find the index of the closest control student
    best_match_idx = distances.idxmin()
    matched_control_indices.append(best_match_idx)
    
    # Remove this control student from the pool (without replacement)
    available_controls = available_controls.drop(best_match_idx)

# Extract the perfectly matched control group
matched_control = df_psm.loc[matched_control_indices]

# 4. Calculate True Effect (ATT)
att = treated['outcome'].mean() - matched_control['outcome'].mean()
# INTERPRETATION OF PSM ESTIMATE:
# This will print roughly 11.47 runs. Notice how it stripped out a massive chunk of the 
# selection bias, dropping from 16.20 down to ~11.5!
# Why isn't it perfectly 5.0? Because our sample size is tiny (only ~25 matched pairs). 
# With such a small pool of controls to pick from, some elite 'treated' kids had to be matched 
# with slightly less elite 'control' kids. If we had N=10,000, PSM would converge accurately.
print(f"PSM Estimated Uplift: {att:.2f} runs (True effect is 5.0)")

# To get the p-value, we run a simple regression on the matched dataset
matched_dataset = pd.concat([treated, matched_control])
import statsmodels.formula.api as smf
psm_model = smf.ols('outcome ~ treatment', data=matched_dataset).fit()
print(f"P-Value of Uplift: {psm_model.pvalues['treatment']:.4f}")
```

## 4. Instrumental Variables (IV): The Heavy Artillery
**The Final Plot Twist**: The academy decides the Pro coach is an optional, paid add-on. Parents must opt-in and pay extra.

We now have Unobserved Confounding. Parents who pay for the coach might also be highly motivated parents who buy their kids better diets, personal batting cages, and top-tier gear.
* We cannot use DiD because parallel trends are broken by this underlying parental wealth/motivation.
* We cannot use PSM because we suffer from Unobserved Confounding. PSM has a strict mathematical rule: you cannot match on what you cannot measure. If you try to match a Treated student and a Control student based only on their visible traits (like Skill and Age), you might inadvertently pair a kid with highly involved, wealthy parents against a kid without that invisible support system. They might look like statistical "twins" on your spreadsheet, but the hidden wealth factor gives the Treated kid an invisible advantage (better diet, expensive bats, weekend games). You are matching apples to oranges without realizing it, meaning the selection bias remains completely intact.

### The Intuition
We need an Instrument ($Z$): A random event that "pushes" kids into getting the coach ($X$) but has absolutely zero direct effect on their Batting Average ($Y$).
**The Instrument**: A promotional lottery. You randomly give 50 kids a "50% Off" voucher for the Pro coach.
* Relevance: Winning the voucher heavily increases the chance a parent hires the coach.
* Exclusion Restriction: A piece of paper (the voucher) doesn't magically make a kid hit the ball better. It only works through the hiring of the coach.

### The Mechanism (Two-Stage Least Squares - 2SLS)
We use the random voucher to extract the "clean" variation in coaching through two consecutive regression equations. *(Note: We use standard Linear Regression (OLS) for both stages to prevent a mathematical trap called the "Forbidden Regression," ensuring our math stays unbiased even though Coach_Hired is binary)*.

### Stage 1: The "Clean" Push
**Objective**: Figure out exactly how much the random voucher ($Z$) influences a student's likelihood of hiring the coach ($X$), ignoring everything else.

$$\text{Coach\_Hired}_i = \alpha_0 + \alpha_1(\text{Voucher}_i) + u_i$$

Let's look at a concrete example. Suppose out of the 50 kids who didn't get the voucher, 10 of them (20%) had wealthy/motivated parents who paid full price for the coach anyway. But out of the 50 kids who did win the voucher, 35 of them (70%) hired the coach.
* The base rate ($\alpha_0$) is 0.20 (20%).
* The voucher effect ($\alpha_1$) is 0.50 (70% - 20%). The voucher successfully "pushed" an extra 50% of kids into getting coaching!

Instead of returning a simple 1 or 0, this Stage 1 equation generates a Predicted Probability ($\hat{X}$) for every kid:
* Losers get a "clean" score of 0.20
* Winners get a "clean" score of 0.70

This $\hat{X}$ value is pure. It is completely stripped of "parental wealth" bias because it was derived 100% from the random lottery results.

### Stage 2: The True Effect
**Objective**: Calculate how this pure, clean variation ($\hat{X}$) affects the final Batting Average ($Y$).

$$\text{Batting\_Average}_i = \beta_0 + \beta_{IV}(\hat{X}_i) + \epsilon_i$$

Let's say the average batting score for the Lottery Losers (our 0.20 bucket) is 20 runs. The average score for the Lottery Winners (our 0.70 bucket) is 24 runs.

The math calculates the slope: How much did the Batting Average jump when the clean probability jumped?
* Change in Outcome ($Y$) = 24 - 20 = +4.0 runs
* Change in Probability ($\hat{X}$) = 0.70 - 0.20 = 0.50

$$\beta_{IV} = \frac{\text{Change in } Y}{\text{Change in } \hat{X}} = \frac{4.0}{0.50} = \mathbf{+8.0 \text{ runs}}$$

**The Intuition Check**: We saw a raw +4.0 run improvement across the whole winning group. But we know the voucher only swayed 50% of the group to actually take action. Therefore, the true effect on the kids who actually used the coach must have been double (+8.0 runs) to account for the fact that the other half of the group diluted the average!

### IV Python Simulation
*(For production use, it is highly recommended to use linearmodels.iv to get correct standard errors, but we will demonstrate the manual 2-stage OLS here to show the pure math).*

```
import statsmodels.api as sm

np.random.seed(42)
n = 100 # Adjusted from 1000 to 100 to match the 100-student scenario

# 1. The Unobserved Confounder (Parental Wealth/Motivation)
unobserved_wealth = np.random.normal(50, 10, n)

# 2. The Instrument: Randomly assign 50% off voucher (0 or 1)
voucher_Z = np.random.binomial(1, 0.5, n) # ~50 kids will get the voucher

# 3. Treatment Assignment (Coach): Depends on BOTH Voucher and Unobserved Wealth
# Voucher gives a push, wealth gives a push
coach_prob = -4 + (2.5 * voucher_Z) + (0.05 * unobserved_wealth)
coach_X = np.random.binomial(1, 1 / (1 + np.exp(-coach_prob)))

# 4. Outcome: Depends on Coach (+8.0 True Effect) AND Unobserved Wealth
batting_Y = 20 + (8.0 * coach_X) + (0.3 * unobserved_wealth) + np.random.normal(0, 2, n)

df_iv = pd.DataFrame({'voucher': voucher_Z, 'coach': coach_X, 'outcome': batting_Y})

# Naive Regression (Fails due to unobserved wealth)
naive_model = smf.ols('outcome ~ coach', data=df_iv).fit()
print(f"Naive Estimate (Biased by Wealth): {naive_model.params['coach']:.2f}")

# --- Two-Stage Least Squares (2SLS) ---

# Stage 1: Regress X on Z to get predicted X_hat
stage1 = smf.ols('coach ~ voucher', data=df_iv).fit()
df_iv['coach_hat'] = stage1.predict()

# Stage 2: Regress Y on X_hat
stage2 = smf.ols('outcome ~ coach_hat', data=df_iv).fit()
print(f"IV 2SLS Estimate (True Effect):    {stage2.params['coach_hat']:.2f} (True is 8.0)")
```

## Conclusion
A/B testing is the gold standard, but data scientists rarely work in perfectly controlled laboratories.
* If you have grouped time-series data with parallel trends, use DiD.
* If you have selection bias but you recorded all the confounding variables, use PSM.
* If you have unmeasured confounding but can find a clever randomized "nudge," use Instrumental Variables.

Mastering these three techniques transitions you from just reading data, to uncovering the true mechanics of how your business operates.

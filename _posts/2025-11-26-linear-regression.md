---
title: "All about Linear Regression"
date: 2025-11-25 11:30:00 +0530
categories: [Machine Learning]
tags: [ML]
math: true
---

# Derivation of the Normal Equation
The Normal Equation is a method for finding the optimal parameters in a linear regression model without resorting to an iterative optimization algorithm like Gradient Descent. It provides a direct, analytical solution. Here is a step-by-by-step derivation using matrix notation.

### 1. The Linear Regression Model

First, let's define the hypothesis of a linear regression model. For a given input feature vector $x$, the predicted output $h_\theta(x)$ is given by:

$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n$

where:
* $\theta_0, \theta_1, \dots, \theta_n$ are the parameters (or weights) of the model.
* $x_1, x_2, \dots, x_n$ are the input features.
* For convenience, we define $x_0 = 1$ to incorporate the bias term $\theta_0$.

This can be written more compactly in vector form as:

$h_\theta(x) = \theta^T x$

where:
* $\theta = [\theta_0, \theta_1, \dots, \theta_n]^T$ is the parameter vector.
* $x = [x_0, x_1, \dots, x_n]^T$ is the feature vector.

For a training set with $m$ examples, we can represent all the training data in matrix form:

* $X$ is the design matrix of size $m \times (n+1)$, where each row is a training example.
* $y$ is a vector of size $m \times 1$ containing the true output values.
* The vector of all predictions can be written as $X\theta$.

$$
X = \begin{pmatrix}
(x^{(1)})^T \\
(x^{(2)})^T \\
\vdots \\
(x^{(m)})^T
\end{pmatrix} = \begin{pmatrix}
x_0^{(1)} & x_1^{(1)} & \cdots & x_n^{(1)} \\
x_0^{(2)} & x_1^{(2)} & \cdots & x_n^{(2)} \\
\vdots & \vdots & \ddots & \vdots \\
x_0^{(m)} & x_1^{(m)} & \cdots & x_n^{(m)}
\end{pmatrix}, \quad
y = \begin{pmatrix}
y^{(1)} \\
y^{(2)} \\
\vdots \\
y^{(m)}
\end{pmatrix}, \quad
\theta = \begin{pmatrix}
\theta_0 \\
\theta_1 \\
\vdots \\
\theta_n
\end{pmatrix}
$$

### 2. The Cost Function (Sum of Squared Errors)

The goal of linear regression is to find the optimal parameter vector $\theta$ that minimizes the difference between the predicted values and the actual values. This is typically done by minimizing a cost function. The most common cost function for linear regression is the Mean Squared Error (MSE), which is the average of the sum of squared differences (errors).

For $m$ training examples, the sum of squared errors $J(\theta)$ is:

$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$

The term $\frac{1}{2m}$ is a constant for scaling that simplifies the derivative later on.

Using matrix notation, the vector of errors is $(X\theta - y)$. The sum of the squares of these errors can be expressed as the dot product of the error vector with itself:

$(X\theta - y)^T (X\theta - y)$

So, the cost function in matrix form is:

$J(\theta) = \frac{1}{2m} (X\theta - y)^T (X\theta - y)$

### 3. Minimizing the Cost Function

To find the value of $\theta$ that minimizes $J(\theta)$, we need to find the point where the gradient of $J(\theta)$ with respect to $\theta$ is the zero vector.

$\nabla_\theta J(\theta) = 0$

Let's first expand the cost function:

$J(\theta) = \frac{1}{2m} ((X\theta)^T - y^T) (X\theta - y)$

$J(\theta) = \frac{1}{2m} (\theta^T X^T - y^T) (X\theta - y)$

$J(\theta) = \frac{1}{2m} (\theta^T X^T X \theta - \theta^T X^T y - y^T X \theta + y^T y)$

A key point here is that $\theta^T X^T y$ is a scalar, so its transpose is equal to itself: $(\theta^T X^T y)^T = y^T (X^T)^T (\theta^T)^T = y^T X \theta$. Therefore, the two middle terms are identical.

$J(\theta) = \frac{1}{2m} (\theta^T X^T X \theta - 2y^T X \theta + y^T y)$

Now, we compute the gradient of $J(\theta)$ with respect to $\theta$. We use the following matrix calculus identities:
* $\nabla_\theta (\theta^T A \theta) = 2A\theta$ (for symmetric A, and $X^TX$ is symmetric)
* $\nabla_\theta (c^T \theta) = c$

Applying these rules to our cost function:

$\nabla_\theta J(\theta) = \frac{1}{2m} (2X^T X \theta - 2X^T y + 0)$
$\nabla_\theta J(\theta) = \frac{1}{m} (X^T X \theta - X^T y)$

### 4. The Normal Equation

Set the gradient to zero to find the minimum:

$\frac{1}{m} (X^T X \theta - X^T y) = 0$

$X^T X \theta - X^T y = 0$

$X^T X \theta = X^T y$

This final expression is the **Normal Equation**. To solve for the optimal parameter vector $\theta$, we can multiply both sides by the inverse of $(X^T X)$:

$\theta = (X^T X)^{-1} X^T y$

This equation gives a direct, analytical solution for the optimal $\theta$, provided that the matrix $(X^T X)$ is invertible. If $(X^T X)$ is not invertible (i.e., it is singular), it could be due to redundant features (linear dependence) or having more features than training examples. In such cases, techniques like regularization can be used.
---
toc: true
layout: post
description: Pattern Recognition and Machine Learning Summary
categories: [Machine Learning]
title: Pattern Recognition and Machine Learning Summary
---

## 1.2 Probability Theory

- The probability that $X$ will take the value $x_i$ and $Y$ will take the value $y_j$ is
written $p(X = x_i, Y = y_j)$ and is called the **joint probability** of $X = x_i$ and
$Y = y_j$. It is given by the number of points falling in the cell $i,j$ as a fraction of the
total number of points, and hence

$$ p(X = x_i, Y = y_j) = \frac{n_{ij}}{N} $$

$$ p(X = x_i) = \frac{c_i}{N} $$

- **sum rule of probability** $$p(X = x_i) = \sum_{j=1}^{L} p(X = x_i, Y = y_j)$$
  - Note that $p(X = x_i)$ is called the **marginal probability** because it is obtained by marginalizing or summing out the other variables (in this case Y)

- If we consider only those instances for which $X = x_i$, then the fraction of
such instances for which $Y = y_j$ is written $p(Y = y_j |X = x_i)$ and is called the
**conditional probability** of $Y = y_j$ given $X = x_i$. It is obtained by finding the
fraction of those points in column i that fall in cell i,j and hence is given by

$$ p(Y = y_j |X = x_i) = \frac{n_{ij}}{c_i} $$

- **product rule of probability** $$p(X = x_i, Y = y_j) = \frac{n_{ij}}{N} = \frac{n_{ij}}{c_i} \cdot \frac{c_i}{N} = p(Y=y_j|X=x_i)p(X = x_i)$$

- More compact notation,
  - sum rule $$p(X) = \sum_Y p(X,Y)$$
  - product rule $$p(X,Y) = p(Y|X)p(X)$$
  - $p(X,Y) = p(Y,X)$ so, we can derive **Bayes' theorem** $$p(X,Y) = p(Y|X)p(X) = p(X|Y)p(Y)$$ $$\therefore p(Y|X) = \frac{p(X|Y)p(Y)}{p(X)}$$

- Using the sum rule, the denominator in Bayes’ theorem can be expressed in terms of the quantities appearing in the numerator $$p(X) = \sum_Y p(X,Y) = \sum_Y p(X|Y)p(Y)$$ 
  - We can view the denominator in Bayes’ theorem as being the normalization constant
required to ensure that the sum of the conditional probability on the left-hand side of over all values of $Y$ equals one. 

- if $x$ and $y$ are two real variables, then the sum and product rules take the form $$p(x) = \int p(x, y) dy$$ $$p(x, y) = p(y|x)p(x)$$

---
toc: true
layout: post
description: Pattern Recognition and Machine Learning Summary
categories: [Machine Learning]
title: Pattern Recognition and Machine Learning Summary
---

## 1.2 Probability Theory

The probability that $X$ will take the value $x_i$ and $Y$ will take the value $y_j$ is
written $p(X = x_i, Y = y_j)$ and is called the **joint probability** of $X = x_i$ and
$Y = y_j$. It is given by the number of points falling in the cell $i,j$ as a fraction of the
total number of points, and hence  

$$p(X = x_i, Y = y_j) = \frac{n_{ij}}{N} $$  

$$p(X = x_i) = \frac{c_i}{N} $$

- **sum rule of probability** 

$$p(X = x_i) = \sum_{j=1}^{L} p(X = x_i, Y = y_j)$$

 - Note that $p(X = x_i)$ is called the **marginal probability** because it is obtained by marginalizing or summing out the other variables (in this case Y)

If we consider only those instances for which $X = x_i$, then the fraction of
such instances for which $Y = y_j$ is written $p(Y = y_j \mid X = x_i)$ and is called the
**conditional probability** of $Y = y_j$ given $X = x_i$. It is obtained by finding the
fraction of those points in column $i$ that fall in cell $i,j$ and hence is given by 

$$p(Y = y_j  \mid X = x_i) = \frac{n_{ij}}{c_i} $$

- **product rule of probability** 

$$p(X = x_i, Y = y_j) = \frac{n_{ij}}{N} = \frac{n_{ij}}{c_i} \cdot \frac{c_i}{N} = p(Y=y_j \mid X=x_i)p(X = x_i)$$

More compact notation,
- sum rule 

$$p(X) = \sum_Y p(X,Y)$$

- product rule 

$$p(X,Y) = p(Y \mid X)p(X)$$

- $p(X,Y) = p(Y,X)$ so, we can derive **Bayes' theorem** 

$$p(X,Y) = p(Y \mid X)p(X) = p(X \mid Y)p(Y)$$ 

$$\therefore p(Y \mid X) = \frac{p(X \mid Y)p(Y)}{p(X)}$$

- Using the sum rule, the denominator in Bayes’ theorem can be expressed in terms of the quantities appearing in the numerator 

$$p(X) = \sum_Y p(X,Y) = \sum_Y p(X \mid Y)p(Y)$$ 

- We can view the denominator in Bayes’ theorem as being the normalization constant
required to ensure that the sum of the conditional probability on the left-hand side of over all values of $Y$ equals one. 

### 1.2.1 Probability densities

If $x$ and $y$ are two real variables, then the sum and product rules take the form 

$$p(x) = \int p(x, y) dy$$ 

$$p(x, y) = p(y \mid x)p(x)$$

### 1.2.2 Expectations and covariances

The average value of some function $f(x)$ under a probability distribution $p(x)$ is called the **expectation of f(x)** and will be denoted by 

$$E[f] = \sum_x p(x) f(x) \space \space or \space \space E[f] = \int p(x) f(x) dx $$

- If we are given a finite number $N$ of points drawn from the probability distribution or probability density, then the expectation can be approximated as a finite sum over these points 

$$E[f] \sim \frac{1}{N} \sum_{n=1}^{N}f(x_n) $$

- $E_x[f(x, y)]$ will be a function of y, **conditional expectation** with respect to a conditional distribution, 

$$E_x[f \mid y] = \sum_x p(x \mid y)f(x)$$

The variance of $f(x)$ is defined by 

$$var[f] = E[(f(x) - E[f(x)])^2] = E[f(x)^2] - E[f(x)]^2$$

- and provides a measure of how much variability there is in $f(x)$ around its mean
value $E[f(x)]$. 

- In particular, we can consider the variance of the variable x itself, which is given by 

$$var[x] = E[x^2] - E[x]^2 $$

- For two random variables $x$ and $y$, the covariance is defined by 

$$cov[x,y] = E_{x,y}[(x-E[x])(y-E[y])] = E_{x,y}[xy] - E[x]E[y]$$

- If x and y are independent, then their covariance vanishes(become 0).
  
- In the case of two vectors of random variables x and y, the covariance is a matrix 

$$cov[\textbf{x},\textbf{y}] = E_{\textbf{x},\textbf{y}} [(\textbf{x} - E[\textbf{x}])(\textbf{y}^T -E[\textbf{y}^T])] = E_{\textbf{x},\textbf{y}}[\textbf{x}\textbf{y}^T] - E[\textbf{x}]E[\textbf{y}^T]$$

### 1.2.3 Bayesian probabilities

Bayes’ theorem, which takes the form

$$p(\textbf{w} \mid D) = \frac{p(D \mid \textbf{w})p(\textbf{w})}{p(D)} $$

- then **allows us to evaluate the uncertainty in w after we have observed D in the form
of the posterior probability $p(\textbf{w} \mid D)$**

- The quantity $p(D \mid \textbf{w})$ on the right-hand side of Bayes’ theorem is evaluated for the observed data set $D$ and can be viewed as a function of the parameter vector $\textbf{w}$, in which case it is called the **likelihood function**. **It expresses how probable the observed data set is for different settings of the parameter vector $\textbf{w}$**. Note that the likelihood is not a probability distribution over w, and its integral with respect to $\textbf{w}$ does not (necessarily) equal one.

- Given this definition of likelihood, we can state Bayes’ theorem in words

$$posterior \propto likelihood \times piror $$

- where all of these quantities are viewed as functions of $\textbf{w}$. The denominator is the normalization constant, which ensures that the posterior distribution on the left-hand side is a valid probability density and integrates to one.

$$p(D) = \int p(D \mid \textbf{w})p(\textbf{w})d \textbf{w} $$

-  In a frequentist setting, $\textbf{w}$ is considered to be a fixed parameter, whose value is determined by some form of ‘estimator’, and error bars on this estimate are obtained by considering the distribution of possible data sets $D$

- By contrast, from the Bayesian viewpoint there is only a single data set $D$ (namely
the one that is actually observed), and the uncertainty in the parameters is expressed
through a probability distribution over $\textbf{w}$. 

### 1.2.4 The Gaussian distribution

For the case of a single real-valued variable $x$, the Gaussian distribution is defined by

$$N(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi} \sigma}exp(-\frac{1}{2}(\frac{x-\mu}{\sigma})^2) $$

- which is governed by two parameters: $\mu$, called the **mean**, and $\sigma^2$, called the **variance**. The square root of the variance, given by $\sigma$, is called the standard deviation, and the reciprocal of the variance, written as $\beta = 1 / \sigma^2$, is called the **precision**.

$$E[x] = \int_{-\infty}^{\infty} N(x \mid \mu, \sigma^2)x dx = \mu$$

$$E[x^2] = \int_{-\infty}^{\infty} N(x \mid \mu, \sigma^2)x^2 dx = \mu^2 + \sigma^2$$

- (proof) we make the change of variables $u = x - \mu$, and get

$$E[x] =  \int_{-\infty}^{\infty} \frac{1}{\sqrt{2 \pi} \sigma}exp(-\frac{1}{2}(\frac{u}{\sigma})^2)(\mu + u) du$$

$$E[x] =  \int_{-\infty}^{\infty} \frac{1}{\sqrt{2 \pi} \sigma}exp(-\frac{1}{2}(\frac{u}{\sigma})^2)\mu du + \int_{-\infty}^{\infty} \frac{1}{\sqrt{2 \pi} \sigma}exp(-\frac{1}{2}(\frac{u}{\sigma})^2)u du$$

- We see that the first term of the right hand side of the equation is equal to $\mu$ because, it is the normal distribution’s normalization condition multiplied with a constant $\mu$. And we can see by inspection that the second term vanishes because the integrand is an odd function evaluated in $(-\infty,\infty)$, which is the multiplication of an even function ( $\exp({\frac{-u^2}{2\sigma^2}})$ ) and an odd function ( $u$ ). 

$$E[x] = u + 0$$

- Also, $E[x^2] = \sigma^2 + \mu^2$ can be proved using $var[x] = E[x^2] - E[x]^2$

Gaussian distribution defined over a D-dimensional vector x of continuous variables, which is given by

$$N(\mathbf{x} \mid \mathbf{\mu}, \mathbf{\Sigma}) = \frac{1}{\sqrt{(2 \pi)^D \lvert \mathbf{\Sigma} \rvert}} exp (-\frac{1}{2}(\mathbf{x} - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu}))$$

- where the D-dimensional vector $\mathbf{\mu}$ is called the mean, the $D \times D$ matrix $\mathbf{\Sigma}$ is called the covariance, and $\lvert \mathbf{\Sigma} \rvert$ denotes the determinant of $\Sigma$

 - Data points that are drawn independently from the same distribution are said to be **independent and identically distributed**, which is often abbreviated to i.i.d. We have
seen that the joint probability of two independent events is given by the product of the marginal probabilities for each event separately. Because our data set $\mathbf{x}$ is i.i.d.,
we can therefore write the probability of the data set, given $\mu$ and $\sigma^2$, in the form

$$p(\mathbf{x} \mid \mu, \sigma^2) = \prod_{n=1}^N N(x_n \mid \mu, \sigma^2)$$

- When viewed as a function of $\mu$ and $\sigma^2$, this is the likelihood function for the Gaussian

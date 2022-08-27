---
toc: true
layout: post
description: From nothing to gaussian process
categories: [gaussian process]
title: From nothing to gaussian process (아무 지식이 없는 상태에서 Gaussian Process 까지 이해)
---


## 1. Conditional probability
> In probability theory, **[conditional probability](https://en.wikipedia.org/wiki/Conditional_probability)** is a measure of the probability of an event occurring, given that another event (by assumption, presumption, assertion or evidence) has already occurred.

$$P(A | B) = \frac{P( A \cap B)}{P(B)}$$

 - Independence(독립) : Two events A and B are independent if and only if their joint probability equals the product of their probabilities

$$P(A \cap B) = P(A) \cdot P(B)$$

 - 조건부 확률에서 P(A)와 P(B)가 독립인 경우, 
$P(A | B) = \frac{P( A \cap B)}{P(B)} = \frac{P(A) \cdot P(B)}{P(B)} = P(A) $
가 성립하며, 조건부 확률로 사건A에 대해 사건B가 주어지는 경우와 주어지지 않는 경우의 확률이 같은 경우를 의미하는 것으로 이해할 수 있다. 또한 다른말로 표현해보면, 전체에서 A가 발생할 확률과 사건B가 발생했을 때 사건A가 발생할 확률이 같은 경우를 의미하는 것으로도 이해할 수 있다.  

## 2. Bayes's Theorem
 - Bayes' theorem is stated mathematically as the following equation:

$$ P(A|B) = \frac {P(B|A) \cdot P(A)} {P(B)}, where \space P(B)\neq 0. $$

 - Bayes's Theorem은 conditonal probability로 부터 유도됩니다.

$$ P(A | B) = \frac{P( A \cap B)}{P(B)}, \space then \space P(A \cap B) = P(A | B) \cdot P(B) $$

$$ P(B | A) = \frac{P( A \cap B)}{P(A)}, \space then \space P(A \cap B) = P(B | A) \cdot P(A) $$

$$ so, P(A | B) \cdot P(B) = P(B | A) \cdot P(A) $$

## 3. Bayes's Theorem 예제
우리나라 사람이 폐암에 걸릴 확률은 3%이고, 폐암을 99% 양성으로 진단하는 시약이 있다. 이 시약으로 폐암을 진단했을 때 양성반응을 보인 경우,
실제 폐암에 걸렸을 확률은 얼마인가?

 - 우리가 구해야할 확률은 P(폐암|양성)이고, 문제에서 주어진 조건은, 

$$P(폐암) = 0.03 \space (=> P(정상) = 0.97), $$

$$P(양성|폐암) = 0.99 \space (=> P(양성|정상) = 0.01)$$

 - Bayes's theorem을 이용하면, 
$$P(폐암|양성) = P(양성|폐암) \cdot P(폐암) / P(양성)$$
$$P(양성) = P(양성|정상) \cdot P(정상) + P(양성|폐암) \cdot P(폐암)$$
이며, 이를 계산하면, 
$P(양성) = 0.01 * 0.97 + 0.99 * 0.03 = 0.03939109$
 - 따라서 
$P(폐암|양성) = 0.99 * 0.03 / 0.03939109 = 0.7539776127037866$
이며, 약 75%임.

## 4. Maximum likelihood estimation(MLE)
> In statistics, **[maximum likelihood estimation (MLE)](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)** is a method of estimating the parameters of an assumed probability distribution, given some observed data. This is achieved by maximizing a likelihood function so that, under the assumed statistical model, the observed data is most probable. The point in the parameter space that maximizes the likelihood function is called the maximum likelihood estimate. 
> -- From wikipedia

 - MLE는 주어진 관측데이터(
$x_1, x_2, \dots, x_n$
)에 대해, 어떠한 분포를 가정했을 때, 관측데이터와 가장 유사한 분포를 만드는 파라메터(
$\theta$
)를 추정하는 방법입니다.
 - We pick the distribution family p(·), but don’t know the parameter θ
 - MLE를 수식으로 표현하면 아래와 같습니다.

$$\hat \theta_{MLE} = \underset{\theta}{\operatorname{argmax}} p(x_1, x_2, \dots, x_n|\theta)$$

- Assume data is independent and identically distributed (iid). This is written

$$ x_i \overset{i.i.d}{\sim} p(x|\theta), i = 1, \dots, n$$

- Writing the density as p(x|θ), then the joint density decomposes as

$$ p(x_1, x_2, \dots, x_n|\theta) = \prod_{i=1}^n p(x_i | \theta) $$

- 그리고 다음과 같이 Maximum Likelihood가 되는 파라메터(θ)를 추정할 수 있습니다. 

$$ \nabla_{\theta} p(x_1, x_2, \dots, x_n|\theta) = \nabla_{\theta} \prod_{i=1}^n p(x_i | \theta) = 0 $$

- Logarithm tric : It is complicated to calcuate it directly. So we use the fact that the logarithm is monotonically increasing on R+, and the equality

$$\underset{\theta}{\operatorname{argmax}} g(y) = \underset{\theta}{\operatorname{argmax}} ln(g(y))$$

$$ ln(\prod_i f_i) = \sum_i ln(f_i) $$

 - 데이터 특징에 따라 선택할 수 있는 분포들은 binomial, poisson, gaussian 등이 있습니다. 

## 5. MLE 예제
 - 공장에서 10개의 제품을 검사했을 때, 정상이 8개, 불량이 2개인 경우가 관측되었습니다. 이 경우, 우리는 binomial distribution을 가정할 수 있습니다. binomial distribution의 pmf는 다음과 같이 정의됩니다.

$$ \Pr(K = k) = f(k;n,\theta)={n\choose k}\theta^k(1-\theta)^{n-k} $$ 

 - 파라메터 θ를 MLE로 추정해보면,

$$ \nabla_{\theta} \prod_{i=1}^n p(x_i | \theta) = \nabla_{\theta} ln(\prod_{i=1}^n p(x_i | \theta)) = \nabla_{\theta} ln({n\choose k}\theta^k(1-\theta)^{n-k}) = 0$$

$$ \nabla_{\theta} ln(\theta^k(1-\theta)^{n-k}) = \nabla_{\theta} k \cdot ln(\theta) + (n-k) \cdot ln(1-\theta) = \displaystyle \frac{k}{\theta} - \frac{n-k}{1- \theta} = 0$$

$$ k(1-\theta) - \theta(n-k) = 0 $$

$$ \theta = \displaystyle \frac{k}{n} $$

- MLE를 통해 불량확률에 파라메터는 θ는, n = 10, k = 2인 경우, 0.2로 추정할 수 있습니다.
- multivariate gaussian distribution에 MLE를 적용하면, mean과 covariance는 각각 아래와 같습니다.(계산유도과정은 생략합니다)

$$ \hat \mu_{MLE} = \frac{1}{n}\sum_{i=1}^n x_i, \space \hat \sigma_{MLE} = \frac{1}{n}\sum_{i=1}^n (x_i - \hat \mu_{MLE})(x_i - \hat \mu_{MLE})^T$$

- If 
$x_1, \dots x_n$
don’t “capture the space” well, 
$\theta_{MLE}$
can overfit the data.
- 관측데이터에 overfitting되지 않는 파라메터(모수) 추정 방법에 대한 문제를 제기하는 것 같습니다. 아마도 뒤에서 Bayesian 방법론을 적용할 듯합니다.

## 6. Univariate and Multivariate Gaussian Distribution
> In statistics, a **[normal distribution](https://en.wikipedia.org/wiki/Normal_distribution)** (also known as Gaussian, Gauss, or Laplace–Gauss distribution) is a type of continuous probability distribution for a real-valued random variable. The general form of its probability density function is

$$ f(x) = \frac {1}{\sigma \sqrt{2\pi}} e ^ {-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$


- random variable X is normally distributed with mean 
$\mu$
and standard deviation
$\sigma$
, one may write
$\displaystyle X\sim {\mathcal {N}}(\mu ,\sigma ^{2})$

$$ $$

- The multivariate normal distribution of a k-dimensional random vector
$X =(X_{1},\dots ,X_{k})^{T}$
can be written in the following notation:

$$ X \sim {\mathcal {N}}_{k} ({\boldsymbol {\mu }},{\boldsymbol {\Sigma }}),$$

- with k-dimensional mean vector

$${\boldsymbol {\mu }=\operatorname {E} [\mathbf {X} ]=(\operatorname {E} [X_{1}],\operatorname {E} [X_{2}],\ldots ,\operatorname {E} [X_{k}])^{\textbf {T}},}$$

- and k x k covariance matrix

$$ \Sigma_{i,j}=\operatorname {E} [(X_{i}-\mu _{i})(X_{j}-\mu _{j})]=\operatorname {Cov} [X_{i},X_{j}], $$

- such that
$1 \leq i \leq k$
and 
$1 \leq j \leq k$

## 7. Covariance의 의미
- Variance(분산) : 데이터가 펼쳐진 정도, 분산이 작으면, 데이터가 좁은영역에 모여있고, 분산이 크면 데이터가 넓은 영역에 퍼지는 형태를 보임

$$\operatorname {var} (\mathbf {X} )=\operatorname {cov} (\mathbf {X} ,\mathbf {X} )=\operatorname {E} \left[(\mathbf {X} -\operatorname {E} [\mathbf {X} ])(\mathbf {X} -\operatorname {E} [\mathbf {X} ])^{\rm {T}}\right]$$

- Covariance(공분산) : 두 변수간 데이터가 퍼진 정도를 나타냄

$$ \operatorname {cov} (\mathbf {X} ,\mathbf {Y} )=\operatorname {E} \left[(\mathbf {X} -\operatorname {E} [\mathbf {X} ])(\mathbf {Y} -\operatorname {E} [\mathbf {Y} ])^{\rm {T}}\right] $$

- Covariance를 의미를 살펴보면,

## 8. Linear Regression

## 8. Probability View of Linear Regression

## 9. MAP(

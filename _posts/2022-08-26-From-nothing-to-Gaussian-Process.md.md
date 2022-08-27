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

## 3. Maximum likelihood estimation(MLE)
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

- Logarithm tric : It is complicated to calcuate it directly. So we use the fact that the logarithm is monotonically increasing on R+, and the equality(
$\underset{\theta}{\operatorname{argmax}} g(y) = \underset{\theta}{\operatorname{argmax}} ln(g(y))$
)

$$ ln(\prod_i f_i) = \sum_i ln(f_i) $$

 - 데이터 특징에 따라 선택할 수 있는 분포들은 binomial, poisson, gaussian 등이 있습니다. 

## 2. MLE 예제
 - 공장에서 10개의 제품을 검사했을 때, 정상이 8개, 불량이 2개인 경우가 관측되었습니다. 이 경우, 우리는 binomial distribution을 가정할 수 있습니다. binomial distribution의 pmf는 다음과 같이 정의됩니다.

$$ \Pr(K = k) = f(k;n,\theta)={n\choose k}\theta^k(1-\theta)^{n-k} $$ 

 - 파라메터 θ를 MLE로 추정해보면,

$$ \nabla_{\theta} \prod_{i=1}^n p(x_i | \theta) = \nabla_{\theta} ln(\prod_{i=1}^n p(x_i | \theta)) = \nabla_{\theta} ln({n\choose k}\theta^k(1-\theta)^{n-k}) = 0$$

$$ \nabla_{\theta} ln(\theta^k(1-\theta)^{n-k}) = \nabla_{\theta} k \cdot ln(\theta) + (n-k) \cdot ln(1-\theta) = \displaystyle \frac{k}{\theta} - \frac{n-k}{1- \theta} = 0$$

$$ k(1-\theta) - \theta(n-k) = 0 $$

$$ \theta = \displaystyle \frac{k}{n} $$

- MLE를 통해 불량확률에 파라메터는 
$\displaystyle \frac{k}{n}$
로 계산되며, n = 10, k = 2인 경우, 
$\theta$
는 0.2로 추정됩니다.



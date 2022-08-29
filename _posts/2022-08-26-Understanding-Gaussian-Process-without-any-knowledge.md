---
toc: true
layout: post
description: Understanding Gaussian Process without any knowledge
categories: [gaussian process]
title: Understanding Gaussian Process without any prior knowledge (아무 지식이 없는 상태에서 Gaussian Process 까지 이해해보기)
---

## 0. Basic concepts

- Sample space: the sample space of an experiment or random trial is the **set of all possible outcomes or results of that experiment**. An event is a subset of the sample space. 동전을 던졌을 때 나올 수 있는 결과는 앞면(H) 또는 뒷면(T)임, 발생할 수 있는 모든 경우의 수를 의미함.
 
- Random Variable: A Random Variable is a mathematical formalization of a quantity or object which depends on random events. It is a mapping or a function from possible outcomes in a sample space to a measurable space(아마도 확률이라는 값을 의미하는 것이 아닐지..., 즉, random variable이란 sample space(모든경우의 수)와 probability(확률값, 발생가능한 값들의 영역의 넓이?)의 맵핑이라는 것으로 해석 됨), often the real numbers.

  - Randome experiment 통해 Random variable은 어떠한 event로 realization됨. 결국 확률값은 event가 발생하기 이전에 가지는 값임. 이벤트가 발생하면 관측이 되는 것이고...

- Probability: The probability of an event  A  is the sum of the probabilities of the individual outcomes of which it is composed. It is denoted  P(A).
 
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

- 우리나라 사람이 폐암에 걸릴 확률은 3%이고, 폐암을 99% 양성으로 진단하는 시약이 있다. 이 시약으로 폐암을 진단했을 때 양성반응을 보인 경우,
실제 폐암에 걸렸을 확률은 얼마인가?

$$ $$

- 우리가 구해야할 확률은 
$P(폐암|양성)$
이고, 문제에서 주어진 조건은, 

$$P(폐암) = 0.03 \space (\Rightarrow P(정상) = 0.97), $$

$$P(양성|폐암) = 0.99 \space (\Rightarrow P(양성|정상) = 0.01)$$

$$ $$

- Bayes's theorem을 이용하면, 

$$P(폐암|양성) = P(양성|폐암) \cdot P(폐암) / P(양성)$$

$$P(양성) = P(양성|정상) \cdot P(정상) + P(양성|폐암) \cdot P(폐암)$$

- 이며, 이를 계산하면, 
$P(양성) = 0.01 * 0.97 + 0.99 * 0.03 = 0.03939109$

$$ $$

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

- Writing the density as 
$p(x|θ)$
, then the joint density decomposes as

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

## 6. Gaussian Distribution

### 6-1. Univariate Gaussian Distribution

- In statistics, a **[normal distribution](https://en.wikipedia.org/wiki/Normal_distribution)** (also known as Gaussian, Gauss, or Laplace–Gauss distribution) is a type of continuous probability distribution for a real-valued random variable. The general form of its probability density function is

$$ f(x) = \frac {1}{\sigma \sqrt{2\pi}} e ^ {-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$

   - random variable X is normally distributed with mean 
$\mu$
and standard deviation
$\sigma$
, one may write
$\displaystyle X\sim {\mathcal {N}}(\mu ,\sigma ^{2})$

$$ $$

### 6-2. Multivariate Gaussian Distribution

- The **[multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)** of a k-dimensional random vector
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

$$ $$

- The general form of its probability density function is

$$ {\displaystyle f_{\mathbf {X} }(x_{1},\ldots ,x_{k})={\frac {\exp \left(-{\frac {1}{2}}({\mathbf {x} }-{\boldsymbol {\mu }})^{\mathrm {T} }{\boldsymbol {\Sigma }}^{-1}({\mathbf {x} }-{\boldsymbol {\mu }})\right)}{\sqrt {(2\pi )^{k}|{\boldsymbol {\Sigma }}|}}}} $$

- (참고) Mahalanobis distance
> The Mahalanobis distance is a measure of the distance between a point P and a distribution D

$$ \sqrt{({\mathbf {x} }-{\boldsymbol {\mu }})^{\mathrm {T} }{\boldsymbol {\Sigma }}^{-1}({\mathbf {x} }-{\boldsymbol {\mu }})} $$

- 만약,
$\Sigma = \sigma^2 I$
이라면, 즉, 각 변수간 공분산이 모두 0인 경우, 마할라노비스 거리는 유클리디안 거리와 같아집니다.

## 7. Covariance의 의미

- Variance(분산) : 데이터가 펼쳐진 정도, 분산이 작으면, 데이터가 좁은영역에 모여있고, 분산이 크면 데이터가 넓은 영역에 퍼지는 형태를 보임

$$\operatorname {var} (\mathbf {X} )=\operatorname {Cov} (\mathbf {X} ,\mathbf {X} )=\operatorname {E} \left[(\mathbf {X} -\operatorname {E} [\mathbf {X} ])(\mathbf {X} -\operatorname {E} [\mathbf {X} ])^{\rm {T}}\right]$$

- Covariance(공분산) : 두 변수간 데이터가 퍼진 정도를 나타냄

$$ \operatorname {Cov} (\mathbf {X} ,\mathbf {Y} )=\operatorname {E} \left[(\mathbf {X} -\operatorname {E} [\mathbf {X} ])(\mathbf {Y} -\operatorname {E} [\mathbf {Y} ])^{\rm {T}}\right] $$

- Covariance를 의미를 살펴보면,

$$ Cov(X, Y) = \frac{(x_1 - \mu_x)(y_1 - \mu_y) + (x_2 - \mu_x)(y_2 - \mu_y) + \dots + (x_n - \mu_x)(y_n - \mu_y)}{n} $$

- 이고, 
$(x_i - \mu_x)(y_i - \mu_y)$
가 양수인 경우는 각각 평균보다 크거나, 각각 평균보다 작은 경우이며, 음수인 경우 그 반대이다. 또한 평균으로부터 값이 멀어질 수록 그 값이 커지게 된다. 즉, 각 변수의 평균을 중심으로 분산의 방향을 확인할 수 있다. 데이터를 축에 plotting했을 때, Cov(X,Y)가 양수이면, 1,3분면에 분포하며, 음수이면 주로 2,4분면에 분포할 것으로 생각할 수 있다. 다만, 공분산은 각 변수의 단위에 따라 값의 크기가 달라져서 절대적인 값의 크기로 비교하는 것은 타당하지 않다.

$$ $$
- 그래서, 공분산을 각각의 표준편차로 나누어 그 값을 [-1, 1]로 변환하여 계산한 것을 correlation(상관계수)라고 한다.

$$ {\displaystyle \rho _{X,Y}=\operatorname {corr} (X,Y)={\operatorname {Cov} (X,Y) \over \sigma _{X}\sigma _{Y}}={\operatorname {E} [(X-\mu _{X})(Y-\mu _{Y})] \over \sigma _{X}\sigma _{Y}},\quad {\text{if}}\ \sigma _{X}\sigma _{Y}>0.} $$

## 8. Linear Regression

### 8.1 Problem definition

- Data: Measured pairs 
$(x,y)$
, where 
$x \in R^{d+1}$
(input) and 
$y \in R$
(output)

- Goal: Find a function
$f: R^{d+1} \rightarrow R$
such that 
$y \sim f(x;w)$
for the data pair 
$(x,y)$
. 
$f(x;w)$ 
is the **regression function** and the vector 
$w$
are its paramenters.

- Definition of linear regression: A regression method is called linear if the prediction 
$f$
is a linear function of the unknown parameters 
$w$
.

### 8.2 Least squares solution
- Least squares finds the w that minimizes the sum of squared errors. The least squares objective in the most basic form where 
$f(x;w) = x^Tw$
is

$$ L = \sum_{i=1}^n (y_i - x_i^T w)^2 = \lVert y-Xw \rVert ^2 = (y-Xw)^T(y-Xw)$$

- We defined 
$y = [y_1, \dots, y_n]^T \space$
and
$X = [x_1, \dots, x_n]^T
.

$$ w_{LS} = \underset{w}{\operatorname{argmin}} \sum_{i=1}^n (y_i - (w_0 + \sum_{j=1}^d x_{ij}w_j))^2 $$

- Taking the gradient with respect ot w and setting to zero, using vectors, this can be written:

$$ \nabla_{w} L = 0 \space \Rightarrow \space \sum_{i=1}^n \nabla_{w} (y_i^2 - 2w^T x_i y_i + w^T x_i x_i^T w) = 0 $$

- solving gives,

$$ - \sum_{i=1}^n 2 y_i x_i + (\sum_{i=1}^n 2 x_i x_i^T)w = 0 \space \Rightarrow \space w_{LS} = (\sum_{i=1}^n x_i x_i^T)^{-1}(\sum_{i=1}^n y_i x_i)$$

- solving gives as matrix version,

$$ w_{LS} = (X^T X)^{-1} X^T y $$

- In other words, 
$w_{LS}$
is the vector that minimizes
$L$
.

### 8.3 Maximum likelihood for Gaussian linear regression
- Assume a diagonal covariance matrix 
$\Sigma = \sigma^2I$
. The density is

$$ p(y|\mu, \sigma^2) = \frac{1}{(2 \pi \sigma^2)^{\frac{n}{2}}} exp(-\frac{1}{2 \sigma^2}(y-\mu)^T(y-\mu)) $$

- Plug
$\mu = Xw$
into the multivariate Gaussian distribution and solve for 
$w$
using maximum likelihood

$$ w_{ML} = \underset{w}{\operatorname{argmax}} ln \space p(y|\mu = Xw, \sigma^2)$$

$$ = \underset{w}{\operatorname{argmax}} -\frac{1}{2 \sigma^2} \lVert y - Xw \rVert^2 - \frac{n}{2}ln(2 \pi \sigma^2)$$

- Least squares(LS) and maximum likelihood(ML) share the same solution:

$$LS: \space \underset{w}{\operatorname{argmin}} \lVert y - Xw \rVert^2  \Leftrightarrow ML: \space \underset{w}{\operatorname{argmax}} -\frac{1}{2 \sigma^2} \lVert y - Xw \rVert ^ 2 $$

- therefore, in a sense we are making a *independent Gaussian noise assumption* about the error, 
$\epsilon_i = y_i - x_i^Tw$

- Other ways of saying this: \
  1)
$y_i = x_i^Tw + \epsilon_i, \space \epsilon_i \overset{i.i.d}{\sim} N(0, \sigma^2), \space for \space i = 1, \dots, n.$ \
  2)
$y_i \overset{ind}{\sim} N(x_i^Tw, \sigma^2), \space for \space i = 1, \dots, n.$ \
  3)
$y \sim N(Xw, \sigma^2I)$

## 9. Bayesian linear regression

### 9.1 Model

- Have vector
$y \in R^n$
and covariates matrix
$X \in R^{n \times d}$
. The *i*th row of 
$y$
and 
$X$
correspond to the *i*th observation
$(y_i, x_i)$

- In a Bayesian setting, we model this data as:

$$ Likelihood: \space y \sim N(Xw, \sigma^2I) $$

$$ Prior: \space w \sim N(0, \lambda^{-1}I) $$

  - Regarding prior distribution, although not covered here, see [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior).
  
- The unknow model variable is 
$w \in R^d$
  - The "likelihood model" says how well the observed data agrees with w.
  - The "model prior" is our prior belief (or constraints) on w.

- This is called Bayesian linear regression because we have defined a prior on the unknown parameter and will try to learn its posterior.

### 9.2 MAP(Maximum A Posteriori) solution

 - Let us assume that the prior for 
 $w$ 
 is Gaussian, 
 $w \sim N(0, \lambda^{-1}I)$
 . Then
 
 $$ p(w) = (\frac{\lambda}{2 \pi})^{\frac{d}{2}}e^{-\frac{\lambda}{2}w^Tw}$$
 
 - We can now try to find a 
 $w$
 that satisfies both the data likelihood, and our prior conditions about 
 $w$
 
 - Maximum a posteriori (MAP) estimation seeks the most probable value
 $w$
 under the posterior:
 
 $$ w_{MAP} = \underset{w}{\operatorname{argmax}} \space ln \space p(w|y,X) $$
 
 $$ = \underset{w}{\operatorname{argmax}} \space ln \frac{p(y|w,X_)p(w)}{p(y|X)} $$
 
 $$ = \underset{w}{\operatorname{argmax}} \space ln \space p(y|w,X) + ln \space p(w) - ln \space p(y|X) $$
 
 - The normalizing constant term 
 $ln \space p(y|X)$
 doesn't involve 
 $w$
 . Therefore, we can maximize the first two terms alone.

- In many models we don't know
$ln \space p(y|X)$
, so this fact is useful. 

- Hence,

$$ w_{MAP} = \underset{w}{\operatorname{argmax}} \space ln \space p(y|w,X) + ln \space p(w) $$

$$ = \underset{w}{\operatorname{argmax}} \space - \frac{1}{2 \sigma^2}(y - Xw)^T(y-Xw) - \frac{\lambda}{2}w^Tw + const $$

- this solution for 
$w_{MAP}$
is the same as for ridge regression (we do not cover ridge regression(RR) here).

$$ w_{MAP} = (\lambda \sigma^2I + X^TX)^{-1}X^Ty \space \Leftrightarrow \space w_{RR}$$

### 9.3 Point estimates

- $w_{MAP}$ and $w_{ML}$ are referred to as point estimates of the model parameters.
- The find a specific value(point) of the vector $w$ that maximizes an objective function (MAP or ML)
  - ML: Only consider data model
  - MAP: Takes into account model prior
- Bayesian inference goes one step further by characterizing uncertainty about the values in w using Bayes rule.

- In posterior calculation, we get an updated distribution on $w$ through the transition

$$ prior \rightarrow likelihood \rightarrow posterior $$

- Bayesian learning is naturally thought of a sequential process. That is, the posterior after seeing some data becomes the prior for the next data.

- Maximum likelihood는 데이터가 주어졌을 때, 이를 잘 적합하는 w를 찾는 것이라면, MAP는 w에 대한 prior distribution(사전정보)이 있다고 가정하고, 새로운데이터가 입력될때마다 likelihood와 prior를 반복적으로 업데이트하여 posteriori를 계산하는 방법이다.
 
## 10. Random process

## 11. Gaussian process

- A **random process** X(t) is a Gaussian process if for all k ∈ N for all t1, ... ,tk , a random vector formed by X(1), ... , X(tk) is jointly Gaussian.
- The joint density is completely specified by
 - Mean: m(t) = E(X(t)), where m(·) is known as a mean function.
 - Covariance: k(t, s) = Cov(X(t), X(s)), where k(·,·) is known as a covariance function.
- Notation:

$$X(t) \sim GP(m(t), k(t,s))$$ 

- Example: X(t) = tA, where 
$A \sim N(0,1)$
and t ∈ R
  - Mean: m(t) = E(X(t)) = tE(A) = 0
  - Covariance: k(t,s) = E(tAsA) = ts

- **Gaussian process** and **Gaussian process regression** are different.

## 12. Gaussian process regression

- A nonparametric **Bayesian regression** method using the properties of **Gaussian processes**.
- Two views to interpret Gaussian process regression
  - Weight-space view
  - Function-space view

# 12.1 Weight-space view

- 

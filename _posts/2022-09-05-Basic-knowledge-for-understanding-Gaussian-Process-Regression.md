---
toc: true
layout: post
description: 
categories: [Statistics]
comments: true
author: Hyunho Lee
title: Gaussian Process Regression 이해를 위한 기초지식
---

## 0. 들어가며
Spatiotemporal Analysis를 수강하면서, Gaussian Process Regression 이해하기 위해 필요한 지식들을 정리한 글입니다.  
- 참고자료 : [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- 이 책의 chpater1 ~ 6을 읽고 Gaussian Process regression을 이해하기 위해 필요한 기초지식들을 다시 정리하였습니다. 

## 1. Probability

- Sample space: the sample space of an experiment or random trial is the **set of all possible outcomes or results of that experiment**. An event is a subset of the sample space. 동전을 던졌을 때 나올 수 있는 결과는 앞면(H) 또는 뒷면(T)임, 발생할 수 있는 모든 경우의 수를 의미함.
 
- Random Variable: A Random Variable is a mathematical formalization of a quantity or object which depends on random events. It is a mapping or a function from possible outcomes in a sample space to a measurable space, often the real numbers. (random variable이란 sample space(모든경우의 수)와 probability(확률값, 발생가능한 값들의 영역의 넓이)의 맵핑(함수)으로 이해함)

- Probability: The probability of an event  A  is the sum of the probabilities of the individual outcomes of which it is composed. It is denoted  P(A).
 
## 2. Conditional probability

> In probability theory, **[conditional probability](https://en.wikipedia.org/wiki/Conditional_probability)** is a measure of the probability of an event occurring, given that another event (by assumption, presumption, assertion or evidence) has already occurred.

$$P(A \mid B) = \frac{P( A \cap B)}{P(B)}$$

> Independence(독립) : Two events A and B are independent if and only if their joint probability equals the product of their probabilities

$$P(A \cap B) = P(A) \cdot P(B)$$

- 조건부 확률에서 P(A)와 P(B)가 독립인 경우, $P(A \mid B) = \frac{P( A \cap B)}{P(B)} = \frac{P(A) \cdot P(B)}{P(B)} = P(A)$가 성립하며, 조건부 확률로 사건A에 대해 사건B가 주어지는 경우와 주어지지 않는 경우의 확률이 같은 경우를 의미하는 것으로 이해할 수 있다. 또한 다른 말로 표현해보면, 전체에서 A가 발생할 확률과 사건B가 발생했을 때 사건A가 발생할 확률이 같은 경우를 의미하는 것으로도 이해할 수 있다. (독립인 경우, $P(A) \cap P(B) \neq 0$임) 

## 3. Bayes's Theorem

 - Bayes' theorem is stated mathematically as the following equation:

$$ P(A \mid B) = \frac {P(B \mid A) \cdot P(A)} {P(B)}, where \space P(B)\neq 0. $$

 - Bayes's Theorem은 conditonal probability로 부터 유도됩니다.

$$ P(A \mid B) = \frac{P( A \cap B)}{P(B)}, \space then \space P(A \cap B) = P(A \mid B) \cdot P(B) $$

$$ P(B \mid A) = \frac{P( A \cap B)}{P(A)}, \space then \space P(A \cap B) = P(B \mid A) \cdot P(A) $$

$$ hence, P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A) $$

## 4. Bayes's Theorem 예제

- 우리나라 사람이 폐암에 걸릴 확률은 3%이고, 폐암을 99% 양성으로 진단하는 시약이 있다. 이 시약으로 폐암을 진단했을 때 양성반응을 보인 경우,
실제 폐암에 걸렸을 확률은 얼마인가?

- 구해야할 확률과 문제에서 주어진 확률을 구분해보면,
  - 우리가 구해야할 확률 : $P(폐암 \mid 양성)$
  - 문제에서 주어진 확률은, 

$$P(폐암) = 0.03 \space (\Rightarrow P(정상) = 0.97), $$

$$P(양성 \mid 폐암) = 0.99 \space (\Rightarrow P(양성 \mid 정상) = 0.01)$$

- $P(폐암 \mid 양성)$을 직접 계산할 수 없기 때문에, Bayes's theorem을 이용하면, 

$$P(폐암 \mid 양성) = P(양성 \mid 폐암) \cdot P(폐암) / P(양성)$$

$$P(양성) = P(양성 \mid 정상) \cdot P(정상) + P(양성 \mid 폐암) \cdot P(폐암)$$

- 이며, 이를 계산하면, $P(양성) = 0.01 * 0.97 + 0.99 * 0.03 = 0.03939109$

- 따라서 $P(폐암 \mid 양성) = 0.99 * 0.03 / 0.03939109 = 0.7539776127037866$
이며, 약 75%임.

## 5. Likelihood vs. Probability

- Likelihood는 가능도로 번역됩니다. 저에게 Probability와 Likelihood는 다소 헷갈리는 개념인데요, 저는 통계를 깊이있게 공부하는 사람은 아니기 때문에 이 지식을 사용하기 위한 목적으로만 있는 그대로 표현해보겠습니다. 
- Binomial distribution의 pmf를 예로 들겠습니다. 이 pmf는 n은 전체시행횟수, k는 이벤트발생횟수, $\theta$가 1회 시행시 발생확률을 매개변수로 하는 함수식입니다.

$$ \Pr(K = k) = f(k,n,\theta)={n\choose k}\theta^k(1-\theta)^{n-k} $$

- (Probability) 우리가 믿고 있는 1회 시행시 발생확률 $\theta$ 가 주어지고, 사건이 발생했을 때(n과 k가 관찰됨), pmf를 계산한 값입니다. 이 경우, pmf는 n가 k를 매개변수로 가지는 함수식이 되며, $\sum_k^{n} f(k,n,\theta) = 1$입니다. 다시 표현하면, $\theta$가 주어지고, data가 관찰되었을 때, data가 발생할 확률을 구하는 것이며, 발생가능한 모든 data의 확률의 합은 1이 됩니다.
  
- (Likelihood) 사건이 발생했을 때(n과 k가 관찰됨), 1회 시행시 발생확률 $\theta$ 에 따른 pmf를 계산한 값입니다. 이것은 $\theta$에 따른 사건의 발생가능도를 계산한 것으로 표현할 수 있습니다. 이 경우, pmf는 $\theta$를 매개변수로 가지는 함수식이 되며, $\sum_{\theta} f(k,n,\theta)$는 반드시 1은 아닙니다. 

- 최종 정리를 해보면, pmf에 대해 $\theta$가 주어지고, n과 k가 변수인 경우를 확률이라고 하며, 이때는 모든 n과 k에 대한 f(n,k)의 합은 1이됩니다. pmf에 대해 n과 k가 주어지고, $\theta$ 가 변수인 경유를 가능도라고 하며, 이때 모든 $\theta$에 대한 f( $\theta$ )의 합은 꼭 1이 되지 않습니다. 확률과 가능도는 동일한 pmf에 대해 어떤 매개변수를 pmf라는 함수의 변수로 볼 것인지에 따라 달라지는 개념이라고 최종적으로 이해했습니다.

## 6. Maximum likelihood estimation(MLE)

> In statistics, **[maximum likelihood estimation (MLE)](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)** is a method of estimating the parameters of an assumed probability distribution, given some observed data. This is achieved by maximizing a likelihood function so that, under the assumed statistical model, the observed data is most probable. The point in the parameter space that maximizes the likelihood function is called the maximum likelihood estimate. -- From wikipedia

- MLE는 주어진 관측데이터( $x_1, x_2, \dots, x_n$ )에 대해, 어떠한 분포를 가정했을 때, 관측데이터와 가장 유사한 분포를 만드는 파라메터( $\theta$ )를 추정하는 방법입니다.
- We pick the distribution family p(·), but don’t know the parameter θ
- MLE를 수식으로 표현하면 아래와 같습니다.

$$\hat \theta_{MLE} = \underset{\theta}{\operatorname{argmax}} p(x_1, x_2, \dots, x_n \mid \theta)$$

- Assume data is independent and identically distributed (iid). This is written

$$ x_i \overset{i.i.d}{\sim} p(x \mid \theta), i = 1, \dots, n$$

- Writing the density as $p(x \mid θ)$, then the joint density decomposes as

$$ p(x_1, x_2, \dots, x_n \mid \theta) = \prod_{i=1}^n p(x_i \mid \theta) $$

- 그리고 다음과 같이 Maximum Likelihood가 되는 파라메터(θ)를 추정할 수 있습니다. 

$$ \nabla_{\theta} p(x_1, x_2, \dots, x_n \mid \theta) = \nabla_{\theta} \prod_{i=1}^n p(x_i \mid \theta) = 0 $$

- Logarithm tric : It is complicated to calcuate it directly. So we use the fact that the logarithm is monotonically increasing on R+, and the equality

$$\underset{\theta}{\operatorname{argmax}} g(y) = \underset{\theta}{\operatorname{argmax}} ln(g(y))$$

$$ ln(\prod_i f_i) = \sum_i ln(f_i) $$

- 데이터 특징에 따라 선택할 수 있는 분포들은 binomial, poisson, gaussian 등이 있습니다.  

## 7. MLE 예제

- 공장에서 10개의 제품을 검사했을 때, 정상이 8개, 불량이 2개인 경우가 관측되었습니다. 이 경우, 우리는 binomial distribution을 가정할 수 있습니다. binomial distribution의 pmf는 다음과 같이 정의됩니다.

$$ \Pr(K = k) = f(k;n,\theta)={n\choose k}\theta^k(1-\theta)^{n-k} $$ 

- 파라메터 θ를 MLE로 추정해보면,

$$ \nabla_{\theta} \prod_{i=1}^n p(x_i \mid \theta) = \nabla_{\theta} ln(\prod_{i=1}^n p(x_i \mid \theta)) = \nabla_{\theta} ln({n\choose k}\theta^k(1-\theta)^{n-k}) = 0$$

$$ \nabla_{\theta} ln(\theta^k(1-\theta)^{n-k}) = \nabla_{\theta} k \cdot ln(\theta) + (n-k) \cdot ln(1-\theta) = \displaystyle \frac{k}{\theta} - \frac{n-k}{1- \theta} = 0$$

$$ k(1-\theta) - \theta(n-k) = 0 $$

$$ \theta = \displaystyle \frac{k}{n} $$

- MLE를 통해 불량확률에 파라메터는 θ는, n = 10, k = 2인 경우, 0.2로 추정할 수 있습니다.
- multivariate gaussian distribution에 MLE를 적용하면, mean과 covariance는 각각 아래와 같습니다.(계산유도과정은 생략합니다)

$$ \hat \mu_{MLE} = \frac{1}{n}\sum_{i=1}^n x_i, \space \hat \sigma_{MLE} = \frac{1}{n}\sum_{i=1}^n (x_i - \hat \mu_{MLE})(x_i - \hat \mu_{MLE})^T$$

- MLE는 데이터를 관찰하고, 관찰된 데이터를 가장 잘 표현하는 매개변수( $\theta$ )를 추정하는 방식입니다. 하지만, 관찰된 데이터는 전체 모집단을 잘 표현할 수 있는 데이터여야 할 것입니다. 그렇지 않다면, MLE로 추정된 변수로 만들어진 모델은 overfitting될 가능성이 높습니다. MLE는 빈도주의자들의 방식이라고 합니다.
  - If $x_1, \dots x_n$ don’t “capture the space” well, $\theta_{MLE}$ can overfit the data. 

## 8. Gaussian Distribution

- 분포가 가지는 계산의 편리함으로 인해, Bayesian 방법론에서는 Gaussian distribution을 많이 다룹니다. 

### 8-1. Univariate Gaussian Distribution

- In statistics, a **[normal distribution](https://en.wikipedia.org/wiki/Normal_distribution)** (also known as Gaussian, Gauss, or Laplace–Gauss distribution) is a type of continuous probability distribution for a real-valued random variable. The general form of its probability density function is

$$ f(x) = \frac {1}{\sigma \sqrt{2\pi}} e ^ {-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$

   - random variable X is normally distributed with mean $\mu$ and standard deviation $\sigma$, one may write $\displaystyle X\sim {\mathcal {N}}(\mu ,\sigma ^{2})$

### 8-2. Multivariate Gaussian Distribution

- The **[multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)** of a k-dimensional random vector $X =(X_{1},\dots ,X_{k})^{T}$ can be written in the following notation:

$$ X \sim {\mathcal {N}}_{k} ({\boldsymbol {\mu }},{\boldsymbol {\Sigma }}),$$

- with k-dimensional mean vector

$${\boldsymbol {\mu }=\operatorname {E} [\mathbf {X} ]=(\operatorname {E} [X_{1}],\operatorname {E} [X_{2}],\ldots ,\operatorname {E} [X_{k}])^{\textbf {T}},}$$

- and k x k covariance matrix

$$ \Sigma_{i,j}=\operatorname {E} [(X_{i}-\mu _{i})(X_{j}-\mu _{j})]=\operatorname {Cov} [X_{i},X_{j}], $$

- such that
$1 \leq i \leq k$ and $1 \leq j \leq k$

- The general form of its probability density function is

$$ {\displaystyle f_{\mathbf {X} }(x_{1},\ldots ,x_{k})={\frac {\exp \left(-{\frac {1}{2}}({\mathbf {x} }-{\boldsymbol {\mu }})^{\mathrm {T} }{\boldsymbol {\Sigma }}^{-1}({\mathbf {x} }-{\boldsymbol {\mu }})\right)}{\sqrt {(2\pi )^{k} \lVert {\boldsymbol {\Sigma }} \rVert }}}} $$

- (참고) Mahalanobis distance
> The Mahalanobis distance is a measure of the distance between a point P and a distribution D

$$ \sqrt{({\mathbf {x} }-{\boldsymbol {\mu }})^{\mathrm {T} }{\boldsymbol {\Sigma }}^{-1}({\mathbf {x} }-{\boldsymbol {\mu }})} $$

- 만약, $\Sigma = \sigma^2 I$ 이라면, 즉, 각 변수간 공분산이 모두 0인 경우, 마할라노비스 거리는 유클리디안 거리와 같아집니다.

## 9. Covariance의 의미

- Variance(분산) : 데이터가 펼쳐진 정도, 분산이 작으면, 데이터가 좁은영역에 모여있고, 분산이 크면 데이터가 넓은 영역에 퍼지는 형태를 보임

$$\operatorname {var} (\mathbf {X} )=\operatorname {Cov} (\mathbf {X} ,\mathbf {X} )=\operatorname {E} \left[(\mathbf {X} -\operatorname {E} [\mathbf {X} ])(\mathbf {X} -\operatorname {E} [\mathbf {X} ])^{\rm {T}}\right]$$

- Covariance(공분산) : 두 변수간 데이터가 퍼진 정도를 나타냄

$$ \operatorname {Cov} (\mathbf {X} ,\mathbf {Y} )=\operatorname {E} \left[(\mathbf {X} -\operatorname {E} [\mathbf {X} ])(\mathbf {Y} -\operatorname {E} [\mathbf {Y} ])^{\rm {T}}\right] $$

- Covariance를 의미를 살펴보면,

$$ Cov(X, Y) = \frac{(x_1 - \mu_x)(y_1 - \mu_y) + (x_2 - \mu_x)(y_2 - \mu_y) + \dots + (x_n - \mu_x)(y_n - \mu_y)}{n} $$

- 이고, $(x_i - \mu_x)(y_i - \mu_y)$ 가 양수인 경우는 각각 평균보다 크거나, 각각 평균보다 작은 경우이며, 음수인 경우 그 반대이다. 또한 평균으로부터 값이 멀어질 수록 그 값이 커지게 된다. 즉, 각 변수의 평균을 중심으로 분산의 방향을 확인할 수 있다. 데이터를 축에 plotting했을 때, Cov(X,Y)가 양수이면, 1,3분면에 분포하며, 음수이면 주로 2,4분면에 분포할 것으로 생각할 수 있다. 다만, 공분산은 각 변수의 단위에 따라 값의 크기가 달라져서 절대적인 값의 크기로 비교하는 것은 타당하지 않다.

- 그래서, 공분산을 각각의 표준편차로 나누어 그 값을 [-1, 1]로 변환하여 계산한 것을 correlation(상관계수)라고 한다.

$$ {\displaystyle \rho _{X,Y}=\operatorname {corr} (X,Y)={\operatorname {Cov} (X,Y) \over \sigma _{X}\sigma _{Y}}={\operatorname {E} [(X-\mu _{X})(Y-\mu _{Y})] \over \sigma _{X}\sigma _{Y}},\quad {\text{if}}\ \sigma _{X}\sigma _{Y}>0.} $$

## 10. Linear Regression

### 10.1 Problem definition

- Data: Measured pairs $(x,y)$, where $x \in R^{d+1}$ (input) and $y \in R$ (output)

- Goal: Find a function $f: R^{d+1} \rightarrow R$ such that $y \sim f(x;w)$ for the data pair $(x,y)$. $f(x;w)$ is the **regression function** and the vector $w$ are its paramenters.

- Definition of linear regression: A regression method is called linear if the prediction $f$ is a linear function of the unknown parameters $w$.

### 10.2 Least squares solution
- Least squares finds the w that minimizes the sum of squared errors. The least squares objective in the most basic form where $f(x;w) = x^Tw$ is

$$ L = \sum_{i=1}^n (y_i - x_i^T w)^2 = \lVert y-Xw \rVert ^2 = (y-Xw)^T(y-Xw)$$

- We defined $y = [y_1, \dots, y_n]^T \space$ and $X = [x_1, \dots, x_n]^T.

$$ w_{LS} = \underset{w}{\operatorname{argmin}} \sum_{i=1}^n (y_i - (w_0 + \sum_{j=1}^d x_{ij}w_j))^2 $$

- Taking the gradient with respect ot w and setting to zero, using vectors, this can be written:

$$ \nabla_{w} L = 0 \space \Rightarrow \space \sum_{i=1}^n \nabla_{w} (y_i^2 - 2w^T x_i y_i + w^T x_i x_i^T w) = 0 $$

- solving gives,

$$ - \sum_{i=1}^n 2 y_i x_i + (\sum_{i=1}^n 2 x_i x_i^T)w = 0 \space \Rightarrow \space w_{LS} = (\sum_{i=1}^n x_i x_i^T)^{-1}(\sum_{i=1}^n y_i x_i)$$

- solving gives as matrix version,

$$ w_{LS} = (X^T X)^{-1} X^T y $$

- In other words, $w_{LS}$ is the vector that minimizes $L$.

### 10.3 Maximum likelihood for Gaussian linear regression
- Assume a diagonal covariance matrix $\Sigma = \sigma^2I$. The density is

$$ p(y \mid \mu, \sigma^2) = \frac{1}{(2 \pi \sigma^2)^{\frac{n}{2}}} exp(-\frac{1}{2 \sigma^2}(y-\mu)^T(y-\mu)) $$

- Plug $\mu = Xw$ into the multivariate Gaussian distribution and solve for $w$ using maximum likelihood

$$ w_{ML} = \underset{w}{\operatorname{argmax}} ln \space p(y \mid \mu = Xw, \sigma^2)$$

$$ = \underset{w}{\operatorname{argmax}} -\frac{1}{2 \sigma^2} \lVert y - Xw \rVert^2 - \frac{n}{2}ln(2 \pi \sigma^2)$$

- Least squares(LS) and maximum likelihood(ML) share the same solution:

$$LS: \space \underset{w}{\operatorname{argmin}} \lVert y - Xw \rVert^2  \Leftrightarrow ML: \space \underset{w}{\operatorname{argmax}} -\frac{1}{2 \sigma^2} \lVert y - Xw \rVert ^ 2 $$

- therefore, in a sense we are making a *independent Gaussian noise assumption* about the error, $\epsilon_i = y_i - x_i^Tw$

- Other ways of saying this: \
  1)
$y_i = x_i^Tw + \epsilon_i, \space \epsilon_i \overset{i.i.d}{\sim} N(0, \sigma^2), \space for \space i = 1, \dots, n.$ \
  2)
$y_i \overset{ind}{\sim} N(x_i^Tw, \sigma^2), \space for \space i = 1, \dots, n.$ \
  3)
$y \sim N(Xw, \sigma^2I)$

## 11. Bayesian linear regression

### 11.1 Model

- Have vector $y \in R^n$ and covariates matrix $X \in R^{n \times d}$. The *i*th row of $y$ and $X$ correspond to the *i*th observation $(y_i, x_i)$

- In a Bayesian setting, we model this data as:

$$ Likelihood: \space y \sim N(Xw, \sigma^2I) $$

$$ Prior: \space w \sim N(0, \lambda^{-1}I) $$

  - Regarding prior distribution, although not covered here, see [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior).
  
- The unknow model variable is $w \in R^d$
  - The "likelihood model" says how well the observed data agrees with w.
  - The "model prior" is our prior belief (or constraints) on w.

- This is called Bayesian linear regression because we have defined a prior on the unknown parameter and will try to learn its posterior.

### 11.2 MAP(Maximum A Posteriori) solution

 - Let us assume that the prior for $w$ is Gaussian, $w \sim N(0, \lambda^{-1}I)$. Then
 
 $$ p(w) = (\frac{\lambda}{2 \pi})^{\frac{d}{2}}e^{-\frac{\lambda}{2}w^Tw}$$
 
 - We can now try to find a $w$ that satisfies both the data likelihood, and our prior conditions about $w$.
 
 - Maximum a posteriori (MAP) estimation seeks the most probable value $w$ under the posterior:
 
 $$ w_{MAP} = \underset{w}{\operatorname{argmax}} \space ln \space p(w \mid y,X) $$
 
 $$ = \underset{w}{\operatorname{argmax}} \space ln \frac{p(y \mid w,X_)p(w)}{p(y \mid X)} $$
 
 $$ = \underset{w}{\operatorname{argmax}} \space ln \space p(y \mid w,X) + ln \space p(w) - ln \space p(y \mid X) $$
 
 - The normalizing constant term $ln \space p(y \mid X)$ doesn't involve $w$. Therefore, we can maximize the first two terms alone.

- In many models we don't know $ln \space p(y \mid X)$, so this fact is useful. 

- Hence,

$$ w_{MAP} = \underset{w}{\operatorname{argmax}} \space ln \space p(y \mid w,X) + ln \space p(w) $$

$$ = \underset{w}{\operatorname{argmax}} \space - \frac{1}{2 \sigma^2}(y - Xw)^T(y-Xw) - \frac{\lambda}{2}w^Tw + const $$

- this solution for $w_{MAP}$ is the same as for ridge regression (we do not cover ridge regression(RR) here).

$$ w_{MAP} = (\lambda \sigma^2I + X^TX)^{-1}X^Ty \space \Leftrightarrow \space w_{RR}$$

### 11.3 Point estimates

- $w_{MAP}$ and $w_{ML}$ are referred to as point estimates of the model parameters.
- The find a specific value(point) of the vector $w$ that maximizes an objective function (MAP or ML)
  - ML: Only consider data model
  - MAP: Takes into account model prior
  
- Bayesian inference goes one step further by characterizing uncertainty about the values in w using Bayes rule. (Bayesian inference는 파라메터의 uncertainty(=variance)를 계산할 수 있다. parameter의 uncertainty를 구하는 것과, prediction의 uncertainty를 구하는 것에 대해 추가공부 필요!)

- In posterior calculation, we get an updated distribution on $w$ through the transition

$$ prior \rightarrow likelihood \rightarrow posterior $$

- Bayesian learning is naturally thought of a sequential process. That is, the posterior after seeing some data becomes the prior for the next data.

- Maximum likelihood는 데이터가 주어졌을 때, OLS라는 목적함수를 최소화하는 w를 찾는 것이라면, MAP는 w에 대한 prior distribution을 가정하고, 데이터가 주어졌을 때, prior distribution을 만족하는 w를 찾는 방법이다. 이 때, $p(w \mid y,X)$를 바로 구하기 어렵기 때문에, likelihood와 prior의 곱을 최소로 하는 w를 찾는다. 
 



---
toc: true
layout: post
description: 
categories: [Statistics]
comments: true
title: GP 중심으로 PRML 훑어보기(1,2,3,6장)
---

## 1.1 Example: Polynomial Curve Fitting
- 모델의 파라메터수, 데이터 수, regularization term이 모델에 미치는 영향에 대해 예를 들어 살펴봄
  - 모델의 파라메터수가 많아지면 overfitting이 될 수 있음
  - 데이터수가 많아지면 overfitting을 막을 수 있음
  - regularization term의 변수에 따라 모델이 overfitting될 수 도 있고, underfitting될 수도 있음. 즉 fitting 정도를 조절할 수 있음.  
- 이후 MLE, MAP 등에서 모양을 조금씩 바꾸며 계속 이 예제가 등장함

## 1.2 Probability Theory
- Sum rule, production rule을 예를 들어 설명함
- Probability density, Expectations 및 Covariance를 설명함
- prior, likelihood, posterior를 설명함
  - 빈도주의자는 관찰된 데이터를 가장 잘표현하는 파라메터를 찾는다면(MLE), 
  - 베이지안은 믿음(사전지식, prior)에서 데이터가 관찰될때마다 그것을 조금씩 갱신하는 방식임
  - posterior는 prior와 likelihood의 곱에 비례하며, 데이터가 갱신될 때마다, posterior는 prior에서 likelihood로 점차 변화함
  - 다만 베이지안은 계산상의 편의를 위해(prior와 likelihood를 곱해야하므로) conjugacy 관계인 분포가 많이 사용됨
- gaussian distribution을 설명함, MLE 방식으로 gaussian distribution의 variance를 구하면, $\frac{N-1}{n}$만큼 bias 됨
- 앞서 나왔던 커브피팅 문제에서 관측데이터와 예측데이터의 차이(error 또는 residual)가 gaussian distribution을 따른다고 가정하고,
  - MAP를 최소로 하는 해를 구하면, 그것은 ridge regression의 해를 구하는 형태와 동일해짐
- 베이지안 인퍼런스를 위해서는 prediction distribution을 구해야함, prediction distribution은 production rule에 의해 
  - likelihood와 posterior(갱신된 prior)의 곱으로 표현되며, posterior가 gaussian distribution인 경우, prediction distribution을 gaussian form으로 정리하면 
  - gausssian process 형태인( $N(t \mid m(x), s^2(x)$ )로 정리됨
  - 즉, bayesian infrerence를 위해 구한 prediction distribution은 gaussian process로 표현됨.

## 2.1 binary variable
- Conjugate Prior : A prior is conjugate for the likelihood function if the posterior is of the sam form as the prior.
- binomial 분포와 beta분포, bernuii 분포와 beta분포는 각각 conjugacy임

## 2.2 multinomial variable
- multinomial distribution과 Dirichlet distribution은 conjugacy임

## 2.3 Gaussian Distribution
- multivariate gaussian distribution form에서 exp의 지수파트를 mahalanobis distance라고 하며, 
  - $\Sigma$가 Identity Matrix일때, 유클리언 디스턴스가 됨
- 가우시안 분포는 중심극한정리에 쓰임(데이터 갯수가 커질수록 분포의 평균이 가우시안 분포에 가까워짐)
- spectral theorem을 보임, 이것은 이후 multivariate gaussian distribution의 mean과 variance를 증명하는데 사용됨
- 데이터의 수가 많아지면, mean으로 D개, covariance matrix의 요소로 $D(D+1)/2$ 개의 파라메터를 가짐(symmetric이므로),  
  - 행렬사이즈가 quadratically하게 증가하니까 계산량도 많아지고, 역행렬 구하는것도 어려워짐
  - 그래서 $\Sigma = diag(\sigma_i^2)$ 놓거나(probability density contour 그림을 그리면 coordinate axie방향으로만 ellipse가 그려짐, 제한된 방향성을 가진다는 의미임) 
  - 또는 더 제한해서 $\Sigma = \sigma^2I$로 할 수 있음. 이때는 모든 대각행렬이 같은 값을 가지므로 동그란 모양으로 나타나고, 이것을 isotropic covariance라고 한다. (방향성이 없고, 거리만 관계가 있음)
  - 빠른 계산과 데이터간의 관계의 특징을 잡아내는 것은 trade off 관계임
- 또한 gaussian distribution은 multimodal distribution을 표현하는데 한계가 있음. 그래서 gaussian mixture model을 이후에 배울 것임
- (8장에서 다룬다고 함, 추가로 살펴볼것!) **For instance, the Gaussian version of the Markov random field, which is widely used as a probabilistic model of images, is a Gaussian distribution over the joint space of pixel intensities but rendered tractable through the imposition of considerable structure reflecting the spatial organization of the pixels**
-  joint distribution $p(x_a, x_b)$ is Gaussian, then the conditional distribution $p(x_a|x_b)$ will again be Gaussian. 시간관계상(?) 증명은 이해는 하지 못하고 그냥 받아들임.
- marginal gaussian distribution( $p(x_a) = \int p(x_a, x_b) dx_b$ )도 gaussian distribution임
- Gaussian variable에 대한 bayes theorem과 gaussian에 대한 MLE를 유도함. 시간관계상 이해하지 않고 받아들임.
- 데이터가 특정 시간마다 업데이트 되는 경우, N번째 mean의 MLE는 N-1까지의 mean에 새로 업데이트 된 데이터와 N-1까지 mean의 차이에 대해 1/N만큼 반영한다. N이 커질수록, 차이가 동일한 경우, 업데이트 되는 값의 크기가 작아지게된다.
- Bayesian관점으로 Gaussian 분포에서 분산을 알고, 평균을 추정해야하는 경우(MAP), posterior의 평균은 prior의 평균과 likelihood의 평균사이에 존재한다. $N = 0$이면, prior의 mean이 되고, $N \rightarrow \infty$이면 maximum likelihood의 mean이 된다. 분산은 $N=0$인 경우, prior의 분산을 따르며, $N \rightarrow \infty$이면 0이 되고 posterior분포는 maximum likelihood의 평균값에서 무한대의 값을(분산 = 0) 표현하는 분포가 된다.
- 평균을 알고 분산을 모르는 경우, 평균과 분산을 모두 모르는 경우에 대해 각각 gaussian 분포에서 MAP방식으로 mean과 variance를 추정하는 법을 다룬다.

## 3.1 Linear Basis Function Model
- basis function( $\phi(x)$ )을 적용한 형태로 linear model을 정의함
$$y(x, w) = w_0 + \sum_{j=1}^{M-1}w_j\phi_j(x) $$
- basis function으로 non-linear function을 사용함으로써 비선형적 feature들을 모델이 표현할 수 있음, w에 대해 선형 모형이므로 linear 모델이라고 부름
- chapter1에서 나온 예제는 $\phi(x)_j = x^j$ 인 경우임
- 모형의 예측값과 실제값의 차이가 gaussian distribution을 따른다고 가정하고, MLE로 평균을 추정하면,
$$w_{ML} = (\Phi^T\Phi)^{-1}\Phi^Tt$$
- 이며, 이것을 normal equations for the least squares problem이라고 함. $\Phi$는 design matrix라고 함

## 3.3 Bayesian Linear Regression
- posterior distribution은 다음과 같다.
$$p(w \mid t) = N(w \mid m_N, S_N)$$
where
$$m_N = S_N(S_0^{-1}m_0 + \beta\Phi^Tt)$$
$$S_N^{-1} = S_0^-1 + \beta \Phi^T \Phi$$

- 여기서 $m_0 = 0$, $S_0 = \alpha^{-1}I$ 인 경우, 아래와 같이 m_N과 S_N을 구하게 된다.
$$m_N = \beta S_N \Phi^Tt$$
$$S_N^{-1} = \alpha I + \beta \Phi^T \Phi$$
- bayesian linear regression에서, $\Phi(x)^T S_N \Phi(x')$를 k(x, x_n)의 형태로 변경할 수 있고, 이것은 gaussian process의 형태가 된다.

## 6.1 Dual representation
- linear regression model의 솔루션을 kernel function을 중심으로 다시 표현할 수 있음
- 이때, gram matrix $K = \PhiPhi^T$ 이고, $K_{nm} = \Phi(x_n)^T \Phi(x_m) = k(x_n, x_m)$이며, k()는 kernel function임

## 6.2 Constructing Kernels
- basis function을 이용하여 직접 커널을 만들고, 그 커널이 valid한지 확인하는 방법과, kernel function의 valid한 특징을 이용하여 kernel function을 생성하는 방법이 있다.

## 6.3 Gaussian Process Regression

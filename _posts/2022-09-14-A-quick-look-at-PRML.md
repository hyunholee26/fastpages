---
toc: true
layout: post
description: 
categories: [Statistics]
comments: true
title: PRML 훑어보기
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
- 

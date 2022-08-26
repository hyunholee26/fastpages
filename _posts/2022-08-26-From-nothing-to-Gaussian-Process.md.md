---
toc: true
layout: post
description: From nothing to gaussian process
categories: [gaussian process]
title: From nothing to gaussian process (아무 지식이 없는 상태에서 Gaussian Process 까지 이해하기)
---

## 1. Maximum likelihood estimation(MLE)
> In statistics, **[maximum likelihood estimation (MLE)](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)** is a method of estimating the parameters of an assumed probability distribution, given some observed data. This is achieved by maximizing a likelihood function so that, under the assumed statistical model, the observed data is most probable. The point in the parameter space that maximizes the likelihood function is called the maximum likelihood estimate. 
> -- From wikipedia

 - MLE는 주어진 관측데이터(D)에 대해, 어떠한 분포를 가정했을 때, 관측데이터와 가장 유사한 분포를 만드는 파라메터(θ)를 추정하는 방법입니다.
 - We pick the distribution family p(·), but don’t know the parameter θ
 - MLE를 수식으로 표현하면 아래와 같습니다.
$$\hat \theta_{MLE} = \underset{\theta}{\operatorname{argmax}} p(D|\theta), where D = \{ x_i \}_{i=1}$$ 

- 그리고 다음과 같이 Maximum Likelihood가 되는 파라메터(θ)를 추정할 수 있습니다.

$$ \nabla_{\theta} p(D|\theta) $$

 - 
 - 
 - , 데이터 특징에 따라 우리가 선택할 수 있는 분포들은 binomial, poisson, gaussian 등이 있으며, 각각 해당하는 pmf 또는 pdf가 존재함. 
 

- 예를들면, 공장에서 


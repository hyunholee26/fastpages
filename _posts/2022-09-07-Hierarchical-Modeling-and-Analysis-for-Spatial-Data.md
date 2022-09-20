---
toc: true
layout: post
description: 
categories: [Spatial Statistics]
comments: true
author: Hyunho Lee
title: Hierarchical Modeling and Analysis for Spatial Data - 주요개념 (작성중)
---

## 0. 들어가며
Spatiotemporal Analysis를 수강하면서, 주요개념을 정리하기 위해 참고사이트의 내용을 다시 정리한 글입니다. 대부분의 내용은 참고한 사이트를 따르며, 일부 제가 이해한 내용을 추가적으로 작성하였습니다.
- 참고자료 : https://books.google.dm/books?id=YqpZKTp-Wh0C&printsec=frontcover#v=onepage&q&f=false

## 1. Stationarity

### 1.1 Strictly stationary(strong stationary)
If, for any given $n \le 1$, any set of $n$ sites { $s_1, \cdots s_n$ } and any $h \in R^r$ the distribution of $(Y(s_1), \cdots , Y(s_n))$ is the same as that of $(Y(s_1 + h), \cdots , Y(s_n + h))$

### 1.2 Weak stationary
If $\mu(s) \equiv \mu$ (i.e., it has a constant mean) and $Cov(Y(s), Y(s+h)) = C(h)$ for all $h \in R^r$ such that $s$ and $s+h$ both lie within D.
 - A less restrictive condition, also called second-order stationarity.

### 1.3 Intrinsic stationarity
Here, we assume $E[Y(s+h) - Y(s)] = 0$ and define

$$E[Y(s+h) - Y(s)]^2 = Var(Y(s+h) - Y(s)) = 2 \gamma (h)$$

 - The function $2 \gamma (h)$ is called the variogram, and $\gamma (h)$ is called semivariogram
 - covariance function $C(h)$ is sometimes referred to as the covariogram

### 1.4 Relation between variogram and covariance function

$$
\begin{aligned}
2 \gamma (h) &= Var(Y(s + h) - Y(s)) \\
&= Var(Y(s + h)) + Var(Y(s)) - 2 Cov(Y(s+h),Y(s)) \\
&= C(0) + C(0) - 2C(h) \\
&= 2(C(0) - C(h))
\end{aligned}$$ 

so, $\gamma(h) = C(0) - C(h)$

## 2. Isotropy
If the semivariogram function $\gamma(h)$ depends upon the separation vector only through its length $\lVert h \rVert$, then we say that the variogram is isotropic; if not, we say it is anisotropic.

 - If a process is intrinsically stationary and isotropic, it is called homogeneous
 - Isotropic variograms are popular because of their simplicity, interpretability, etc.

## 3. Moran's I and Geary's C

## 4. Simultaneous Autoregressive(SAR) and Conditional Autoregressive(CAR)
In the case of time-series data, SAR and CAR are same model. How about spatial data?

 - SAR


 - CAR

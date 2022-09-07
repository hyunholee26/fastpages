---
toc: true
layout: post
description: 
categories: [Statistics]
comments: true
title: Random Process 기초 (작성중)
---

## 0. 들어가며
Spatiotemporal Analysis를 수강하면서, Random Process를 공부하기 위해 참고사이트의 내용을 다시 정리한 글입니다. 대부분의 내용은 참고한 사이트를 따릅니다.
- 참고자료 : [https://www.probabilitycourse.com/chapter10/10_1_0_basic_concepts.php](https://www.probabilitycourse.com/chapter10/10_1_0_basic_concepts.php)

## 1. Ramdom Process
A **random process** is **a collection of random variables** usually indexed by time. 랜덤프로세스는 인덱스트된 확률변수들의 모임으로 정의되며, 각 요소는 확률변수와 동일하게 다루면 되는 것으로 이해했습니다. 보통은 시간으로 index되지만, 공간의 경우, 2차원 또는 3차원으로 확장되어 index될 수 있습니다. 랜덤프로세스는 인덱스에 따라 아래와 같이 구분됩니다.

 - **descrete-time** random process
 > A continuous-time random process is a random process {X(t),t∈J}, where J is an interval on the real line such as [−1,1], [0,∞), (−∞,∞), etc.
 
 - **continuous-time** random process
 > A discrete-time random process (or a random sequence) is a random process {X(n)=Xn,n∈J}, where J is a countable set such as N or Z.

또한, 확률변수가 가지는 값의 종류에 따라 아래와 같이 구분할 수 있습니다.
 - **descrete-valued** random process
 - **continuous-valued** random process

(연습문제) $X_n = 1000(1+R)^n, for \space n = 0,1,2, \cdots.$ 이고, $R ~ Uniform(0.04, 0.05)$인 경우, $E[X_3]$을 구하면,

 - The random variable $X_3$ is given by $X_3 = 1000(1+R)^3$.
 - If you let $Y = 1 + R$, then $Y ~ Uniform(1.04, 1.05)$, so
$$f_Y(y) = \begin{cases} 100 \: 1.04 \le y \le 1.05 \\ 0 \: otherwise \end{cases} $$

## 2. Random Processes as Random Functions:
A random process is a random function of time.
- random process는 index(여기서는 시간)에 따른 random function으로 볼 수 있습니다.

We call each of these possible functions of X(t) a sample function or sample path. It is also called a realization of X(t). 
- X(t)로서 가능한 모든 함수들을 sample function, sample path 또는 realization of X(t)라고 부릅니다. In engineering applications, random processes are often referred to as random signals. (공간통계는 주로 2차원의 공간데이터를 다루어서 index가 2차원 또는 3차원인 경우 random field라고 부릅니다.)




### 3.1

 
 




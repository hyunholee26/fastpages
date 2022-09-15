---
toc: true
layout: post
description: 
categories: [Statistics]
comments: true
author: Hyunho Lee
title: Gaussian Process and Gaussian Process Regression
---

## 0. 들어가며
Spatiotemporal Analysis를 수강하면서, Gaussian Process를 공부하기 위해 참고사이트의 내용을 다시 정리한 글입니다. 대부분의 내용은 참고한 사이트를 따르며, 일부 제가 이해한 내용을 추가적으로 작성하였습니다.
- 참고자료
  - [https://pasus.tistory.com/209?category=1287736](https://pasus.tistory.com/209?category=1287736) 
  - [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)

## 1. Gaussian Process

가우시안 프로세스(GP, Gaussian process)는 프로세스 집합 내에 있는 랜덤변수들의 임의의 조합이 모두 결합(joint) 가우시안 분포를 갖는 랜덤 프로세스로 정의된다.

 - 예를 들어서 인덱스 $x_1, x_2, \cdots, x_n$에 해당하는 랜덤변수가 $f_i = f(x_i)$ 일 때, 이로부터 가능한 모든 부분 집합 ${f_1}, {f_2, f_3}, \cdots, {f_1, f_2, \cdots f_m}$ 이 모두 결합 가우시안 분포(joint gaussian distribution)를 갖는 프로세스이다. 
 - 달리 설명하자면, $f_1$ 을 성분으로 하는 벡터 $f_{1:m} = [f_1, f_2, \cdots, f_m]^T$가 가우시안 랜덤벡터인 프로세스이다. 여기서 $m$ 은 프로세스에서 임의로 선정한 인덱스의 갯수이므로, 가우시안 랜덤벡터는 무한 차원을 가질 수 있다. 즉 가우시안 프로세스는 가우시안 랜덤벡터를 무한 차원으로 확장한 것으로 설명할 수도 있겠다.

가우시안 분포의 특성을 평균과 공분산으로 표현하듯이 가우시안 프로세스도 평균함수 $\mu(x)$ 와 공분산 $k(x,x')$ 로 특징지울 수 있다.

$$f(x) \sim GP(\mu(x), k(x, x'))$$

여기서 공분산 $k(x,x')$ 는 다음과 같다.

$$k(x,x') = E[(f(x) - \mu(x))(f(x') - \mu(x'))]$$ 


---
toc: true
layout: post
description: 
categories: [Statistics]
comments: true
title: Random Process
---

## 0. 들어가며
Spatiotemporal Analysis를 수강하면서, Random Process를 공부하기 위해 참고사이트의 내용을 다시 정리한 글입니다. 대부분의 내용은 참고한 사이트를 따르며, 일부 제가 이해한 내용을 추가적으로 작성하였습니다.
- 참고자료
  - [https://pasus.tistory.com/209?category=1287736](https://pasus.tistory.com/209?category=1287736) 
  - [https://www.probabilitycourse.com/chapter10/10_1_0_basic_concepts.php](https://www.probabilitycourse.com/chapter10/10_1_0_basic_concepts.php)

## 1. Ramdom Process
A **random process** is **a collection of random variables** usually indexed by time. 
 - 랜덤변수(random variable)는 확률 실험의 결과에 실숫값을 대응시키는 함수로 정의된다. 또한 랜덤 프로세스(random process)는 어떤 파라미터로 인덱스(index)된 무한개의 랜덤변수의 집합으로 정의된다.
 - 인덱스 파라미터를 고정시킨다면 랜덤 프로세스는 랜덤변수가 된다.

(연습문제1) $X_n = 1000(1+R)^n, for \space n = 0,1,2, \cdots.$ 이고, $R \sim Uniform(0.04, 0.05)$인 경우, $E[X_3]$ 는?

 - The random variable $X_3$ is given by $X_3 = 1000(1+R)^3$.
 - If you let $Y = 1 + R$, then $Y \sim Uniform(1.04, 1.05)$, so
$$f_Y(y) = \begin{cases} 100 \quad 1.04 \le y \le 1.05 \qquad \because 1/(1.05 - 1.04) = 100 \newline \newline 0 \qquad otherwise \end{cases} $$

 - To obtain $E[X_3]$, we can write
 
$$ 
\begin{aligned} 
E[X_3] &= 1000E[Y^3] \\
&= 1000\int_{1.04}^{1.05} 100y^3 dy \qquad \because E(x) = \int xp(x) dx, \text{in this case} \space x \space is \space y^3 \space and \space p(x) = 100  \\
&= \frac{10^5}{4} [ y^4 ]_{1.04}^{1.05} \\
&= \frac{10^5}{4} [(1.05)^4 - (1.04)^4] \\
&\approx 1141.2
\end{aligned}$$

(연습문제2) Let ${X(t), t \in [0, \infty )}$ be defined as

$$X(t) = A + Bt, \text{for all} \space t \in [0,\infty)$$

where A and B are independent normal $N(1,1)$ random variables.

2-1. Define the random variable $Y=X(1)$. Find the PDF of $Y$
  - We have

$$ Y = X(1) = A + B$$

  - Since A and B are independent $N(1,1) random variable, Y = A + B is also normal with

$$
\begin{aligned} 
E[Y] &= E[A + B] \\
&= E[A] + E[B] \\
&= 1 + 1 \\
&= 2,
\\
Var(Y) &= Var(A + B) \\
&= Var(A) + Var(B) \qquad \text{(since A and B are independent)} \\
&= 1 + 1 \\
&= 2
\end{aligned}$$

- Thus, we conclude that $Y∼N(2,2)$

2-2. Let also $Z=X(2)$. Find $E[YZ]$
 - Since Y = A + B and Z = A + 2B, so

$$
\begin{aligned} 
E[YZ] &= E[(A + B)(A + 2B)] \\
&= E[A^2 + 3AB + 2B^2] \\
&= E[A^2] + 3E[AB] + 2[B^2] \\
&= E[A^2] + 3E[A]E[B] + 2[B^2] \qquad \text{(since A and B are independent)} \\
&= Var(A) + E[A]^2 + 3E[A]E[B] + 2(Var(B) + E[B]^2) \\
&= 1 + 1 + 3 + 2(1 + 1) \\
&= 9
\end{aligned}$$ 

(참고1) Expectation value는 아래의 성질을 만족함

$$\begin{aligned}
E[X+Y] &= E[X] + E[Y] \\
E[aX] &= aE[X] \\
\end{aligned}$$

(참고2) Variance는 아래의 성질을 만족함

$$\begin{aligned}
Var(X) &= E[(X - E[X])^2] \\
&= \sum_x (x - E[X])^2 p(x) \\
&= \sum_x (x^2 - 2 E[X] x + E[X]^2)p(x) \\
&= \sum_x x^2 p(x) -2 E[X] \sum_x xp(x) + E[X]^2 \sum_x p(x) \\
&= E[X^2] - 2E[X]^2 + E[X]^2 \\
&= E[X^2] - E[X]^2
\\
\\
Var(aX + b) &= E[(aX + b - aE[X] - b)^2] \\
&= E[a^2(X - E[X])^2] \\
&= a^2E[(X - E[X])^2] \\
&= a^2Var(x)
\end{aligned}$$

(참고3) 두 random variable X, Y가 독립인 경우, 두 random variable은 uncorrelated이며, 아래의 등식이 성립합니다.
$$E[XY] = E[X]E[Y] $$

또한 이 성질을 이용하면 아래의 수식이 성립합니다.

$$\begin{aligned}
Var(X+Y) &= E[(X+Y)^2] - (E[X+Y])^2 \\
&= E[X^2 + 2XY + Y^2] - (E[X] + E[Y])^2 \\
&= E[X^2] + 2E[XY] + E[Y^2] - (E[X]^2 +2E[X]E[Y] + E[Y]^2) \\
&= E[X^2] - E[X]^2 + E[Y^2] - E[Y]^2 \\
&= Var(X) + Var(Y)
\end{aligned}$$


## 2. Random Processes as Random Functions:
A random process is a random function of time.
- random process는 index에 따른 random function으로 볼 수 있다.

We call each of these possible functions of X(t) a sample function or sample path. It is also called a realization of X(t). 
- X(t)로서 가능한 모든 함수들을 sample function, sample path 또는 realization of X(t)라고 부릅니다. 


## 3. Mean Function of a Random Process
For a random process ${X(t), t \in J}$, the mean function $\mu_X(t) : J \leftarrow R$, is defined as

$$\mu_X = E[X(t)]$$


## 4. Autocorrelation and Autocovariance
For a random process ${X(t), t \in J}$, the autocorrelation function or, simply, the correlation function, $R_X(t_1, t_2)$ is defined by

$$R_X(t1, t2) = E[X(t_1)X(t_2)], \space for \space t1, t2 \in J$$

For a random process ${X(t), t \in J}$, the autocovariance function or, simply, the covariance function, $C_X(t_1, t_2)$ is defined by

$$\begin{aligned}
C_X(t1, t2) &= Cov(X(t_1), X(t_2)) \\
&= R_X(t_1, t_2) - \mu_X(t_1)\mu_X(t_2), \space for \space t_1, t_2 \in J
\end{aligned}$$

 - Intuitively, $C_X(t1, t2)$ shows how $X(t_1)$ and $X(t_2)$ move relative to each other. If large values of $X(t_1)$ tend to imply large values of $X(t_2)$, then $(X(t_1) - E[X(t_1)])(X(t_2) - E[X(t_2])$ is positive on average. In this case, $C_X(t_1, t_2)$ is positive, and we say $X(t_1)$ and $X(t_2)$ are positively correlated. 
 - On the other hand, if large values of $X(t_1)$ imply small values of $X(t_2)$, then $(X(t_1) - E[X(t_1)])(X(t_2) - E[X(t_2)])$ is negative on average, and we say $X(t_1)$ and $X(t_2)$ are negatively correlated. 
 - If $C_X(t_1, t_2) = 0$ then $X(t_1)$ and $X(t_2)$ are uncorrelated.
 - 

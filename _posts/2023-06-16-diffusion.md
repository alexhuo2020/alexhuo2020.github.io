# Diffusion model
### see the theoretical introduction in https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

<!-- The diffusion model is based on the Langevin dynamics
$$x_t = x_{t-1} + \frac{\delta}{2} \nabla_x \log p(x_{t-1}) + \sqrt{\delta} \epsilon_t, \quad \epsilon_t \sim N(0,1)$$
So that the equilibrium distribution of $x_t$ as $t\to\infty$ is
$$\log p  = C$$ -->

### A stochastic process
A stochastic process given by
$$d X_t = \mu(X_t,t) dt + \sigma(X_t,t) dW_t$$
then its probability density $p(x,t)$ satisfies the Fokker-Planck equation
$$\partial_t p (x,t) = - \partial_x (\mu(x,t)p(x,t)) + \partial_x^2 (D(x,t)p(x,t)),\quad D(x,t)=\frac12 \sigma^2(x,t) $$
(see wiki:https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck_equation)

Hence if we take $\sigma(X_t,t) = \delta$ a constant and $\mu(X_t,t) = \nabla \log q(X_t)$, then the Fokker-Planck equation becomes
$$\partial_t p = -\partial_x (p \nabla \log q) + \partial_x^2 (\frac12p \delta^2 )$$
When $t\to\infty$,
$$p_\infty\nabla \log q = \frac12 \delta^2 \partial_x p_\infty$$
which is
$$\nabla \log q = \frac12 \delta^2 \nabla \log p_\infty$$
i.e.
$$\frac12 \delta^2 \log p_\infty =\log  q$$

Taking $\delta=1$,
we conclude:

1. For SDE $dX_t = \nabla_x \log q(X_t) dt +  dW_t$, $\log p_\infty = 2 \log q$.
2. if we take $\log q(x)=-\frac14 x^2$, then the equilibrium distribution is $p_\infty \propto e^{-\frac12 x^2}$ and is standard Gaussian.
the SDE becomes
$$dX_t = -\frac12 X_t dt + dW_t$$

### Euler-Maruyama Method
To numerically solve the SDE, we need a method. The Euler-Maruyama method is
1. partition $[0,T]$ into $0=\tau_0 <\tau_1 < \cdots  < \tau_N = T$;
2. set $Y_0 = X_0$;
3. $Y_{n+1} = Y_n + \nabla_x \log q(Y_n) \Delta t_n + \Delta W_n$, $\Delta W_n = W_{\tau_{n+1}}-W_{\tau_n}$;
By the theorem of Brownian motion, $\Delta W_n \sim N(0,\Delta t_n)$.

$x_n - x_{n-1} = -\frac12 \Delta t_n x_{t-1} + \sqrt{\Delta t_n} \epsilon_{n-1}$
Rewriting using $n\to t, \Delta t_n\to \beta_t$,
$$x_{t}  - x_{t-1} = - \frac12 \beta_t x_{t-1} + \sqrt{\beta_t} \epsilon_{t-1}, \epsilon_{t-1} \sim N(0,1)$$
We can replace $1-\frac12\beta_t$ by $\sqrt{1-\beta_t}$ since when $\beta_t$ is small, $\sqrt{1-\beta_t} \sim 1-\frac12\beta_t$ (Taylor Series).

We arrive at the following formula

$$x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_{t-1}$$

Now, let's implement the Euler-Maruyama Method.

```
import numpy as np
import torch
x_0  = torch.rand(1000)
betas = torch.linspace(0.0001,0.02,1000)
x = x_0
for i in range(len(betas)):
    eps = torch.randn_like(x)
    x = x - 0.5*betas[i]*x + np.sqrt(betas[i])*eps
import seaborn as sns
sns.distplot(x)
sns.distplot(eps)
```
![image](https://github.com/alexhuo2020/alexhuo2020.github.io/assets/136142213/f63e50cf-7f50-48b4-93cc-68fd9372bc52 = 25x)

Use the second approach:
```
import numpy as np
import torch
import torch.nn as nn
x_0  = torch.rand(1000)
betas = torch.linspace(0.0001,0.02,1000)
x = x_0
for i in range(len(betas)):
    eps = torch.randn_like(x)
    x = np.sqrt(1-betas[i]) * x + np.sqrt(betas[i])*eps
sns.distplot(x)
sns.distplot(eps)
```
![image](https://github.com/alexhuo2020/alexhuo2020.github.io/assets/136142213/689f390a-b77a-42f4-ac00-0ead73ca3976)




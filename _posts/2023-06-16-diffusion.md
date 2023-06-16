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
![image](https://github.com/alexhuo2020/alexhuo2020.github.io/assets/136142213/f63e50cf-7f50-48b4-93cc-68fd9372bc52)

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

### Distance between distributions, KL divergence
If we want to compare the generated data with real data, one can use the mean square error (MSE) if we want to fit a determinstic function.

However if we want to compare two distributions (MSE) will not work.

A tool we can use if the KL divergence, defined to be
$$KL(p||q) = \mathbb{E}_{p}[\log \frac{p}{q}]$$
For two Normal distributions $p = N(\mu_1,\Sigma_1)$, $q = N(\mu_2,\Sigma_2)$, the value is ($d$ is the dimension)
$$KL(p||q) = \frac12 tr (\Sigma_2^{-1}\Sigma_1)- d + (\mu_2 - \mu_1)^T \Sigma_2^{-1} (\mu_2-\mu_1) + \log \frac{\det \Sigma_2}{\det \Sigma_2}$$
If $\Sigma_1 = \Sigma_2 = I$, then
$$KL(p||q) = (\mu_2 - \mu_1)^T (\mu_2-\mu_1)$$
is just the $L^2$ loss.

now let's demonstrate this with an example.

We train a neural network to map from 0 to the 2 by making it Gaussian.

```
x = torch.randn((2000,1)) + 2.

class MM(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1,20)
        self.l2 = nn.Linear(20,1)
    def forward(self,x):
        return self.l2(torch.relu(self.l1(x)))
model = MM()
optim = torch.optim.Adam(model.parameters())
for epoch in range(1000):
    eps = torch.zeros((2000,1))
    loss = torch.sum((model(eps) + torch.randn((2000,1))  -x)**2)
    optim.zero_grad()
    loss.backward()
    optim.step()
eps = torch.zeros((1000,1))
y = model(eps)
sns.distplot(y.detach().numpy())
```
![image](https://github.com/alexhuo2020/alexhuo2020.github.io/assets/136142213/93ee998a-e44b-4dad-bd3b-3d88f2090d0e)






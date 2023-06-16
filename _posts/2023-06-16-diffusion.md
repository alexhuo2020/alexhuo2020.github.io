# Diffusion model 1: Some theoretical background
### see the theoretical introduction in https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

<!-- The diffusion model is based on the Langevin dynamics
$$`x_t = x_{t-1} + \frac{\delta}{2} \nabla_x \log p(x_{t-1}) + \sqrt{\delta} \epsilon_t, \quad \epsilon_t \sim N(0,1)`$$
So that the equilibrium distribution of $x_t$ as $t\to\infty$ is
$$`\log p  = C`$$ -->

### A stochastic process
A stochastic process given by
$$`d X_t = \mu(X_t,t) dt + \sigma(X_t,t) dW_t`$$
then its probability density $`p(x,t)`$ satisfies the Fokker-Planck equation
$$`\partial_t p (x,t) = - \partial_x (\mu(x,t)p(x,t)) + \partial_x^2 (D(x,t)p(x,t)),\quad D(x,t)=\frac12 \sigma^2(x,t) `$$
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

Thus we can add noise to make predictions.

### Brief introduction to variational inference
Suppose the distribution of $x_0$ depends on some hidden parameter $x_1$.
We want to infer the distribution of $x_0$ given its observations.

Beyes rule
$$p(x_1|x_0) = \frac{p(x_0|x_1)p(x_1)}{p(x_0)}$$

#### MLE (Maximum Likelihood Estimation):
Assume $x_1$ is a constant parameter.
Assume we have a model $x_0|x_1 \sim P_{x_1}$
$$\max_{x_1} \log \Pi_{i=1}^n p(x_0^i| x_1) $$
The log-likelihood
$$ \max_{x_1} \sum_{i=1}^n \log p(x_0^i|x_1) = \max_{x_1} \sum_{i=1}^n \log p(x_0^i|x_1) - \log p(x_0^i|x_1^*)  $$
since the second term is a constant, where $x_1^*$ is the ideal value. The above formula equals
$$\frac{1}{n}\sum_{i=1}^n \log \frac{p(x_0^i|x_1)}{p(x_0^i|x^*)} \to \mathbb{E}_{x_0|x_1^*} [\log \frac{p(x_0|x_1)}{p(x_0|x_1^*)}] = - KL(p(x_0|x_1)||p(x_0|x_1^*))$$
That is, maximizing the likelihood is equal to minimizing the KL divergence between the posterior distributions.


#### Bayesian posterior sampling
Instead get a point estimate of $x_1$, one can assign a prior distribution $p(x_1)$ and use the Bayesian formula to get the posterior distribution $p(x_1|x_0)$. Then one can do the prediction using
$$p(x_p|x_0) = \int p(x_p|x_1) p(x_1|x_0) dx_1$$
However, the posterior sampling is not easy since $p(x_0)$ is unknown.


#### Variational Bayes
We can approximate the posterior distribution by using a distribution $q_\phi(x_1|x_0)$. In order to get a nice approximation, we want the distance between this approximated posterior distribution is close to the real one.

$$\min_{\phi} KL(q_\phi(x_1|x_0)||p(x_1|x_0))$$

$$\begin{aligned}- \int_{z}{q_\phi(\cdot|x)} \log \frac{q_\phi(z|x_0)}{p(z|x_0)} dz
 &= \int_z{q_\phi(\cdot|x_0)} \log \frac{p(z|x_0)}{q_\phi(z|x)} dz = \int_{q_\phi(\cdot|x)} \log  \frac{p(z,x_0)}{q_\phi(z|x)p(x_0)} dz\\&=  \int_{q_\phi(\cdot|x_0)} \log  \frac{p(z,x_0)}{q_\phi(z|x)} dz - \int_{q_\phi(\cdot|x_0)} \log  {p(x_0)}dz=\int_{q_\phi(\cdot|x_0)} \log  \frac{p(z,x_0)}{q_\phi(z|x)} dz - \log p(x_0)\end{aligned}$$
The first term has a name ELBO (Evidence Lower BOund), $\mathcal L$.
<!-- Use the Jensen's inequality, the first term is bounded by
$$\int_z {q_\phi(\cdot|x_0)} \log  \frac{p_\theta(z|x_0)p_\theta(x_0)}{q_\phi(z|x_0)} dz \le \int_z {q_\phi(\cdot|x_0)} \log  \frac{p_\
(z|x_0)}{q_\phi(z|x_0)} dz + \log \int_z p_\theta(z,x_0) dx = -KL(q_\phi(x_1|x_0)||p_\theta(x_1|x_0)+ \log p_\theta(x_0) $$ -->

Example:
$x_1 \sim N(0,1)$, $x_0\sim N(x_1,1)$, then the posterior distribution 
$p(x_1|x_0) \propto p(x_0|x_1) p(x_1) \propto e^{-\frac12 (x_0-x_1)^2} e^{-\frac12 x_1^2}\propto e^{-(x_1-\frac12 x_0)^2}$, hence $x_1|x_0 \sim N(\frac12 x_0|\frac{1}{\sqrt{2}})$

Here $p(x_0,x_1) = p(x_0|x_1)p(x_1) \propto e^{-\frac12 (x_0-x_1)^2} e^{-\frac12 x_1^2} $ is a joint distribution.

```
z = torch.randn((1000,1))
x0 = torch.randn(1) + z**2
def normalpdf(x,mu):
  return 1/np.sqrt(2*np.pi)*torch.exp(-0.5*(x-mu)**2)
model = MM()
optim = torch.optim.Adam(model.parameters())
for epoch in range(1000):
    z = torch.randn(1) + model(x0)
    loss = -torch.mean(torch.log(normalpdf(z,0)*normalpdf(x0,z)/(normalpdf(z,model(x0)))))
    optim.zero_grad()
    loss.backward()
    optim.step()
import matplotlib.pyplot as plt
plt.plot(x0,model(x0).detach().cpu())
```
![image](https://github.com/alexhuo2020/alexhuo2020.github.io/assets/136142213/cb3350f7-d152-4f81-8559-c1c4d5a642a7)

after we get the posterior distribution $q_\phi(\cdot|x)$, we can use it to generate the distribution of $x_0$

```
z = torch.randn(1) + model(x0)
x0_pred = torch.randn(1) + z
sns.distplot(x0_pred.detach().numpy())
sns.distplot(x0)
```
![image](https://github.com/alexhuo2020/alexhuo2020.github.io/assets/136142213/cf950642-5000-4fda-933e-5e283c3fbb51)

![image](https://github.com/alexhuo2020/alexhuo2020.github.io/assets/136142213/dcf833ed-c743-4b9e-9a5b-a79f879a75dc)

here we have assumed that we have the model $x_0|x_1 \sim N(x_0|x_1)$. In general (as we assume the distribution $p$ is exact. In general, we need to infer the model with parameters $p_\theta$. The way to do is similar, but we need to replace $p$ by $p_\theta$ in the above formula.

Example:
$x_1 \sim N(0,1)$, $x_0 \sim N(x_1^2,1)$, let's infer the $x_1^2$ function.
```
x1 = torch.randn((1000,1))
x0 = torch.randn((1000,1)) + x1**2
lognormpdf = lambda x,mu: -0.5*(x-mu)**2
model = MM()
model_theta = MM() # used for the theta 
optim = torch.optim.Adam(model.parameters())
optim_theta = torch.optim.Adam(model_theta.parameters())
for epoch in range(5000):
    z = torch.randn(1) + model(x0)
    loss = -torch.mean(lognormpdf(z,0)+lognormpdf(x0,model_theta(z)) - lognormpdf(z,model(x0)))
    # loss=- torch.mean(torch.log(normalpdf(z,0)*normalpdf(x0,model_theta(z))/(normalpdf(z,model(x0))))) # this cause error sometimes
    optim.zero_grad()
    optim_theta.zero_grad()
    loss.backward()
    optim.step()
    optim_theta.step()
z = torch.randn((1000,1)) #+ model(x0)
x0_pred = torch.randn((1000,1)) + model_theta(z)
sns.distplot(x0_pred.detach().cpu())
sns.distplot(x0)
```
![image](https://github.com/alexhuo2020/alexhuo2020.github.io/assets/136142213/14f995ed-1780-4b8a-a6de-937d0ba42963)
Note here to sample prediction, we use the model $x_1\sim N(0,1),$ $x_0 \sim N(f_\theta(x_1),1)$

remark: one can also use the obtained posterior distribution $q_\phi(x_1|x_0)$ to make predictions using Bayesian as 
$$p(x_0^{pred}|x_0) = \int_z p_\theta(x_0^{pred}|z) q_\phi(z|x_0) dz$$
For example, one may use the pymc package to do this.




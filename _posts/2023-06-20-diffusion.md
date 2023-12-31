# Diffusion model 2: simple illustration
### Diffusion model

We have seen that to make predictions of an unkown distribution, we can use a neural network $f_\theta$ to construct the model and then use the variational inference to infer the parameter. One can combine this idea with the stochastic process to get the diffusion model.

#### The model:

$$x_0\sim p(x_0), x_1\mid x_0 \sim N(\sqrt{1-\beta_1}x_0,\beta_1 I),\ldots, x_t\mid x_{t-1} \sim N(\sqrt{1-\beta_t}x_{t-1}|\beta_t I)$$

#### Reparametrization trick:
introduce $\alpha_t = 1-\beta_t$, $\bar\alpha_t =\Pi_{i=1}^t \alpha_t$,

$$x_t \sim N(\sqrt{\bar\alpha_t} x_0\mid {(1-\bar\alpha_t)} I)$$

#### Posterior distribution:

$$x_{t-1}\mid x_t,x_0 \sim N(\tilde\mu_t(x_t,x_0),\tilde\beta_t)$$

with $\tilde\beta_t = \beta_t \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}$, $\tilde\mu_t(x_t,x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} {x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} {x}_0$
Or introducting $\epsilon_t = (x_t - \sqrt{\bar\alpha_t} x_0)/\sqrt{1-\bar\alpha_t}$,

$$\tilde \mu_t(\epsilon_t,x_0) =  {\frac{1}{\sqrt{\alpha_t}} ( {x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} {\epsilon}_t )}$$


#### The model $f_\theta$:

$$x_T \sim N(0,1), x_{T-1}\mid x_{T},x_0 \sim N(\mu_\theta(x_T,x_0),\sigma_{\theta}(x_T,x_0)),\ldots, x_{t-1}\mid x_t \sim N(\mu_\theta(x_t,x_0),\sigma_{\theta}(x_t,x_0)),\ldots, x_0\mid x_1 \sim N(\mu_\theta(x_1,x_0),\sigma_{\theta}(x_1,x_0))$$

Now instead of one $KL$ between the true posterior distribution, we need more. We take $q=p$ to be the real posterior.

$$\min KL(q(x_{1:T}\mid x_0)||p_\theta(x_{1:T}\mid x_0))$$

is equivalent to maximize the ELBO

$$\mathcal{L} = \mathbb{E}_{q(\cdot \mid x_0)}\left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]$$



after some algebra (see Weng's blog)

$$\mathcal{L} = L_0 + L_1 + \ldots + L_T$$

where

$$L_T = KL(q(x_T|x_0)|| p_\theta(x_T,x_0),~L_t = KL(q(x_{t}|x_{t+1},x_0)||p_\theta(x_{t}|x_{t+1},x_0)), 1\le t\le T-1, ~ L_0=-\log p_\theta(x_0|x_1,x_0).$$


The KL divergence between these two Gaussians,

$$L_t = C(t) \|\mu_t - \mu_{\theta,t}\|_{L^2}$$

if we take $\sigma_\theta$ to be the same with $\tilde\beta_t$, and introducing $\epsilon_{\theta,t} = (x_t - \sqrt{\alpha_t}\mu_\theta)\frac{\sqrt{1-\bar\alpha_t}}{1-\alpha_t}$, it becomes

$$L_t = C(t) \|\epsilon_t - \epsilon_{\theta,t}\|_{L^2}$$

Adding them together gives the loss function.

Note here $q_\phi$ is taken to be the real posterior distribution.

### The forward process 

First generate $\beta_1 < \beta_2 \ldots \beta_T$ and compute $\alpha_t = 1-\beta_t$, $\bar\alpha_t = \Pi_{i=1}^t \alpha_t$:

```
betas = torch.linspace(0.001,0.2,10)
alphas = 1 - betas
alphabars = torch.cumprod(alphas,dim=0)
```
Then the forward process satisfies

$$x_t\mid x_0 \sim N(\sqrt{\bar\alpha_t},\sqrt{1-\bar\alpha_t} I)$$

We consider $x_0\sim U[0,1]$.

```
def forward_sample(x0, t):
  return torch.randn_like(x0)*torch.sqrt(1 - alphabars[t]) + torch.sqrt(alphabars[t])*x0
x0 = torch.rand((1000,1))
from time import sleep
xt = [x0.numpy()]
fig, axes = plt.subplots(2,5, sharex=True, sharey=True)

axes = axes.flatten()
for t in range(10):
  xt = forward_sample(x0,t)
  sns.kdeplot(xt, ax=axes[t])
  axes[t].set(xlabel=t+1)
  axes[t].get_legend().remove()
```

![image](https://github.com/alexhuo2020/alexhuo2020.github.io/assets/136142213/9d58a8b6-d6d1-4bea-9720-b03b28db7067)


### Posterior distribution

The posterior distribution 

$$x_{t-1} \mid x_t, x_0 = N(\mu_t,\tilde \beta_t),~\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$$

and 

$$\mu_t=\frac{1}{\alpha_t} (x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_t), ~~ \epsilon_t = (x_t - \sqrt{\bar\alpha_t}x_0)/\sqrt{1-\bar\alpha_t}$$

However, the above formula holds for $t>1$. For $t=1$, we need $q(x_0\mid x_1)$. Since we have assumed $x_0\sim U[0,1]$,

However the last step $x_0\mid x_1$ is unkown. But for the case $x_0 \sim U[0,1]$, $q(x_0) = 1$ and 

$$q(x_0\mid x_1) \propto q(x_1\mid x_0) q(x_0) \propto e^{-\frac{(x_1-\sqrt{\alpha_1}x_0)^2}{2(1-\alpha_1)}}\cdot 1 \propto e^{-\frac{(x_0-x_1/\sqrt{\alpha_1})^2}{2(1-\alpha_1)/\alpha_1}}$$

hence 

$$x_0\mid x_1 \sim N(\frac{1}{\sqrt\alpha_1} x_1,\frac{1-\alpha_1}{\alpha_1}I).$$

We can do the reverse sampling using this.

```
def posterior_param(xt, epst, t, alphas, alphasbar):
  if t<1:
    mu = 1/torch.sqrt(alphabars[0]) * xt
    Sigma = (1-alphas[0])/alphas[0]**2
  else:
    mu = 1/torch.sqrt(alphas[t]) * (xt - (1-alphas[t])/torch.sqrt(1-alphabars[t]) * epst)
    Sigma = (1-alphas[t]) * (1-alphasbar[t-1]) / ( 1-alphasbar[t])
  return mu, Sigma
xt = forward_sample(x0, 9)
xtt = []
for t in range(10)[::-1]:
  epst = (xt-alphabars[t]*x0)/torch.sqrt(1-alphabars[t])
  mu, beta = posterior_param(xt, epst, t, alphas, alphabars)
  xt = mu +  torch.randn_like(x0) * torch.sqrt(beta)
  xtt.append(xt)
fig, axes = plt.subplots(2,5, sharex=True, sharey=True)
axes = axes.flatten()
for t in range(10):
  sns.kdeplot(xtt[t], ax=axes[t])
  axes[t].get_legend().remove()
```

![image](https://github.com/alexhuo2020/alexhuo2020.github.io/assets/136142213/69b77739-5270-47dd-96e1-2f98b14549a0)


### Learn the diffusion model

The above reverse sampling process needs the knowledge of distributions of $x_0$. If we only have finite observations of $x_0$,
the reverse sampling only gives a posterior distribution rather than recover the distribution of $x_0$. 
We need to learn the model $f_\theta$.

The model:

$X_T \sim N(0,1), ~ x_{t-1}\mid x_t \sim N(\mu_{\theta,t}, \tilde\beta_t)$

We can utilize the similar relation 

$$\mu_{\theta,t} = \frac{1}{\alpha_t} (x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_{\theta,t})$$

Hence to learn the paramter $\theta$, we need to minimize the ELBO. Note the KL between two Gaussians is the MSE.

$$Loss = \sum_{t=1}^T \|\epsilon_t - \epsilon_{\theta,t}\|_{2}^2$$

Build the network
```
class Ftheta(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2,200)
        self.l2 = nn.Linear(200,400)
        self.l3 = nn.Linear(400,200)
        self.l4  = nn.Linear(200,1)
    def forward(self,x,t):
        xt = torch.concat([x,t],dim=-1)
        xt = torch.relu(self.l1(xt))
        xt = torch.relu(self.l2(xt))
        xt = torch.relu(self.l3(xt))
        return self.l4(xt)
model = Ftheta()
optim = torch.optim.Adam(model.parameters())
```
Train the network:
```
def Xt(eps, x0, t):
  return torch.sqrt(alphabars[t])*x0 + torch.sqrt(1-alphabars[t])*eps
optim = torch.optim.AdamW(model.parameters())
losses = []
for epoch in range(1000):
  x0 = torch.rand((1000,1))
  eps = torch.randn((1000,1))
  t = torch.randint(0,10,(1000,1))
  xt = Xt(eps,x0,t)
  loss = torch.mean((model(xt,t) - eps)**2*(1-alphas[t])**2/2/alphas[t]/(1-alphabars[t]))
  optim.zero_grad()
  loss.backward()
  losses.append(loss.item())
  optim.step()
```

Sampling process
```
xt = torch.randn((10000,1))
xtt = []
for t in range(10)[::-1]:
  z = torch.randn((10000,1))
  tt = torch.ones_like(z)*t
  mu = 1/torch.sqrt(alphas[t])*(xt - (1-alphas[t])/torch.sqrt(1-alphabars[t])*model(xt,tt))
  if t>0:
    xt = mu +  torch.randn_like(xt)*torch.sqrt((1-alphas[t])*(1-alphabars[t-1])/(1-alphabars[t]))
  else:
    xt = mu
  xtt.append(xt)
```

The learned distribution:
![image](https://github.com/alexhuo2020/alexhuo2020.github.io/assets/136142213/f7baee9d-0809-457e-a4fd-8ed051e89160)

The loss:
![image](https://github.com/alexhuo2020/alexhuo2020.github.io/assets/136142213/0ae135ea-4b8d-4ba3-ace3-f5042d331d11)

The sampling 

```
fig, axes = plt.subplots(2,5, sharex=True, sharey=True)
axes = axes.flatten()
for t in range(10):
  sns.kdeplot(xtt[t].detach(), ax=axes[t])
  axes[t].get_legend().remove()
```
![image](https://github.com/alexhuo2020/alexhuo2020.github.io/assets/136142213/ac22fa04-6c0d-4a96-9ec1-b885874ec96f)





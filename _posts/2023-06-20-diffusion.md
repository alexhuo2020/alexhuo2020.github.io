# Diffusion model II: simple illustration
### Diffusion model

We have seen that to make predictions of an unkown distribution, we can use a neural network $f_\theta$ to construct the model and then use the variational inference to infer the parameter. One can combine this idea with the stochastic process to get the diffusion model.

#### The model:

$$x_0\sim p(x_0), x_1\mid x_0 \sim N(\sqrt{1-\beta_1}x_0,\beta_1 I),\ldots, x_t\mid x_{t-1} \sim N(\sqrt{1-\beta_t}x_{t-1}|\beta_t I)$$

#### Reparametrization trick:
introduce $\alpha_t = 1-\beta_t$, $\bar\alpha_t =\Pi_{i=1}^t \alpha_t$,
$$x_t \sim N(\sqrt{\bar\alpha_t} x_0|{(1-\bar\alpha_t)} I)$$

#### Posterior distribution:
$$x_{t-1}|x_t,x_0 \sim N(\tilde\mu_t(x_t,x_0),\tilde\beta_t)$$
with $\tilde\beta_t = \beta_t \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}$, $\tilde\mu_t(x_t,x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} {x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} {x}_0$
Or introducting $\epsilon_t = (x_t - \sqrt{\bar\alpha_t} x_0)/\sqrt{1-\bar\alpha_t}$,
$$\tilde \mu_t(\epsilon_t,x_0) =  {\frac{1}{\sqrt{\alpha_t}} ( {x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} {\epsilon}_t )}$$


#### The model $f_\theta$:
$$x_T \sim N(0,1), x_{T-1}|x_{T},x_0 \sim N(\mu_\theta(x_T,x_0),\sigma_{\theta}(x_T,x_0)),\ldots, x_{t-1}|x_t \sim N(\mu_\theta(x_t,x_0),\sigma_{\theta}(x_t,x_0)),\ldots, x_0|x_1 \sim N(\mu_\theta(x_1,x_0),\sigma_{\theta}(x_1,x_0))$$

Now instead of one $KL$ between the true posterior distribution, we need more. We take $q=p$ to be the real posterior.
$$\min KL(q(x_{1:T}|x_0)|p_\theta(x_{1:T}|x_0)|)$$
is equivalent to maximize the ELBO
$$\mathcal{L} = \mathbb{E}_{q(\cdot |x_0)}\left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]$$
after some algebra (see Weng's blog)
$$\mathcal{L} = L_0 + L_1 + \ldots + L_T$$
where
$$L_T = KL(q(x_T|x_0)|| p_\theta(x_T,x_0),~L_t = KL(q(x_{t}|x_{t+1},x_0)||p_\theta(x_{t}|x_{t+1},x_0)), 1\le t\le T-1, ~ L_0=-\log p_\theta(x_0|x_1,x_0).$$


The KL divergence between these two Gaussians,
$$L_t = C(t) \|\mu_t - \mu_{\theta,t}\|_{L^2}$$
if we take $\sigma_\theta=1$, and introducing $\epsilon_{\theta,t} = (x_t - \sqrt{\alpha_t}\mu_\theta)\frac{\sqrt{1-\bar\alpha_t}}{1-\alpha_t}$, it becomes
$$L_t = C(t) \|\epsilon_t - \epsilon_{\theta,t}\|_{L^2}$$

Adding them together gives the loss function.


Note here $q_\phi$ is taken to be the real posterior distribution.

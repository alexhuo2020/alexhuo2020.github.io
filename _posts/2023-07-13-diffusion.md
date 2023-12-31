# Diffusion model 3: training images - a unet structure

We will analyze the code from https://github.com/openai/improved-diffusion.

## The UNET
The unet was introduced in [https://arxiv.org/abs/1505.04597] for image segmentation. Here we need to include the time variable as input.

### The whole structure of unet
Input: x with shape (b, c, dims); t with shape (b,t_dims)

Unet consists of three blocks: downsample block, middle block and upsample block

  * input block: downsample $x_0\to x_1\to x_2\to ...\to x_m$
  * middle block: $x_m \to y_m$ with same dimension
  * upsample block: concat($x_k,y_k$) $\to$ $y_{k-1}$

```
def forward(self,x,timesteps):
 hs = []
 emb = embedding of timesteps
 for module in self.input_blocks:
  h = module(h, emb)
  hs.append(h)
 h = self.middle_block(h,emb)
 for module in self.output_blocks:
  cat_in = torch.cat([h, hs.pop()],dim=1)
  h = module(cat_in, emb)
return result
```

### The time embdedding
Use a sin position embedding, introduced in [https://arxiv.org/abs/1706.03762].

$$\begin{aligned}
  \vec{p_t}^{(i)} = f(t)^{(i)} & :=
  \begin{cases}
      \sin({\omega_k} . t),  & \text{if}\  i = 2k \\
      \cos({\omega_k} . t),  & \text{if}\  i = 2k + 1
  \end{cases}
\end{aligned}$$

where
$$\omega_k = \frac{1}{10000^{2k / d}}$$

see the blog [https://kazemnejad.com/blog/transformer_architecture_positional_encoding/] for an explanation.

```
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
```
We can test using the above code.
```
import numpy as np
import torch as th
import math
timesteps = th.linspace(0,10,50)
xx = th.zeros_like(x)
for i in range(0, len(xx[0])):
    if i % 2:
        xx[:,i] = x[:,int(i/2)]
    else :
        xx[:,i] = x[:,int(i/2+64)]
plt.imshow(xx)
```
![image](https://github.com/alexhuo2020/alexhuo2020.github.io/assets/136142213/a6d49a04-d993-4ccc-b6c2-0de36d2bac88)

### A trick to include time variable as input
In building models, sometimes the input is x and sometimes is (x,t). [1] uses a sequential model to achieve this.
```
from abc import abstract method
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x 
```

### How to downsample and upsample
One can use Conv with stride/averge pooling/interpolation to do down sampling and ConvTranspose/interpolate to do upsampling. 
Here ref [1] use 

```
class Upsample(nn.Module):
...
  F.interpolate(x, scale_factor=2, mode="nearest")
class Downsample(nn.Module):
...
  if use_conv:
    self.op = nn.Conv2d(channels, channels, 3, stride=2,padding=1)
  else:
    self.op = nn.AvgPool2d(stride=2)
```

### Building blocks, ResNet and Attention
ResNet structure:
  * in_layers: normalization -> SiLU -> conv
  * emb_layers: SiLU -> linear
  * out_layer: normlization -> SiLU -> conv(zero_module)
  * output = out_layer(in_layer(x) + emb_layer(t)) + x

```
class ResBlock(TimestepBlock):
  def __init__(self, channels, emb_channels, out_channels):
      super().__init__()
      ....
      self.in_layers = nn.Sequential(normalization(channels), SiLU(), nn.Conv2d(channels, out_channels,padding=1))
      self.emb_layers = nn.Sequential(SiLU(), nn.Linear(emb_channels, out_channels))
      self.out_layers = nn.Sequential(normalization(out_channels, SiLU(), zero_module(nn.Conv2d(channels, out_channels,padding=1)))
      if out_channels == channels:
        self.skip_connection = nn.Identity()
      else:
        self.skip_connection = nn.Conv2d(channels, out_channels,padding=1)
  def forward(self, x, emb):
     h = self.in_layers(x)
     emb_out = self.emb_layers(emb)

     while len(emb_out.shape) < len(h.shape):
       emb_out = emb_out[...,None]
     h = h + emb_out
     return self.skip_connection(x) + self.out_layers(h)
```


AttentionBlock structure:
  * Apply self-attention
  * output = x + MultiheadAtten(x)

### Building UNET
* input_blocks: conv(x), ([ResBlock(ch0)]*m + [AttentionBlock(ch0)]+Downsample) +  ([ResBlock(ch1)]*m + [AttentionBlock(ch1)+Downsample]) + ... +   ([ResBlock(chN)]*m + [AttentionBlock(chN)])
* middle_block: ResBlock + AttentionBlock + ResBlock
* output_blocks: [ResBlock(chN)*(m+1) + AttentionBlock(chN) + Upsample] + ... + [ResBlock(chN)*(m+1) + AttentionBlock(chN) + Upsample]
* out: normalization -> SiLU -> conv

The input blocks
```
# channel_mult = [1,2,4,8]
self.input_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d...)])
for level, mult in enumerate(channel_mult):
   layers = [ResBlock] * num_of_resblock
   ch = mult*model_channels
   if ds in attention_resolutions:
     layers.append(AttentionBlock...)
   input_block_channs.append(ch)
   if level != len(channel_mult) - 1:
     layer.append(TimestepEmbedSequential(Downsample...)
     input_block_channs.append(ch)
     ds = ds*2
```
The middle blocks 
```
self.middle_block = TimestepEmbedSequential(ResBlock, AttentionBlock, ResBlock)
```
The output blocks
```
self.output_blocks = nn.ModuleList([])
for level, mult in list(enumerate(channel_mult))[::-1]:
 for i in range(num_res_blocks +1):
   layers = [ResBlock] 
   ch = mult*model_channels
   if ds in attention_resolutions:
     layers.append(AttentionBlock...)
   if level and i == num_res_blocks:
     layers.append(Upsample)
     ds //=2
   self.output_blocks.append(TimestepEmbedSequential(*layers))
```
out
```
self.out = nn.Sequential(normalization, SiLU, zero_module(conv...))
```

### The zero module
Used to make the parameter zero at initialization.
```
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
```












Refs

[1] Nichol, Alexander Quinn, and Prafulla Dhariwal. "Improved denoising diffusion probabilistic models." International Conference on Machine Learning. PMLR, 2021.
[2] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).


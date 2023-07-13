# Diffusion model 3: training images

We will analyze the code from https://github.com/openai/improved-diffusion.

## The UNET
The unet was introduced in [https://arxiv.org/abs/1505.04597] for image segmentation. Here we need to include the time variable as input.

### The whole structure of unet
Input: x with shape (b, c, dims); t with shape (b,t_dims)

Unet consists of three blocks: downsample block, middle block and upsample block

  * input block: downsample $x_0\to x_1\to x_2\to ...\to x_m$
  * middle block: $x_m$ ->$y_m$ with same dimension
  * upsample block: concat($x_k,y_k$) $\to$ $y_{k-1}$

'''
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
'''

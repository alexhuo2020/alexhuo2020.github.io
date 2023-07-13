# Diffusion model 3: training images

We will analyze the code from https://github.com/openai/improved-diffusion.

## The UNET
The unet was introduced in [https://arxiv.org/abs/1505.04597] for image segmentation. Here we need to include the time variable as input.

### The whole structure of unet
Input: x with shape (b, c, dims); t with shape (b,t_dims)

Unet consists of three blocks: downsample block, middle block and upsample block

input block: downsample c->c/2 -> c/4 -> c/8 ... ($x_0->x_1->x_2->...-> x_m$)
middle block: x_m ->x_m with same dimension
upsample block: concat($x_k$,$y_k$) -> $y_{k+1}$

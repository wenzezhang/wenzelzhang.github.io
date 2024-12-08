# Denoising Diffusion Probabilistic Model (DDPM)
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

image_size = 128
transform = Compose([
    Resize(image_size),
    ToTensor(), # turn into Numpy array of shape HWC, divide by 255
    Lambda(lambda t: (t * 2) - 1), # [0,1] --> [-1,1]
])

reverse_transform = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
])

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

timesteps = 200

# compute betas
betas = linear_beta_schedule(timesteps=timesteps)

# compute alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

# calculations for the forward diffusion q(x_t | x_{t-1})
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)


# utility function to extract the appropriate t index for a batch of indices.
# e.g., t=[10,11], x_shape=[b,c,h,w] --> a.shape = [2,1,1,1]
# e.g., t=[7,12,15,20], x_shape=[b,h,w] --> a.shape = [4,1,1]
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)  # z (it does not depend on t!)

    # adjust the shape
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def get_noisy_image(x_start, t):
    x_noisy = q_sample(x_start, t=t)
    noisy_image = reverse_transform(x_noisy)
    return noisy_image


# Backward process
# q(xt−1|xt, x0)=N(μ~(xt,x0), β~t)=N(μ~t, β~t), 用一个NN来学习~(xt) and σ~(xt)
from unet import Unet

temp_model = Unet(
    dim=image_size,
    channels=3,
    dim_mults=(1, 2, 4)
)
x_start = None
with torch.no_grad():
    # 输入图片和时间步，输出t时刻的噪声
    out = temp_model(x_start, torch.tensor([40]))
# x_{t-1} + z_{t-1} = x_t
# x_t - z_t = x_{t-1}
# loss || z_{t-1} -z_{t} ||_1

def p_losses(denoise_model, x_start, t, loss_type="huber"):
    # random sample z
    noise = torch.randn_like(x_start)

    # compute x_t
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)

    # recover z from x_t with the NN
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss
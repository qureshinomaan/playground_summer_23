#### Link to diffusers tutorial : https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline

import torch
from diffusers import DDPMScheduler, UNet2DModel

unet = UNet2DModel()

#### print number of parameters in unet
print(sum(p.numel() for p in unet.parameters() if p.requires_grad))

#### give random input to unet 
inp = torch.rand(1, 3, 256, 256)
output = unet(inp, 0)

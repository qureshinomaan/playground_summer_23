import torch
import wandb
from diffusers import DDPMScheduler, UNet2DModel

in_channels = 1
out_channels = 1
device = 'cuda'
model = UNet2DModel(
        sample_size=32,  # the target image resolution
        in_channels=1,  # the number of input channels, 3 for RGB images
         out_channels=1,  # the number of output channels
         layers_per_block=2,  # how many ResNet layers to use per UNet block
         block_out_channels=(128, 128, 128, 128, 128, 128),  # the number of output channels for each UNet block
         down_block_types=(
             "DownBlock2D",  # a regular ResNet downsampling block
             "DownBlock2D",
             "DownBlock2D",
             "DownBlock2D",
             "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
             "DownBlock2D",
         ),
         up_block_types=(
             "UpBlock2D",  # a regular ResNet upsampling block
             "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
             "UpBlock2D",
             "UpBlock2D",
             "UpBlock2D",
             "UpBlock2D",
         ),
     ).to(device)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# PATH = './diffusion_mnist_epoch_93.pth'
# model.load_state_dict(torch.load(PATH))
#### print number of parameters in unet
print(sum(p.numel() for p in model.parameters() if p.requires_grad))


in_channels = 1
out_channels = 1
device = 'cuda'
model = UNet2DModel(
        sample_size=32,  # the target image resolution
        in_channels=1,  # the number of input channels, 3 for RGB images
         out_channels=1,  # the number of output channels
         layers_per_block=2,  # how many ResNet layers to use per UNet block
         block_out_channels=(128, 128, 128, 128, 128, 128),  # the number of output channels for each UNet block
         down_block_types=(
             "DownBlock2D",  # a regular ResNet downsampling block
             "DownBlock2D",
             "DownBlock2D",
             "DownBlock2D",
             "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
             "DownBlock2D",
         ),
         up_block_types=(
             "UpBlock2D",  # a regular ResNet upsampling block
             "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
             "UpBlock2D",
             "UpBlock2D",
             "UpBlock2D",
             "UpBlock2D",
         ),
     ).to(device)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# PATH = './diffusion_mnist_epoch_93.pth'
# model.load_state_dict(torch.load(PATH))
#### print number of parameters in unet
print(sum(p.numel() for p in model.parameters() if p.requires_grad))


#### Setup the dataloader
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
batch_size = 128

train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data = torch.nn.functional.pad(example_data,(2, 2, 2, 2)).to(device)
model(example_data, 0)  

#### Defining the training parameters
learning_rate = 1e-4
num_epochs = 100

from diffusers.optimization import get_cosine_schedule_with_warmup
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                               num_training_steps=(len(train_loader) * num_epochs),
                                               num_warmup_steps=100)


#### Training loop
import time 
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        #### Padding the mnist images to make them 32x32
        padded_data = torch.nn.functional.pad(data,(2, 2, 2, 2))

        #### Sample noise 
        noise = torch.randn(data.shape[0], in_channels, model.sample_size, model.sample_size, device=device)

        #### timesteps from diffusers
        timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (data.shape[0],), device=device).long()

        #### add noise to the data
        noisy_data = noise_scheduler.add_noise(padded_data, noise, timesteps)

        #### predict the noise 
        noise_pred = model(noisy_data, timesteps).sample

        #### calculate the loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        #### backpropagate the loss
        loss.backward()

        #### update the parameters
        optimizer.step()
        lr_scheduler.step()

    #### print the epoch and loss
    print('epoch: ', epoch, '  loss: ', loss.item())
    PATH = './diffusion_mnist_epoch_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), PATH)

#### Link to diffusers tutorial : https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline

import torch
from diffusers import DDPMScheduler, UNet1DModel

in_channels = 1
out_channels = 1
device = 'cuda'
model = UNet1DModel(sample_size=1024, in_channels=17, out_channels=out_channels).to(device)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

#### print number of parameters in unet
print(sum(p.numel() for p in model.parameters() if p.requires_grad))


#### Load the mnist dataset
import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)



#### Setup the dataloader
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
batch_size = 256

train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(torch.nn.functional.pad(example_data,(2, 2, 2, 2)).shape)

#### Defining the training parameters
learning_rate = 1e-4
num_epochs = 10

from diffusers.optimization import get_cosine_schedule_with_warmup
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                               num_training_steps=(len(train_loader) * num_epochs),
                                               num_warmup_steps=100)


#### Training loop
for epoch in range(num_epochs):
    print('epoch: ', epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        #### Padding the mnist images to make them 32x32
        padded_data = torch.nn.functional.pad(data,(2, 2, 2, 2))
        #### Chaning the shape of data to (batch_size, channels, sample_size)
        padded_data = padded_data.reshape(data.shape[0], in_channels, model.sample_size)

        #### Sample noise 
        noise = torch.randn(data.shape[0], in_channels, model.sample_size, device=device)

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


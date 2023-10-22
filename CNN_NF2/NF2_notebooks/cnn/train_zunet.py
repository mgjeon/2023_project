import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler

from tool.model import zUnet
from tool.load import *
from tool.evaluate import *
from tool.dataset import *

batches_file_paths = {'inputs': 'inputs.npy', 'outputs':'outputs.npy'}

batch_size = 1
iterations = 10000

dataset = CustomDataset(batches_file_paths, batch_size=batch_size)
dataloaer = DataLoader(dataset, batch_size=None, num_workers=8, pin_memory=True,
                       sampler=RandomSampler(dataset, replacement=True, num_samples=iterations))

model = zUnet().to('cuda')

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=1e-3)

import wandb
wandb.init()
wandb.watch(model, log_freq=10)
model.train()
# for epoch in range(iterations):
for batch_idx, samples in enumerate(dataloaer):
    inputs = samples['inputs']
    labels = samples['outputs']
    inputs = torch.Tensor(inputs).to('cuda')
    labels = torch.Tensor(labels).to('cuda')

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if (batch_idx+1) % 10 == 0:
        model.eval()
        wandb.log({"loss": loss})
        bb = outputs.cpu().detach().numpy().squeeze().transpose(1, 2, 3, 0)
        BB = labels.cpu().detach().numpy().squeeze().transpose(1, 2, 3, 0)
        zz = np.random.randint(low=0, high=49, size=1)[0]
        b_norm = 1/zz if zz != 0 else 1
        fig = plot_overview(bb, BB, z=zz, b_norm=b_norm, ret=True)
        wandb.log({"img": fig})
    if (batch_idx+1) % 1000 == 0:
        path = f"model/model_{batch_idx+1}.pt"
        torch.save({'epoch': batch_idx+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss':loss}, path)


import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler

from tool.model import small_zUnet
from tool.load import *
from tool.evaluate import *
from tool.dataset import *

model_path = 'model_piz'
os.makedirs(model_path, exist_ok=True)

batches_file_paths = {'inputs': 'inputs.npy', 'outputs':'outputs.npy'}

batch_size = 1
iterations = 10000

dataset = CustomDataset(batches_file_paths, batch_size=batch_size)
dataloaer = DataLoader(dataset, batch_size=None, num_workers=8, pin_memory=True,
                       sampler=RandomSampler(dataset, replacement=True, num_samples=iterations))

model = small_zUnet().to('cuda')

# criterion = nn.MSELoss()
from tool.diff import *
def criterion(outputs, labels):
    b = torch.permute(outputs, (0, 2, 3, 4, 1))
    B = torch.permute(labels, (0, 2, 3, 4, 1))
    
    loss_bc = b[..., 0, :] - B[..., 0, :]
    loss_bc = torch.mean(loss_bc**2)

    bx = b[..., 0]
    by = b[..., 1]
    bz = b[..., 2]

    jx, jy, jz = curl(bx, by, bz, 1, 1, 1)
    
    b = torch.stack([bx, by, bz], -1)
    j = torch.stack([jx, jy, jz], -1)

    jxb = torch.cross(j, b, -1)
    loss_ff = (jxb**2).sum(-1)
    loss_ff = torch.mean(loss_ff)

    div_b = divergence(bx, by, bz, 1, 1, 1)
    loss_div = div_b**2
    loss_div = torch.mean(loss_div)

    return loss_bc, loss_ff, loss_div

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
    loss_bc, loss_ff, loss_div = criterion(outputs, labels)
    loss = loss_bc + loss_ff + loss_div
    loss.backward()
    optimizer.step()

    if (batch_idx+1) % 10 == 0:
        model.eval()
        wandb.log({"loss": loss, "loss_bc":loss_bc, "loss_ff":loss_ff, "loss_div":loss_div})
        bb = outputs.cpu().detach().numpy().squeeze().transpose(1, 2, 3, 0)
        BB = labels.cpu().detach().numpy().squeeze().transpose(1, 2, 3, 0)
        zz = np.random.randint(low=0, high=49, size=1)[0]
        b_norm = 1/zz if zz != 0 else 1
        fig = plot_overview(bb, BB, z=zz, b_norm=b_norm, ret=True)
        wandb.log({"img": fig})
    if (batch_idx+1) % 1000 == 0:
        path = f"{model_path}/model_{batch_idx+1}.pt"
        torch.save({'epoch': batch_idx+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss':loss}, path)


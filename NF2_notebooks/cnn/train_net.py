import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler

from tool.model import net
from tool.load import *
from tool.evaluate import *
from tool.dataset import *

# file_list = nc_list("/mnt/obsdata/isee_nlfff_v1.2/12673")

# from tqdm import tqdm

# inputs = []
# outputs = []
# for file in tqdm(file_list[::40]):
#     input_p, output_p = create_pair(file)
#     inputs.append(input_p)
#     outputs.append(output_p)

# inputs = np.array(inputs).astype(np.float32)
# outputs = np.array(outputs).astype(np.float32)

# np.save('inputs.npy', inputs)
# np.save('outputs.npy', outputs)

batches_file_paths = {'inputs': 'inputs.npy', 'outputs':'outputs.npy'}

batch_size = 1
iterations = 10000

dataset = CustomDataset(batches_file_paths, batch_size=batch_size)
dataloaer = DataLoader(dataset, batch_size=None, num_workers=8, pin_memory=True,
                       sampler=RandomSampler(dataset, replacement=True, num_samples=iterations))

model = net().to('cuda')

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=1e-3)

import wandb
wandb.init()
wandb.watch(model, log_freq=10)
model.train()
# for epoch in range(iterations):
for batch_idx, samples in enumerate(dataloaer):
    inputs = samples['inputs']
    outputs = samples['outputs']
    inputs = inputs.to('cuda')
    labels = outputs.to('cuda')

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
        fig = plot_overview(bb, BB, b_norm=1, ret=True)
        wandb.log({"img": fig})
    if (batch_idx+1) % 1000 == 0:
        path = f"model/model_{batch_idx+1}.pt"
        torch.save({'epoch': batch_idx+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss':loss}, path)


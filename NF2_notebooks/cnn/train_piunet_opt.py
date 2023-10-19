import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
# from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, RandomSampler

from tool.model import Unet
from tool.load import *
from tool.evaluate import *
from tool.dataset import *

desc = 'piunet_opt_b3_elr_w10'
model_path = f'model_{desc}'
os.makedirs(model_path, exist_ok=True)

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

batch_size = 3
iterations = 10000

dataset = CustomDataset(batches_file_paths, batch_size=batch_size)
dataloaer = DataLoader(dataset, batch_size=None, num_workers=8, pin_memory=True,
                       sampler=RandomSampler(dataset, replacement=True, num_samples=iterations))

model = Unet().to('cuda')

# criterion = nn.MSELoss()
from tool.diff import *
def criterion(outputs, labels):
    b = torch.permute(outputs, (0, 3, 2, 1, 4))
    B = torch.permute(labels, (0, 3, 2, 1, 4))
    
    loss_bc = b[..., 0, :] - B[..., 0, :]
    loss_bc = torch.mean(loss_bc**2)

    bx = b[..., 0]
    by = b[..., 1]
    bz = b[..., 2]

    jx, jy, jz = curl(bx, by, bz, 1, 1, 1)
    
    b = torch.stack([bx, by, bz], -1)
    j = torch.stack([jx, jy, jz], -1)

    jxb = torch.cross(j, b, -1)
    loss_ff = (jxb**2).sum(-1) / ((b**2).sum(-1) + 1e-7)
    loss_ff = torch.mean(loss_ff)

    div_b = divergence(bx, by, bz, 1, 1, 1)
    loss_div = div_b**2
    loss_div = torch.mean(loss_div)

    return loss_bc, loss_ff, loss_div

lr_start = 1e-4
lr_end = 1e-5
decay_iterations = 10000

gamma = (lr_end / lr_start) ** (1 / decay_iterations)

optimizer = Adam(model.parameters(), lr=lr_start)
scheduler = ExponentialLR(optimizer, gamma=gamma)

import wandb
wandb.init(project='cnn', entity='mgjeon', name=desc)
model.train()
# for epoch in range(iterations):
for batch_idx, samples in enumerate(dataloaer):
    inputs = np.array(samples['inputs']).transpose(0, 4, 3, 2, 1).astype(np.float32)
    labels = np.array(samples['outputs']).transpose(0, 4, 3, 2, 1).astype(np.float32)
    inputs = torch.Tensor(inputs).to('cuda')
    labels = torch.Tensor(labels).to('cuda')

    optimizer.zero_grad()
    outputs = model(inputs)
    loss_bc, loss_ff, loss_div = criterion(outputs, labels)
    loss = loss_bc + 10*loss_ff + 10*loss_div
    loss.backward()
    optimizer.step()

    if scheduler.get_last_lr()[0] > lr_end:
        scheduler.step()

    if (batch_idx+1) % 10 == 0:
        model.eval()
        wandb.log({"loss": loss, "loss_bc":loss_bc, "loss_ff":loss_ff, "loss_div":loss_div, "learning_rate": scheduler.get_last_lr()[0]})
        bb = outputs.cpu().detach().numpy()[0, ...].transpose(2, 1, 0, 3)
        BB = labels.cpu().detach().numpy()[0, ...].transpose(2, 1, 0, 3)
        zz = np.random.randint(low=0, high=49, size=1)[0]
        b_norm = 1/zz if zz != 0 else 1
        fig = plot_overview(bb, BB, z=zz, b_norm=b_norm, ret=True)
        wandb.log({"img": fig})
    if (batch_idx+1) % 1000 == 0:
        path = f"{model_path}/model_{batch_idx+1}.pt"
        torch.save({'epoch': batch_idx+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss':loss}, path)


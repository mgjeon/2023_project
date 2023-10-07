import torch
from torch import nn
import numpy as np
from tqdm import tqdm

def load_cube(save_path, device=None, z=None, strides=1, **kwargs):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    state = torch.load(save_path, map_location=device)
    model = nn.DataParallel(state['model'])
    cube_shape = state['cube_shape']
    z = z if z is not None else cube_shape[2]
    coords = np.stack(np.mgrid[:cube_shape[0]:strides, :cube_shape[1]:strides, :z:strides], -1)
    return load_coords(model, cube_shape, state['spatial_norm'],
                       state['b_norm'], coords, device, **kwargs)

def load_coords(model, cube_shape, spatial_norm, b_norm, coords, device, batch_size=1024, progress=False):

    def _load(coords):
        coords = torch.tensor(coords / spatial_norm, dtype=torch.float32)
        coords_shape = coords.shape

        coords = coords.reshape((-1, 3))
        cube = []
        it = range(int(np.ceil(coords.shape[0] / batch_size)))
        it = tqdm(it) if progress else it
        for k in it:
            coord = coords[k * batch_size : (k+1) * batch_size]
            coord = coord.to(device)
            coord.requires_grad = True 
            cube += [model(coord).detach().cpu()]
        
        cube = torch.cat(cube)
        cube = cube.view(*coords_shape).numpy()
        b = cube * b_norm 
        return b 

    with torch.no_grad():
        return _load(coords)
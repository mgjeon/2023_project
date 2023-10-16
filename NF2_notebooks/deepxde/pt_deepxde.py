import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['DDE_BACKEND'] = 'pytorch'

import deepxde as dde
import numpy as np

spatial_norm = 256
b_norm = 2500

geom = dde.geometry.Cuboid(xmin=[0, 0, 0], xmax=[2, 1, 1])

coords = np.load('coords.npy')/spatial_norm
values = np.load('values.npy')/b_norm

pf_coords = np.load('pf_coords.npy')/spatial_norm
pf_values = np.load('pf_values.npy')/b_norm

coords = np.concatenate([pf_coords, coords])
values = np.concatenate([pf_values, values])

bc_bottom_Bx = dde.icbc.PointSetBC(coords, values[:, 0].reshape(-1, 1), component=0, batch_size=5000)
bc_bottom_By = dde.icbc.PointSetBC(coords, values[:, 1].reshape(-1, 1), component=1, batch_size=5000)
bc_bottom_Bz = dde.icbc.PointSetBC(coords, values[:, 2].reshape(-1, 1), component=2, batch_size=5000)

def pde(X, Y):
    eps = 1e-7
    
    Bx = Y[:, 0:1]
    By = Y[:, 1:2]
    Bz = Y[:, 2:3]

    norm_B = (Bx**2) + (By**2) + (Bz**2)
    
    dBx_x = dde.grad.jacobian(Y, X, i=0, j=0)
    dBx_y = dde.grad.jacobian(Y, X, i=0, j=1)
    dBx_z = dde.grad.jacobian(Y, X, i=0, j=2)

    dBy_x = dde.grad.jacobian(Y, X, i=1, j=0)
    dBy_y = dde.grad.jacobian(Y, X, i=1, j=1)
    dBy_z = dde.grad.jacobian(Y, X, i=1, j=2)

    dBz_x = dde.grad.jacobian(Y, X, i=2, j=0)
    dBz_y = dde.grad.jacobian(Y, X, i=2, j=1)
    dBz_z = dde.grad.jacobian(Y, X, i=2, j=2)

    Jx = dBz_y - dBy_z
    Jy = dBx_z - dBz_x
    Jz = dBy_x - dBx_y

    JxB_x = Jy*Bz - Jz*By
    JxB_y = Jz*Bx - Jx*Bz
    JxB_z = Jx*By - Jy*Bx
    
    norm_JxB = (JxB_x**2) + (JxB_y**2) + (JxB_z**2)

    loss_JxB = norm_JxB / (norm_B + eps)

    divB = dBx_x + dBy_y + dBz_z

    norm_divB = (divB**2)

    return [norm_JxB, norm_divB]

data = dde.data.PDE(geom,
                    pde,
                    [bc_bottom_Bx, bc_bottom_By, bc_bottom_Bz],
                    num_domain=20000,
                    num_boundary=10000,
                    num_test=10000)

layer_size = [3] + [256] * 8 + [3]
activation = 'sin'
initializer = 'Glorot normal'

net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

lr_start = 5e-4
lr_end = 5e-5
decay_iterations = 100000
gamma=(lr_end / lr_start) ** (1 / decay_iterations)

model.compile('adam', lr=5e-4, loss_weights=[1, 1, 1000, 1000, 1000], decay=("exponential", gamma))

pde_resampler = dde.callbacks.PDEPointResampler(bc_points=True)

checkpointer = dde.callbacks.ModelCheckpoint(
    "model3/nf2", verbose=1, save_better_only=True 
)

loss_history, train_state = model.train(iterations=100000, callbacks=[pde_resampler, checkpointer])

dde.saveplot(loss_history, train_state, issave=True, isplot=False)


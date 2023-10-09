# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['DDE_BACKEND'] = 'pytorch'

# %%
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# %%
spatial_norm = 256
b_norm = 2500

# %%
geom = dde.geometry.Cuboid(xmin=[0, 0, 0], xmax=[2, 1, 1])

# %%
coords = np.load('coords.npy')
values = np.load('values.npy')

pf_coords = np.load('pf_coords.npy')
pf_values = np.load('pf_values.npy')

# %%
coords.shape, values.shape, pf_coords.shape, pf_values.shape

# %%
bc_bottom_Bx = dde.icbc.PointSetBC(coords/spatial_norm, values[:, 0].reshape(-1, 1)/b_norm, component=0, batch_size=5000)
bc_bottom_By = dde.icbc.PointSetBC(coords/spatial_norm, values[:, 1].reshape(-1, 1)/b_norm, component=1, batch_size=5000)
bc_bottom_Bz = dde.icbc.PointSetBC(coords/spatial_norm, values[:, 2].reshape(-1, 1)/b_norm, component=2, batch_size=5000)

# %%
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

    loss_divB = (divB**2)

    return [loss_JxB, loss_divB]

# %%
data = dde.data.PDE(geom,
                    pde,
                    [bc_bottom_Bx, bc_bottom_By, bc_bottom_Bz],
                    num_domain=20000,
                    num_boundary=10000,
                    num_test=10000)

# %%
plt.scatter(data.train_x_bc[:, 0], data.train_x_bc[:, 1], s=1)
plt.xlabel('x')
plt.ylabel('y')

# %%
plt.scatter(data.train_x_all[:, 0], data.train_x_all[:, 1], s=1)
plt.xlabel('x')
plt.ylabel('y')

# %%
layer_size = [3] + [256] * 8 + [3]
activation = 'sin'
initializer = 'Glorot normal'

net = dde.nn.FNN(layer_size, activation, initializer)

# %%
model = dde.Model(data, net)

# %%
lr_start = 5e-4
lr_end = 5e-5
decay_iterations = 100000
gamma=(lr_end / lr_start) ** (1 / decay_iterations)

# %%
model.compile('adam', lr=5e-4, loss_weights=[1, 1, 1000, 1000, 1000], decay=("exponential", gamma))

# %%
pde_resampler = dde.callbacks.PDEPointResampler(bc_points=True)

# %%
checkpointer = dde.callbacks.ModelCheckpoint(
    "model/nf2", verbose=1, save_better_only=True 
)

# %%
loss_history, train_state = model.train(iterations=100000, callbacks=[pde_resampler, checkpointer])

# %%
dde.saveplot(loss_history, train_state, issave=False, isplot=True)

# %%
B = model.predict(coords/spatial_norm)*b_norm

# %%
b_slices = B.reshape(513, 257, 1, 3)

# %%
for i in range(b_slices.shape[2]):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(b_slices[..., i, 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    axs[1].imshow(b_slices[..., i, 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    axs[2].imshow(b_slices[..., i, 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    plt.show()

# %%




import numpy as np

def vector_norm(vector):
    return np.linalg.norm(vector, axis=-1)

def dot_product(a, b):
    return (a*b).sum(-1)


def metric(b, B):
    # b : model solution
    # B : reference magnetic field
    eps = 1e-7

    result = {}

    result['c_vec'] = np.sum(dot_product(B, b)) / np.sqrt(np.sum(vector_norm(B)**2) * np.sum(vector_norm(b)**2))
    
    M = np.prod(B.shape[:-1])
    eps = 1e-7
    nu = dot_product(B, b)
    de = vector_norm(B) * vector_norm(b)
    result['c_cs'] = (1 / M) * np.sum(np.divide(nu, de, where=de!=0.))
    result['c_cs_ep'] = (1 / M) * np.sum(nu/(de + eps))

    E_n = np.sum(vector_norm(b - B)) / np.sum(vector_norm(B))
    result["E_n'"] = 1 - E_n
    
    nu = vector_norm(b - B)
    de = vector_norm(B)
    E_m = (1 / M) * np.sum(np.divide(nu, de, where=de!=0.))
    result["E_m'"] = 1 - E_m
    E_m = (1 / M) * np.sum((nu/(de + eps)))
    result["E_m'_ep"] = 1 - E_m

    result['eps'] = np.sum(vector_norm(b)**2) / np.sum(vector_norm(B)**2)

    return result

import matplotlib.pyplot as plt 

def plot_overview(b, B, z=0, b_norm=2500, ret=False):
    fig, axs = plt.subplots(2, 3, figsize=(12, 4))

    ax = axs[0]
    ax[0].imshow(b[..., z, 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    ax[0].set_title('bx')
    ax[1].imshow(b[..., z, 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    ax[1].set_title('by')
    ax[2].imshow(b[..., z, 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    ax[2].set_title('bz')

    ax = axs[1]
    ax[0].imshow(B[..., z, 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    ax[0].set_title('Bx')
    ax[1].imshow(B[..., z, 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    ax[1].set_title('By')
    ax[2].imshow(B[..., z, 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    ax[2].set_title('Bz')

    fig.suptitle(f'z={z}')
    
    plt.tight_layout()

    if ret:
        plt.close()
        return fig
    else:
        plt.show()

def plot_s(mag, title, n_samples):
    fig, axs = plt.subplots(3, n_samples, figsize=(n_samples * 4, 12))
    heights = np.linspace(0, 1, n_samples) ** 2 * (mag.shape[2] - 1)  # more samples from lower heights
    heights = heights.astype(np.int32)
    for i in range(3):
        for j, h in enumerate(heights):
            v_min_max = np.max(np.abs(mag[:, :, h, i]))
            axs[i, j].imshow(mag[:, :, h, i].transpose(), cmap='gray', vmin=-v_min_max, vmax=v_min_max,
                            origin='lower')
            axs[i, j].set_axis_off()
    for j, h in enumerate(heights):
        axs[0, j].set_title('%.01f' % h)
    fig.tight_layout()
    fig.suptitle(title, fontsize=25)
    plt.show()

def plot_sample(b, B, n_samples=10):
    plot_s(b, 'b', n_samples)
    plot_s(B, 'B', n_samples)


class evaluator:
    def __init__(self, b, B):
        self.b = b 
        self.B = B

        self.result = self._metric()
        print(self.result)

        self._plot_overview(z=100, b_norm=100)
        self._plot_sample(n_samples=10)

    def _metric(self):
        return metric(self.b, self.B)
    
    def _plot_sample(self, n_samples=10):
        plot_sample(self.b, self.B, n_samples=n_samples)

    def _plot_overview(self, z=0, b_norm=2500):
        plot_overview(self.b, self.B, z=z, b_norm=b_norm)
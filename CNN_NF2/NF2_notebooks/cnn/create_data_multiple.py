from tool.load import *
from tool.evaluate import *
from tool.dataset import *

noaas = [11302, 11429, 11515, 12192, 12673]

# b_norm = 2500

for noaa in noaas:
    nc_path = f"/mnt/obsdata/isee_nlfff_v1.2/{noaa}"
    file_list = nc_list(nc_path)
    save_input_label(file_list, f'data/train_inputs_{noaa}.npy',\
                     f'data/train_labels_{noaa}.npy', f'data/train_labels_pot_{noaa}.npy')

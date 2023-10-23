# %%
from tool.load import *
from tool.evaluate import *
from tool.dataset import *

# %%
def create_pair_pot_bnorm_max(nc_file):
    b, bp = load_b_pot(nc_file)

    input_data = b[:, :, :, 0]
    input_data = input_data[:, :, :, None]
    input_max = np.max(np.abs(input_data))
    input_data = input_data / input_max
    output_data = b[:, :, :, :50]
    output_data = output_data
    output_data_p = bp[:, :, :, :50]
    output_data_p = output_data_p

    return input_max, input_data, output_data, output_data_p

# %%
file_list = nc_list("/mnt/obsdata/isee_nlfff_v1.2/12673")

# %%
from tqdm import tqdm

# %%
def save_input_label(file_list, inputs_max_path, inputs_path, labels_path, labels_pot_path):
    inputs_max = []
    inputs = []
    labels = []
    labels_pot = []
    for file in tqdm(file_list):
        input_max, input_p, output_p, output_p_pot = create_pair_pot_bnorm_max(file)
        inputs_max.append(input_max)
        inputs.append(input_p)
        labels.append(output_p)
        labels_pot.append(output_p_pot)

    inputs = np.array(inputs).astype(np.float32)
    labels = np.array(labels).astype(np.float32)
    labels_pot = np.array(labels_pot).astype(np.float32)

    np.save(inputs_max_path, inputs_max)
    np.save(inputs_path, inputs)
    np.save(labels_path, labels)
    np.save(labels_pot_path, labels_pot)

# %%
save_input_label(file_list[::40], 'data/bm_train_inputs_max.npy', 'data/bm_train_inputs.npy', 'data/bm_train_labels.npy', 'data/bm_train_labels_pot.npy')



# %%
from tool.load import *
from tool.evaluate import *
from tool.dataset import *

# %%
file_list = nc_list("/mnt/obsdata/isee_nlfff_v1.2/11302")
file_list += nc_list("/mnt/obsdata/isee_nlfff_v1.2/11429")
file_list += nc_list("/mnt/obsdata/isee_nlfff_v1.2/11515")
file_list += nc_list("/mnt/obsdata/isee_nlfff_v1.2/12192")

# len(file_list)

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test = train_test_split(file_list, test_size=0.10, random_state=42)

# %%
# len(X_train), len(X_test)

# %%
from tqdm import tqdm

# %%
def save_input_label(file_list, inputs_path, labels_path):
    inputs = []
    labels = []
    for file in tqdm(file_list):
        input_p, output_p = create_pair(file)
        inputs.append(input_p)
        labels.append(output_p)

    inputs = np.array(inputs).astype(np.float32)
    labels = np.array(labels).astype(np.float32)

    np.save(inputs_path, inputs)
    np.save(labels_path, labels)

# %%
save_input_label(X_train, 'data/train_inputs.npy', 'data/train_labels.npy')

# %%
save_input_label(X_test, 'data/test_inputs.npy', 'data/test_labels.npy')

# %%
# file_list[239]

# %%
# input_p, output_p = create_pair(file_list[239])

# np.save('12673_20170906_083600_input.npy', input_p)
# np.save('12673_20170906_083600_output.npy', output_p)

# %%




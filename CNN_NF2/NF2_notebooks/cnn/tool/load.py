import netCDF4
import numpy as np

def load_b(nc_file):
    nc=netCDF4.Dataset(nc_file, 'r')
    
    nc_bx=nc.variables['Bx']
    bx=nc_bx[:].transpose(2,1,0)
    nc_by=nc.variables['By']
    by=nc_by[:].transpose(2,1,0)
    nc_bz=nc.variables['Bz']
    bz=nc_bz[:].transpose(2,1,0)

    b = np.stack([bx, by, bz])
    b = np.array(b)
    return b

from pathlib import Path

def nc_list(nc_path):
    isee_datapath = Path(nc_path)
    filelist = sorted([x for x in isee_datapath.glob('*.nc')])
    return filelist

def create_pair(nc_file):
    b = load_b(nc_file)

    b_norm = 2500
    input_data = b[:, :, :, 0]
    input_data = input_data[:, :, :, None] / b_norm
    output_data = b[:, :, :, :50] / b_norm

    return input_data, output_data

def load_b_pot(nc_file):
    nc=netCDF4.Dataset(nc_file, 'r')
    
    nc_bx=nc.variables['Bx']
    bx=nc_bx[:].transpose(2,1,0)
    nc_by=nc.variables['By']
    by=nc_by[:].transpose(2,1,0)
    nc_bz=nc.variables['Bz']
    bz=nc_bz[:].transpose(2,1,0)
    b = np.stack([bx, by, bz])
    b = np.array(b)

    nc_bx=nc.variables['Bx_pot']
    bx=nc_bx[:].transpose(2,1,0)
    nc_by=nc.variables['By_pot']
    by=nc_by[:].transpose(2,1,0)
    nc_bz=nc.variables['Bz_pot']
    bz=nc_bz[:].transpose(2,1,0)
    bp = np.stack([bx, by, bz])
    bp = np.array(bp)
    return b, bp

def create_pair_pot(nc_file, b_norm=2500):
    b, bp = load_b_pot(nc_file)

    input_data = b[:, :, :, 0]
    input_data = input_data[:, :, :, None] / b_norm
    output_data = b[:, :, :, :50] / b_norm
    output_data_p = bp[:, :, :, :50] / b_norm

    return input_data, output_data, output_data_p

from tqdm import tqdm

def save_input_label(file_list, inputs_path, labels_path, labels_pot_path):
    inputs = []
    labels = []
    labels_pot = []
    for file in tqdm(file_list):
        input_p, output_p, output_p_pot = create_pair_pot(file)
        inputs.append(input_p)
        labels.append(output_p)
        labels_pot.append(output_p_pot)

    inputs = np.array(inputs).astype(np.float32)
    labels = np.array(labels).astype(np.float32)
    labels_pot = np.array(labels_pot).astype(np.float32)

    np.save(inputs_path, inputs)
    np.save(labels_path, labels)
    np.save(labels_pot_path, labels_pot)
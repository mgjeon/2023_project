import netCDF4
import numpy as np


class nlfff:
    def __init__(self, filename):
        self.filename = filename

        nc = netCDF4.Dataset(self.filename, "r")

        nc_x = nc.variables["x"]
        self.x = np.array(nc_x)
        nc_y = nc.variables["y"]
        self.y = np.array(nc_y)
        nc_z = nc.variables["z"]
        self.z = np.array(nc_z)

        nc_bx = nc.variables["Bx"]
        self.bx = np.array(nc_bx).transpose(2, 1, 0)
        nc_by = nc.variables["By"]
        self.by = np.array(nc_by).transpose(2, 1, 0)
        nc_bz = nc.variables["Bz"]
        self.bz = np.array(nc_bz).transpose(2, 1, 0)

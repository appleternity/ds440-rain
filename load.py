import numpy as np
import xarray
from netCDF4 import MFDataset

def option_1():
    # option 1
    nc_high = MFDataset('chirps-v2.0.*.nc')
    nc_high_ = nc_high.variables["precip"][:]
    print(nc_high_)

def option_2():
    # option 2
    nc_high = xarray.open_mfdataset("chirps-v2.0.*.nc", concat_dim="time", combine='nested')
    np_high = nc_high.as_numpy().astype(np.float16)
    print(np_high)
    print(np_high["precip"])

    # np_high["precip"].data
    np_high1 = np_high["precip"].values
    max_value = np.max(np_high1)
    print(np_high1.shape)
    print(type(np_high1))
    print("max_value", max_value)

    print("====================================")

    # np_high["precip"]
    max_value_2 = np.max(np_high["precip"])
    print(np_high["precip"].shape)
    print(type(np_high["precip"]))
    print("max_value_2", max_value_2)

if __name__ == "__main__":
    #option_1()
    option_2()


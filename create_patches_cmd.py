import numpy as np
import xarray
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
from sklearn import feature_extraction
import h5py
import joblib
import cv2
from tqdm import tqdm

from datetime import date

# TODO:
# (1) what does nan means? Should we fill in 0 for them? (The github repo fills 0)
# (2) numpy.ma (masked array) -> in the training data, patches containing the missing data should be ignored
#                             -> days that containing the missing data should be ignored

# (3) Upscaling for the low-resolution image is missing. 

# (4) imshow to visualize the image

# (5) probably removing missing data would be better

# (6) They assume 0 patch should not exist

# (7) train/valid/test ?

def get_num_days(year_1, year_2):
    d0 = date(year_1, 1, 1)
    d1 = date(year_2, 12, 31)
    delta = d1 - d0
    print(delta.days+1)
    return delta.days+1

def build_norm(offset):
    # set two years
    num_day = 730
    #offset = 11323 # starting 1981
    #offset = 20001

    start_year = 2010
    end_year = 2020

    #ds = xarray.open_mfdataset("chirps-v2.0.*.nc", concat_dim="time", combine='nested')
    filename_list = [
        f"data/chirps-v2.0.{year}.days_p05_CORDEX_CAM.nc"
        for year in range(start_year, end_year+1)
    ]
    num_day = get_num_days(start_year, end_year)
    offset = get_num_days(1950, start_year-1)

    print("start, end = ", start_year, end_year)
    print(filename_list)
    print("num_day", num_day)
    print("offset", offset)

    ds = xarray.open_mfdataset(filename_list, concat_dim="time", combine='nested', parallel=True)
    ds.to_netcdf("high_allYears.nc")

    nc_f_high = 'high_allYears.nc'
    nc_high = Dataset(nc_f_high, 'r')  # Dataset is the class behavior to open the file    
    print(nc_high)
    prcp_high = nc_high.variables['precip'][:]

    nc_f_low="precipitation_daily_1950-2020.nc"
    nc_low = Dataset(nc_f_low, 'r')  # Dataset is the class behavior to open the file
    print(nc_low)

    # 1950-01-01 - 1980-12-31 ==> 11323 days
    prcp_low = nc_low.variables['tp'][offset:offset+num_day]

    max_pixel_h=np.max(prcp_high)
    max_pixel_l=np.max(prcp_low)
    
    def normalise(sst, norm_constant):
      sst=sst/norm_constant
      return sst
    
    # TODO: they should probably be normalized using the same constant (?) 
    prcp_high_norm=normalise(prcp_high, max_pixel_h)
    prcp_low_norm=normalise(prcp_low, max_pixel_l)
   
    print("low.shape", prcp_low_norm.shape)
    print("high.shape", prcp_high_norm.shape)
    
    with open(f"prcp_norm-{offset}.joblib", 'wb') as outfile:
        joblib.dump({
            "high": prcp_high_norm,
            "low": prcp_low_norm,
        }, outfile)
   
def load_prcp_norm(offset=11323):
    with open(f"prcp_norm-{offset}.joblib", 'rb') as infile:
        data = joblib.load(infile)
        return data["high"], data["low"]

def load_prcp_scaled(offset=11323):
    with open(f"prcp_scaled-{offset}.joblib", 'rb') as infile:
        data = joblib.load(infile)
        return data["high"], data["low"]

def create_patches(patch_size, max_patches, sst_X, sst_Y, num_fields_end, total, num_fields_start=0):
  """this creates patches from the entire sst fields 
  num_fields_start is the start_index, like the start_day
  num_fields_end is the start_index, like the end_day"""

  i=0
  void_array=np.zeros(patch_size)
  train_X=np.zeros((total, void_array.shape[0], void_array.shape[0], 1))
  train_Y=np.zeros((total, void_array.shape[0], void_array.shape[0], 1))
  num_fields_start=int(num_fields_start)
  #for index in tqdm(np.arange(num_fields_start, num_fields_end), desc="extracting patches"):
  with tqdm(np.arange(num_fields_start, num_fields_end), desc=f"Extracting") as tepoch:
    for index in tepoch:
      # scaled data
      image_X=sst_X[index].astype(np.float32)
      image_Y=sst_Y[index].astype(np.float32)

      # norm data
      """
      image_X=sst_X[index].data
      image_Y=sst_Y[index].data
      mask_Y = sst_Y[index].mask

      if mask_Y.sum() < 1000:
          continue

      image_Y[mask_Y] = 0.0
      """

      sub_images_X=feature_extraction.image.extract_patches_2d(image_X, patch_size, max_patches=max_patches, random_state=0)
      sub_images_Y=feature_extraction.image.extract_patches_2d(image_Y, patch_size, max_patches=max_patches, random_state=0)
      for num in range(sub_images_X.shape[0]):
        sub_image_X=sub_images_X[num]
        sub_image_Y=sub_images_Y[num]
        if np.all(sub_image_X!=void_array) and np.all(sub_image_Y!=void_array):
          train_X[i]=sub_image_X.reshape((patch_size[0], patch_size[1], 1))
          train_Y[i]=sub_image_Y.reshape((patch_size[0], patch_size[1], 1))
          i=i+1
          tepoch.set_postfix(Patches=i)

        if i==total or index==num_fields_end-1:
          del sub_images_X
          del sub_images_Y
          del void_array
          return train_X, train_Y, i, index, 
      del sub_images_X
      del sub_images_Y
    return train_X, train_Y, i, index, 

def run_create_patches(offset):
    #prcp_high_norm, prcp_low_norm = load_prcp_norm()
    prcp_high_norm, prcp_low_norm = load_prcp_scaled(offset=offset)

    patch_size=(33, 33)
    max_patches = 300000
    #num_fields_end = 14610 # Need to define how long our input will be
    #num_fields_end = 730 # Need to define how long our input will be
    num_fields_end = prcp_high_norm.shape[0] # Need to define how long our input will be
    total = min(200000, max_patches*num_fields_end)
    
    X, Y, values, days=create_patches(patch_size, max_patches, prcp_low_norm, prcp_high_norm, num_fields_end, total)
    print(values)

    X = X[:values]
    Y = Y[:values]
    print("shape", X.shape, Y.shape)

    print(X[10])
    print(Y[10])

    h5f = h5py.File(f'low_patches_{offset}.h5', 'w')
    h5f.create_dataset('samples', data=X)
    h5f.close()
    
    h5f = h5py.File(f'high_patches_{offset}.h5', 'w')
    h5f.create_dataset('samples', data=Y)
    h5f.close()

def modify_upscale(offset):
    import cv2

    prcp_high_norm, prcp_low_norm = load_prcp_norm(offset=offset)

    # fill-in-zeros for missing values
    high = prcp_high_norm.data
    high[prcp_high_norm.mask] = 0.0

    low = prcp_low_norm.data
    low[prcp_low_norm.mask] = 0.0

    # upscale using INTER_CUBIC 
    results = []
    target_size = high.shape[1:]
    for index in tqdm(np.arange(low.shape[0]), total=low.shape[0], desc="Resizing"):
        low_per_day = low[index, :, :]
        resized_low = cv2.resize(low_per_day, target_size, interpolation=cv2.INTER_CUBIC)
        results.append(resized_low.reshape([1, *target_size]))

    # merge to get low-all
    resized_low = np.concatenate(results, axis=0)
    print(resized_low.shape)
    print(high.shape)

    # turn to float-16
    high = high.astype(np.float16)
    low = low.astype(np.float16)

    with open(f"prcp_scaled-{offset}.joblib", 'wb') as outfile:
        joblib.dump({
            "high": high,
            "low": low,
            "mask": prcp_high_norm.mask,
        }, outfile)

if __name__ == "__main__":
    """
    get_num_days(1950, 1980)
    quit()
    """

    #offset = 11323 # starting 1981
    #offset = 23741 # 2015-2019
    offset = 21915 # 2010 - 2020

    #build_norm(offset=None)
    modify_upscale(offset)
    run_create_patches(offset)


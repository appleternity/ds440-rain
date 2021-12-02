import numpy.ma as ma
from netCDF4 import Dataset
import numpy as np
import joblib

def build_norm():
    nc_f_low = "/data/440w/bke5075/Data/raw/raw_data/precipitation_daily_1950-2020.nc"
    nc_low = Dataset(nc_f_low, 'r')
    
    lats = nc_low.variables['latitude'][:11323]
    lons = nc_low.variables['longitude'][:11323]
    time = nc_low.variables['time'][:11323]
    prcp_low = nc_low.variables['tp'][:11323]

    max_pixel = np.max(prcp_low)

    def normalise(prcp, norm_const):
        prcp_ = prcp / norm_const
        return prcp_

    prcp_low_norm = normalise(prcp_low, max_pixel)

    print("low-shape:", prcp_low_norm.shape)

    with open("prcp_norm_pred.joblib", 'wb') as outfile:
        joblib.dump({
            "lats": lats, "lons": lons, "time": time,
            "low": prcp_low_norm}, outfile)

def load_prcp_norm():
    with open("prcp_norm_pred.joblib", 'rb') as infile:
        data = joblib.load(infile)
        return data["lats"], data["lons"], data["time"], data["low"]

def predict():
    lats, lons, time, prcp_low_norm = load_prcp_norm()

    patch_size = (33,33)
    input_shape = (patch_size[0], patch_size[1], 1)
    batch_size = 64

    from tensorflow.python.keras import Sequential
    from tensorflow.keras.layers import Conv2D, Masking, add
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.layers import PReLU, Activation
    from tensorflow.keras import backend as K
    from tensorflow.keras import Input
    from tensorflow import image
    from tensorflow import float32, constant_initializer
    import tensorflow
                                                                             
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)

    def PSNRLoss(y_true, y_pred):
        y_true=K.clip(y_true, 0, 1)
        y_pred=K.clip(y_pred, 0, 1)
        im1 = image.convert_image_dtype(y_true, float32)
        im2 = image.convert_image_dtype(y_pred, float32)
        return image.psnr(im2, im1, max_val=1.0)

    input_img = Input(shape=input_shape)

    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)

    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)

    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)

    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)

    model = Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal')(moodel)
    res_img = model
    
    output_img = add([res_img, input_img])

    model = Model(input_img, output_img)
                                                                                
    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(learning_rate=0.001),
                  # metrics=['accuracy'])
                  metrics=[PSNRLoss])

    model.load_weights('/data/440w/bke5075/ds440-rain/ckptsvdsr5/cp-0097.ckpt')
    
    days = prcp_low_norm.shape[0]
    row_max = prcp_low_norm.shape[1]//33
    col_max = prcp_low_norm.shape[2]//33

    prcp_predict = np.zeroes((days, prcp_low_norm.shape[1], prcp_low_norm.shape[2]))

    for day in range(days):
        
        for row in range(0, row_max*33, 33):
            for col in range(0, col_max*33, 33):
                low_patch = prcp_low_norm[day, row:row+33, col:col+33]
                predict = model.predict(low_patch.reshape((1,33,33,1)))
                prcp_predict[day, row:row+33, col:col+33] = predict.reshape((33,33))

        last_row = prcp_low_norm.shape[1]-33
        for col in range(0, col_max*33, 33):
            low_patch = prcp_low_norm[day, last_row:last_row+33, col:col+33]
            predict = model.predict(low_patch.reshape((1,33,33,1)))
            prcp_predict[day, last_row:last_row+33, col:col+33] = predict.reshape((33,33))

        last_col = prcp_low_norm.shape[2]-33
        for row in range(0, row_max*33, 33):
            low_patch = prcp_low_norm[day, row:row+33, last_col:last_col+33]
            predict = model.predict(low_patch.reshape((1,33,33,1)))
            prcp_predict[day, row:row+33, last_col:last_col+33] = predict.reshape((33,33))

    predict_scaled = prcp_predict*max_pixel

    with open("predict_scaled.joblib", 'wb') as outfile:
        joblib.dump({
            "prediction": predict_scaled}, outfile)

if __name__ == "__main__":

    build_norm()
    #predict()

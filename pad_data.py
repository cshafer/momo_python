import numpy as np

def pad_data(data_array):
    data_array_ud = np.flipud(data_array)
    data_array_lr = np.fliplr(data_array)
    data_array_corner = np.flipud(data_array_lr)

    top = np.hstack((data_array_corner, data_array_ud, data_array_corner))
    middle = np.hstack((data_array_lr, data_array, data_array_lr))
    bottom = np.hstack((data_array_corner, data_array_ud, data_array_corner))

    padded_data_array = np.vstack((top, middle, bottom))

    return padded_data_array
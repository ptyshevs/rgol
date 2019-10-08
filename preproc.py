import numpy as np

def rolling_window(a, shape):
    # https://stackoverflow.com/questions/8174467/vectorized-moving-window-on-2d-array-in-numpy
    s = (a.shape[0] - shape[0] + 1, ) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)

def image_to_windows(x, kernel_size=3):
    pad = kernel_size // 2
    x_pad = np.pad(x, pad, mode='constant', constant_values=-1)
    x_train = rolling_window(x_pad, (kernel_size, kernel_size)).reshape((20 * 20, kernel_size * kernel_size))
    return x_train

def df_to_windows(X, kernel_size=3):
    X = X.reshape((-1, 20, 20))
    X_tf = []
    n = len(X)
    for i in range(n):
        x = X[i]
        X_tf.append(image_to_windows(x, kernel_size=kernel_size))
    X_tf = np.vstack(X_tf)
    return X_tf

def prepare_data(X, Y, kernel_size=3):
    X = df_to_windows(X, kernel_size=kernel_size)
    Y = Y.reshape(-1, 1).ravel()
    return X, Y
import numpy as np
from pywt import wavedec, waverec, threshold


def wavelet_transform(x, wavelet="haar", level=2, declevel=2):
    # estimate level 2 wavelet coefficients
    coef = wavedec(x, wavelet, mode='periodization', level=declevel, axis=0)
    # use mad as the robust measure of the variability and caculate the fixed threshold
    mad = np.median(np.absolute(coef[-level] - np.median(coef[-level], axis=0)), axis=0) * 1.4826
    thresh = mad * np.sqrt(2 * np.log(len(x)))

    coef[1:] = (threshold(i, value=thresh, mode="hard") for i in coef[1:])
    # reconstruction
    y = waverec(coef, wavelet, mode='periodization', axis=0)

    return y[-len(x):]


def step_wise_wavelet_trasform(x, wavelet="haar", level=2, declevel=2):
    pass

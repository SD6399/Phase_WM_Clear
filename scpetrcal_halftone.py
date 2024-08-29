from PIL import Image
import numpy as np
from skimage import io


def spectr_to_spatial(qr,size_WM):

    qr_back = np.zeros((size_WM, size_WM)).astype(complex)

    qr = qr.astype(complex)

    for i in range(qr.shape[0]):
        for j in range(qr.shape[1]):
            if qr[i][j] == 255:
                comp_val = np.e ** (complex(0, 1) * np.random.uniform(0, 2 * np.pi))
                qr[i][j] = comp_val

    qr_back[1:66, 1:33] = qr[:, :32]
    for i in range(65):
        for j in range(32):
            qr_back[-(i + 1), -(j + 1)] = np.conj(qr_back[i + 1, j + 1])

    qr_back[1:66, -33:] = qr[:, 32:]
    print(qr_back[1:66, -33:].shape, qr[:, 32:].shape)
    for i in range(66):
        for j in range(1, 34):
            qr_back[-(i + 1), j] = np.conj(qr_back[i + 1, -j])

    my_fft = np.fft.fft2(qr_back)
    print( np.min(my_fft), np.max(my_fft))
    my_fft = 255 * (my_fft - np.min(my_fft)) / (np.max(my_fft) - np.min(my_fft))
    return my_fft,


def check_spatial2spectr(qr_spatial):

    ifft = np.fft.ifft2(qr_spatial)

    ifft[0, 0] = 0 # - КОСТЫЛЬ

    ifft_r = np.abs(ifft)
    my_ifft = 255 * (ifft_r - np.min(ifft_r)) / (np.max(ifft_r) - np.min(ifft_r))
    return my_ifft


# qr_spectr = io.imread("data/spectral_qr_65_65.png")
# spatial_qr = spectr_to_spatial(qr_spectr,512)
# img1 = Image.fromarray(spatial_qr.astype('uint8'))
# img1.save(r"data/spatial_spectr_wm_65.png")

# qr_spatal = io.imread("data/spatial_spectr_wm_65.png")

# check_qr = check_spatial2spectr(qr_spatal)
# img2 = Image.fromarray(check_qr.astype('uint8'))
# img2.save(r"data/check_ifft_wm.png")


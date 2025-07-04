from PIL import Image
import numpy as np
from skimage import io
from scipy.fftpack import fft2, fftshift
from skimage import data, color


def compare_qr(myqr, orig_qr, shift):
    """
     Comparing the extracted QR with the original one
    :param path: path to code for comparison
    :return: percentage of similarity
    """

    # orig_qr = io.imread(r"data/RS_cod89x89.png")
    # orig_cut = np.zeros((49, 49))
    # orig_qr = np.where(orig_qr > 127, 255, 0)
    # orig_cut[:, :24] = orig_qr[1 + shift:50 + shift, 1 + shift:25 + shift]
    # if shift != 0:
    #     orig_cut[:, 24:] = orig_qr[1 + shift:50 + shift, -25 - shift:-shift]
    # else:
    #     orig_cut[:, 24:] = orig_qr[1 + shift:50 + shift, -25 - shift:]
    # small_qr = big2small(orig_qr)
    # sr_matr = np.zeros((1424, 1424, 3))
    # myqr = io.imread(path)

    # myqr_cut = myqr
    myqr_cut = np.zeros((49, 49))

    myqr_cut[:, :24] = myqr[1 + shift:50 + shift, 1 + shift:25 + shift]
    if shift != 0:
        myqr_cut[:, 24:] = myqr[1 + shift:50 + shift, -25 - shift:-shift]
    else:
        myqr_cut[:, 24:] = myqr[1 + shift:50 + shift, -25 - shift:]
    myqr_cut = np.where(myqr_cut > np.mean(myqr_cut), 255, 0)

    sr_matr = orig_qr == myqr_cut

    k = np.count_nonzero(sr_matr)
    return k / sr_matr.size


def spectr_to_spatial(qr, size_WM):
    size_qr = 49
    half_qr_size = int((size_qr + 1) / 2)  # 25
    qr_back = np.zeros((size_WM, size_WM)).astype(complex)
    shift = 40
    qr = qr.astype(complex)

    for i in range(qr.shape[0]):
        for j in range(qr.shape[1]):
            if qr[i][j] == 255:
                comp_val = np.e ** (complex(0, 1) * np.random.uniform(0, 2 * np.pi))
                qr[i][j] = comp_val

    print(size_qr, half_qr_size)
    qr_back[shift + 1:shift + size_qr + 1, shift + 1:shift + half_qr_size] = qr[:, :half_qr_size - 1]
    for i in range(size_qr + 1):
        for j in range(half_qr_size - 1):
            # print(qr_back[i + 1, j + 1], (shift+i + 1, shift+ j + 1), (-shift- (i + 1),- shift- (j + 1)),
            #       (256-(shift + i + 1),256-(shift + j + 1)))
            qr_back[-shift - (i + 1), -shift - (j + 1)] = np.conj(qr_back[shift + i + 1, shift + j + 1])

    if shift != 0:
        qr_back[shift + 1:shift + size_qr + 1, -shift - half_qr_size:-shift] = qr[:, half_qr_size - 1:]
    else:
        qr_back[shift + 1:shift + size_qr + 1, shift - half_qr_size:] = qr[:, half_qr_size - 1:]
    # print(qr_back[1:66, -33:].shape, qr[:, 32:].shape)
    for i in range(size_qr + 1):
        for j in range(1, half_qr_size + 1):
            if shift != 0:
                qr_back[-shift - (i + 1), shift + j] = np.conj(qr_back[shift + i + 1, -shift - j])
            else:
                qr_back[-shift - (i + 1), shift + j] = np.conj(qr_back[shift + i + 1, -j])

    my_fft = np.fft.ifft2(qr_back)
    print(np.min(my_fft), np.max(my_fft))
    my_fft = 255 * (my_fft - np.min(my_fft)) / (np.max(my_fft) - np.min(my_fft))
    print(my_fft.imag)

    img1 = Image.fromarray(my_fft.real.astype('uint8'))
    img1.save(
        r"data/attempt_new_spatial_spectr_" + str(size_WM) + "_in_shift_" + str(shift) + "_wm_" + str(size_qr) + ".png")

    return my_fft.real


def check_spatial2spectr(qr_spatial):
    ifft = np.fft.fft2(qr_spatial)

    ifft[0, 0] = 0  # - КОСТЫЛЬ

    ifft_r = np.abs(ifft)
    my_ifft = 255 * (ifft_r - np.min(ifft_r)) / (np.max(ifft_r) - np.min(ifft_r))
    return my_ifft


def energy_spector(image):
    F = fft2(image)

    # Спектральная плотность мощности
    Pxx = np.abs(F) ** 2

    ifft = np.fft.ifft2(Pxx)

    ifft = np.abs(ifft) / F.size
    norm_eval = 255 * (ifft - np.min(ifft)) / (np.max(ifft) - np.min(ifft))
    return norm_eval


# qr_spectr = io.imread("data/test_qr_49_49.png")
# spatial_qr = spectr_to_spatial(qr_spectr, 1024)
#
# qr_spatal = io.imread("data/attempt_new_spatial_spectr_1024_in_shift_40_wm_49.png")
#
# check_qr = check_spatial2spectr(qr_spatal)
# img2 = Image.fromarray(check_qr.astype('uint8'))
# img2.save(r"data/attempt_new_check_ifft_wm_1024_shift_40_49.png")
# print(compare_qr(check_qr, qr_spectr, 40))
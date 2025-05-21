import math

import matplotlib.pyplot as plt
from skimage import io
# from reedsolo import RSCodec
from skimage.exposure import histogram
import cv2
import os
import numpy as np
from PIL import Image, ImageFile
# from qrcode_1 import read_qr, correct_qr
from helper_methods import small2big, big2small, sort_spis, read_video
from helper_methods import csv2list, bit_voting, compare_qr, binarize_qr
from scpetrcal_halftone import check_spatial2spectr

ImageFile.LOAD_TRUNCATED_IMAGES = True


def embed(folder_orig_image, folder_to_save, binary_image, amplitude, tt):
    """
    Procedure embedding
    :param binary_image: embedding code
    :param folder_orig_image: the folder from which the original images are taken
    :param folder_to_save: the folder where the images from the watermark are saved
    :param amplitude: embedding amplitude
    :param tt: reference frequency parameter
    """

    fi = math.pi / 2 / 255
    st_qr = cv2.imread(binary_image)
    st_qr = cv2.cvtColor(st_qr, cv2.COLOR_RGB2YCrCb)[:, :, 0]

    # data_length = st_qr[:, :, 0].size
    # shuf_order = np.arange(data_length)
    #
    # np.random.seed(42)
    # np.random.shuffle(shuf_order)
    #
    # # Expand the binary image into a string
    # st_qr_1d = st_qr[:, :, 0].ravel()
    # shuffled_data = st_qr_1d[shuf_order]  # Shuffle the original data
    #
    # # 1d-string in the image
    # pict = np.resize(shuffled_data, (1057, 1920))
    # # the last elements are uninformative. Therefore, we make zeros
    # pict[-1, 256 - 1920:] = 0

    images = [img for img in os.listdir(folder_orig_image)
              if img.endswith(".png")]

    # The list should be sorted by numbers after the name
    sort_name_img = sort_spis(images, "sum_mosaic")[:total_count]
    cnt = 0

    diff_neighb = []

    while cnt < len(sort_name_img):
        # Reads in BGR format
        imgg = cv2.imread(folder_orig_image + sort_name_img[cnt]).astype('float32')
        # translation to the YCrCb space
        a = cv2.cvtColor(imgg, cv2.COLOR_BGR2YCrCb)

        def_img = np.copy(a)
        # a = a.astype(float)

        temp = fi * st_qr
        # A*sin(m * teta + fi)
        wm = np.array((amplitude * np.sin(cnt * tt + temp)))

        # Embedding in the Y-channel
        for row_ind in range(0, a.shape[0], st_qr.shape[0]):
            for col_ind in range(0, a.shape[1], st_qr.shape[0]):
                if ((a.shape[0] - row_ind) >= st_qr.shape[0]) and (
                        (a.shape[1] - col_ind) >= st_qr.shape[0]):
                    def_img[row_ind:row_ind + st_qr.shape[0], col_ind:col_ind + st_qr.shape[0], 0] = np.where(
                        np.float32(a[row_ind:row_ind + st_qr.shape[0], col_ind:col_ind + st_qr.shape[0],
                                   0] + wm) > 255,
                        255,
                        np.where(a[row_ind:row_ind + st_qr.shape[0], col_ind:col_ind + st_qr.shape[0],
                                 0] + wm < 0, 0,
                                 np.float32(
                                     a[row_ind:row_ind + st_qr.shape[0], col_ind:col_ind + st_qr.shape[0],
                                     0] + wm)))
                elif ((a.shape[0] - row_ind) < st_qr.shape[0]) and (
                        (a.shape[1] - col_ind) >= st_qr.shape[0]):
                    def_img[row_ind:a.shape[0], col_ind:col_ind + st_qr.shape[0], 0] = np.where(
                        np.float32(
                            a[row_ind:a.shape[0], col_ind:col_ind + st_qr.shape[0], 0] + wm[
                                                                                         :a.shape[0] - row_ind,
                                                                                         :]) > 255,
                        255,
                        np.where(a[row_ind:a.shape[0], col_ind:col_ind + st_qr.shape[0], 0] + wm[:a.shape[
                                                                                                      0] - row_ind,
                                                                                              :] < 0, 0,
                                 np.float32(
                                     a[row_ind:a.shape[0], col_ind:col_ind + st_qr.shape[0], 0] + wm[:a.shape[
                                                                                                          0] - row_ind,
                                                                                                  :])))
                elif ((a.shape[0] - row_ind) >= st_qr.shape[0]) and (
                        (a.shape[1] - col_ind) < st_qr.shape[0]):
                    # print(wm[:, a.shape[1] - col_ind].shape)
                    def_img[row_ind:row_ind + st_qr.shape[0], col_ind:a.shape[1], 0] = np.where(
                        np.float32(
                            a[row_ind:row_ind + st_qr.shape[0], col_ind:a.shape[1], 0] + wm[:, :a.shape[
                                                                                                    1] - col_ind]) > 255,
                        255,
                        np.where(a[row_ind:row_ind + st_qr.shape[0], col_ind:a.shape[1], 0] + wm[:, :a.shape[
                                                                                                         1] - col_ind] < 0,
                                 0,
                                 np.float32(
                                     a[row_ind:row_ind + st_qr.shape[0], col_ind:a.shape[1], 0] + wm[:, :a.shape[
                                                                                                             1] - col_ind])))
                else:
                    def_img[row_ind:a.shape[0], col_ind:a.shape[1], 0] = np.where(
                        np.float32(a[row_ind:a.shape[0], col_ind:a.shape[1], 0] + wm[
                            a.shape[0] - row_ind, a.shape[1] - col_ind]) > 255,
                        255,
                        np.where(a[row_ind:a.shape[0], col_ind:a.shape[1], 0] + wm[
                                                                                :a.shape[0] - row_ind,
                                                                                :a.shape[1] - col_ind] < 0, 0,
                                 np.float32(
                                     a[row_ind:a.shape[0], col_ind:a.shape[1], 0] + wm[:
                                                                                       a.shape[0] - row_ind,
                                                                                    :a.shape[1] - col_ind])))
        # a[20:1060, 440:1480, 0] = np.where(np.float32(a[20:1060, 440:1480, 0] + wm[:, :, 0]) > 255, 255,
        #                                    np.where(a[20:1060, 440:1480, 0] + wm[:, :, 0] < 0, 0,
        #                                             np.float32(a[20:1060, 440:1480, 0] + wm[:, :, 0])))
        # diff_neighb.append((def_img - a)[100, 100, 0])
        tmp = cv2.cvtColor(def_img, cv2.COLOR_YCrCb2BGR)

        # Converting the YCrCb matrix to BGR
        img_path = os.path.join(folder_to_save)
        cv2.imwrite(img_path + "frame" + str(cnt) + ".png", tmp)

        if cnt % 70 == 0:
            print("wm embed", cnt)

        cnt += 1
    print(diff_neighb)


def read2list(file):
    """

    :param file: file which transform to list
    :return: list of values
    """
    # opening the file in utf-8 reading mode
    file = open(file, 'r', encoding='utf-8')
    # we read all the lines and delete the newline characters
    lines = file.readlines()
    lines = [line.rstrip('\n') for line in lines]
    file.close()

    return lines


def extract(alf, beta, tt, size_wm, rand_fr, shift_qr):
    """
    Procedure embedding
    :param shift_qr: shift for spectral WM from (0,0)
    :param alf: primary smoothing parameter
    :param beta: primary smoothing parameter0
    :param tt:reference frequency
    :param size_wm: side of embedding watermark
    :param rand_fr: the frame from which the extraction begins
    :return: the path to the final image
    """
    PATH_VIDEO = r'D:/pythonProject/phase_wm\frames_after_emb\RB_codec.mp4'

    count = read_video(PATH_VIDEO, 'D:/pythonProject/phase_wm/extract/', total_count)
    psnr_full = 0
    for i in range(50):
        image1 = cv2.imread("D:\pythonProject\phase_wm/frames_after_emb/frame" + str(i) + ".png")
        image2 = cv2.imread("D:\pythonProject\phase_wm/extract/frame" + str(i) + ".png")

        psnr_full += (cv2.PSNR(image1, image2))

    print("A = ", ampl, "PSNR: ", psnr_full / 50)
    cnt = int(rand_fr)
    g = np.asarray([])
    f = g.copy()
    f1 = f.copy()
    orig100 = []
    smooth100 = []

    while cnt < total_count:
        arr = io.imread(r"D:/pythonProject/phase_wm\extract/frame" + str(cnt) + ".png").astype('float32')
        orig100.append(cv2.cvtColor(arr, cv2.COLOR_BGR2YCrCb)[100, 100, 0])
        d1 = f1
        if cnt == rand_fr:
            f1 = arr.astype('float32')
            d1 = np.zeros((1080, 1920))
        # elif cnt == change_sc[scene-1] + 1:
        else:
            f1 = np.float32(d1) * alf + np.float32(arr) * (1 - alf)
        # else:
        #     f1 = (1-alf)*(1-alf)*a+(1-alf)*alf*d1+alf*g1

        np.clip(f1, 0, 255, out=f1)
        smooth100.append(cv2.cvtColor(f1, cv2.COLOR_BGR2YCrCb)[100, 100, 0])
        img = Image.fromarray(f1.astype('uint8'))
        if cnt % 100 == 0:
            print("first smooth", cnt)
        img.save(r'D:/pythonProject/phase_wm\extract\first_smooth/result' + str(cnt) + '.png')

        cnt += 1

    # print("orig", orig100)
    # print("smooth", smooth100)
    # plt.plot(orig100,label = "orig")
    # plt.plot(smooth100,label = "smooth")
    # plt.legend()
    plt.show()
    variance = []
    cnt = int(rand_fr)
    g = np.asarray([])
    f = g.copy()
    d = g.copy()

    g2 = np.zeros((1024, 1920), dtype=np.complex_)
    f2 = np.zeros((1024, 1920), dtype=np.complex_)
    d2 = np.zeros((1024, 1920), dtype=np.complex_)

    count = total_count

    # reading a shuffled object

    # subtracting the average
    # while cnt < count:
    #
    #     arr = np.float32(cv2.imread(r"D:/pythonProject/phase_wm/extract/first_smooth/result" + str(cnt) + ".png"))
    #     # arr = np.float32(cv2.imread(r"D:/pythonProject/phase_wm/extract/frame" + str(cnt) + ".png"))
    #     a = cv2.cvtColor(arr, cv2.COLOR_BGR2YCrCb)
    #
    #     f1 = np.float32(
    #         cv2.imread(r"D:/pythonProject/phase_wm\extract\frame" + str(cnt) + ".png"))
    #     f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2YCrCb)
    #     a1 = np.where(a < f1, f1 - a, a - f1)
    #
    #     a1 = a1[0:1024, 0:1920, 0]
    #     # a1 = a[0:512, 0:512, 0]
    #
    #     # res_1d = np.ravel(a1)[:256 - 1920]
    #     # start_qr = np.resize(res_1d, (size_wm, size_wm))
    #     #
    #     # unshuf_order = np.zeros_like(shuf_order)
    #     # unshuf_order[shuf_order] = np.arange(start_qr.size)
    #     # unshuffled_data = np.ravel(start_qr)[unshuf_order]
    #     # matr_unshuf = np.resize(unshuffled_data, (size_wm, size_wm))
    #     a = a1
    #     # extraction of watermark
    #     # a = a1[20:1060, 440:1480, 0]
    #     g = np.copy(d)
    #     d = np.copy(f)
    #
    #     if cnt == rand_fr:
    #         f = np.copy(a1)
    #         d = np.ones((1024, 1920))
    #
    #     else:
    #         if cnt == rand_fr + 1:
    #             f = 2 * beta * np.cos(tt) * np.float32(d) + np.float32(a)
    #
    #         else:
    #             f = 2 * beta * np.cos(tt) * np.float32(d) - (beta ** 2) * np.float32(g) + np.float32(a)
    #
    #     # if cnt == rand_fr:
    #     #     f2 = np.copy(a1)
    #     #     d2 = np.ones((size_wm, size_wm))
    #     #
    #     # else:
    #     #     if cnt == rand_fr + 1:
    #     #         f2 = 2 * beta * np.cos(tt) * np.float32(d2) + np.float32(f)
    #     #
    #     #     else:
    #     #         f2 = 2 * beta * np.cos(tt) * np.float32(d2) - (beta ** 2) * np.float32(g2) + np.float32(f)
    #
    #     yc = np.float32(f) - beta * np.cos(tt) * np.float32(d)
    #     ys = beta * np.sin(tt) * np.float32(d)
    #
    #     g2 = np.copy(d2)
    #     d2 = np.copy(g2)
    #
    #     tmp_signal = np.zeros((1024, 1920), dtype=np.complex_)
    #     tmp_signal.real = yc
    #     tmp_signal.imag = ys
    #
    #     if cnt == rand_fr:
    #         f2 = tmp_signal
    #         d2 = np.ones((1024, 1920), dtype=np.complex_)
    #         d2.imag = np.ones((1024, 1920))
    #
    #     else:
    #         if cnt == rand_fr + 1:
    #             f2.real = 2 * beta * np.cos(tt) * np.float32(d2.real) + np.float32(tmp_signal.real)
    #             f2.imag = 2 * beta * np.cos(tt) * np.float32(d2.imag) + np.float32(tmp_signal.imag)
    #         else:
    #             f2.real = 2 * beta * np.cos(tt) * np.float32(d2.real) - (beta ** 2) * np.float32(g2.real) + np.float32(
    #                 tmp_signal.real)
    #             f2.imag = 2 * beta * np.cos(tt) * np.float32(d2.imag) - (beta ** 2) * np.float32(g2.imag) + np.float32(
    #                 tmp_signal.imag)
    #
    #     c = np.cos(tt * cnt) * np.float32(f2.real) + np.sin(tt * cnt) * np.float32(f2.imag)
    #     s = np.cos(tt * cnt) * np.float32(f2.imag) - np.sin(tt * cnt) * np.float32(f2.real)
    #
    #     # c = np.cos(tt * cnt) * np.float32(yc) + np.sin(tt * cnt) * np.float32(ys)
    #     # s = np.cos(tt * cnt) * np.float32(ys) - np.sin(tt * cnt) * np.float32(yc)
    #
    #     try:
    #         fi = np.where(c < 0, np.arctan((s / c)) + np.pi,
    #                       np.where(s >= 0, np.arctan((s / c)), np.arctan((s / c)) + 2 * np.pi))
    #     except ZeroDivisionError:
    #         fi = np.full(f.shape, 255)
    #     fi = np.nan_to_num(fi)
    #     fi = np.where(fi < -np.pi / 4, fi + 2 * np.pi, fi)
    #     fi = np.where(fi > 9 * np.pi / 4, fi - 2 * np.pi, fi)
    #
    #     wm = 255 * fi / 2 / math.pi
    #
    #     wm[wm > 255] = 255
    #     wm[wm < 0] = 0
    #
    #     a1 = wm
    #     # # a1 = cv2.cvtColor(a1, cv2.COLOR_YCrCb2RGB)
    #     # img = Image.fromarray(big2small(a1).astype('uint8'))
    #     # img.save(r'D:/pythonProject/phase_wm\extract/wm/result' + str(cnt) + '.png')
    #     # bringing to the operating range
    #
    #     # l_kadr = io.imread(r'D:/pythonProject/phase_wm\extract/wm/result' + str(cnt) + '.png')
    #     # compr= l_kadr==a1
    #     # fi = np.copy(l_kadr)
    #     fi = (a1 * np.pi * 2) / 255
    #
    #     coord1 = np.where(fi < np.pi, (fi / np.pi * 2 - 1) * (-1), ((fi - np.pi) / np.pi * 2 - 1))
    #     coord2 = np.where(fi < np.pi / 2, (fi / np.pi / 2),
    #                       np.where(fi > 3 * np.pi / 2, ((fi - 1.5 * np.pi) / np.pi * 2) - 1,
    #                                ((fi - 0.5 * np.pi) * 2 / np.pi - 1) * (-1)))
    #
    #     # noinspection PyTypeChecker
    #     hist, bin_centers = histogram(coord1, normalize=False)
    #     # noinspection PyTypeChecker
    #     hist2, bin_centers2 = histogram(coord2, normalize=False)
    #
    #     mx_sp = np.arange(bin_centers2[0], bin_centers2[-1], bin_centers2[1] - bin_centers2[0])
    #     ver = hist2 / np.sum(hist)
    #     mo = np.sum(bin_centers2 * ver)
    #     dis = np.abs(mo - mx_sp)
    #     pr1 = np.min(dis)
    #
    #     mx_sp2 = np.arange(bin_centers2[0], bin_centers2[-1], bin_centers2[1] - bin_centers2[0])
    #     ver2 = hist2 / np.sum(hist2)
    #     mo = np.sum(bin_centers2 * ver2)
    #     dis2 = np.abs(mo - mx_sp2)
    #     x = np.min(dis2)
    #
    #     idx = np.argmin(np.abs(dis2 - x))
    #     pr2 = bin_centers2[idx]
    #
    #     moment = np.where(pr1 < 0, np.arctan((pr2 / pr1)) + np.pi,
    #                       np.where(pr2 >= 0, np.arctan((pr2 / pr1)), np.arctan((pr2 / pr1)) + 2 * np.pi))
    #
    #     if np.pi / 4 <= moment <= np.pi * 2 - np.pi / 4:
    #         fi_tmp = fi - moment + 0.5 * np.pi * 0.5
    #
    #     elif moment > np.pi * 2 - np.pi / 4:
    #         fi = np.where(fi < np.pi / 4, fi + 2 * np.pi, fi)
    #         fi_tmp = fi - moment + 0.5 * np.pi * 0.5
    #
    #     else:
    #         fi_tmp = fi - 2 * np.pi - moment + 0.5 * np.pi * 0.5
    #
    #     fi_tmp = np.where(fi_tmp < -np.pi / 4, fi_tmp + 2 * np.pi, fi_tmp)
    #     fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)
    #     fi_tmp[fi_tmp < 0] = 0
    #     fi_tmp[fi_tmp > np.pi] = np.pi
    #     l_kadr = fi_tmp * 255 / np.pi
    #
    #     l_kadr = 255 * (l_kadr - np.min(l_kadr)) / (np.max(l_kadr) - np.min(l_kadr))
    #
    #     img = Image.fromarray(l_kadr.astype('uint8'))
    #     img.save(r"D:/pythonProject/phase_wm\extract/after_normal_phas/result" + str(cnt) + ".png")
    #
    #     test_var = np.zeros((1024, 1024))
    #     for row_ind in range(0, l_kadr.shape[0] - 1024, 1024):
    #         for col_ind in range(0, l_kadr.shape[1] - 1024, 1024):
    #             test_var += l_kadr[row_ind:row_ind + 1024, col_ind:col_ind + 512] / 6
    #
    #     variance.append(np.var(test_var - img_wm))
    #     if cnt % 20 == 19:
    #         # ser6 = []
    #         spector = np.zeros((1024, 1024))
    #         for row_ind in range(0, l_kadr.shape[0] - size_wm + 1, 256):
    #             for col_ind in range(0, l_kadr.shape[1] - size_wm + 1, 256):
    #                 # print(row_ind, col_ind, l_kadr.shape)
    #                 spector += check_spatial2spectr(l_kadr[row_ind:row_ind + 1024, col_ind:col_ind + 1024]) / 4
    #
    #         stop_kadr1.append(
    #             compare_qr(spector, io.imread(r"D:\pythonProject/Phase_WM_Clear/data/check_ifft_wm.png"),
    #                        shift_qr), )
    #
    #         print(ampl, cnt, stop_kadr1)
    #
    #         if cnt % 200 == 199:
    #             img = Image.fromarray(spector.astype('uint8'))
    #             img.save(r"D:/pythonProject/phase_wm\extract/after_normal_phas_bin/result" + str(cnt) + ".png")
    #             print(ampl, cnt, stop_kadr1)
    #         # stop_kadr1.append(ser6)
    #
    #     cnt += 1
    diff100 = []
    while cnt < count:

        arr = np.float32(cv2.imread(r"D:/pythonProject/phase_wm/extract/first_smooth/result" + str(cnt) + ".png"))
        # arr = np.float32(cv2.imread(r"D:/pythonProject/phase_wm/extract/frame" + str(cnt) + ".png"))
        a = cv2.cvtColor(arr, cv2.COLOR_BGR2YCrCb)

        f1 = np.float32(
            cv2.imread(r"D:/pythonProject/phase_wm\extract\frame" + str(cnt) + ".png"))
        f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2YCrCb)
        # a1 = np.where(a < f1, f1 - a, a - f1)
        a1 = a - f1

        a1 = a1[0:1024, 0:1920, 0]

        diff100.append(a1[100, 100])
        # a1 = a[0:512, 0:512, 0]

        # res_1d = np.ravel(a1)[:256 - 1920]
        # start_qr = np.resize(res_1d, (size_wm, size_wm))
        #
        # unshuf_order = np.zeros_like(shuf_order)
        # unshuf_order[shuf_order] = np.arange(start_qr.size)
        # unshuffled_data = np.ravel(start_qr)[unshuf_order]
        # matr_unshuf = np.resize(unshuffled_data, (size_wm, size_wm))
        a = a1
        # extraction of watermark
        # a = a1[20:1060, 440:1480, 0]
        g = np.copy(d)
        d = np.copy(f)

        if cnt == rand_fr:
            f = np.copy(a1)
            d = np.ones((1024, 1920))

        else:
            if cnt == rand_fr + 1:
                f = 2 * beta * np.cos(tt) * np.float32(d) + np.float32(a)

            else:
                f = 2 * beta * np.cos(tt) * np.float32(d) - (beta ** 2) * np.float32(g) + np.float32(a)

        # if cnt == rand_fr:
        #     f2 = np.copy(a1)
        #     d2 = np.ones((size_wm, size_wm))
        #
        # else:
        #     if cnt == rand_fr + 1:
        #         f2 = 2 * beta * np.cos(tt) * np.float32(d2) + np.float32(f)
        #
        #     else:
        #         f2 = 2 * beta * np.cos(tt) * np.float32(d2) - (beta ** 2) * np.float32(g2) + np.float32(f)

        yc = np.float32(f) - beta * np.cos(tt) * np.float32(d)
        ys = beta * np.sin(tt) * np.float32(d)

        g2 = np.copy(d2)
        d2 = np.copy(g2)

        tmp_signal = np.zeros((1024, 1920), dtype=np.complex_)
        tmp_signal.real = yc
        tmp_signal.imag = ys

        if cnt == rand_fr:
            f2 = tmp_signal
            d2 = np.ones((1024, 1920), dtype=np.complex_)
            d2.imag = np.ones((1024, 1920))

        else:
            if cnt == rand_fr + 1:
                f2.real = 2 * beta * np.cos(tt) * np.float32(d2.real) + np.float32(tmp_signal.real)
                f2.imag = 2 * beta * np.cos(tt) * np.float32(d2.imag) + np.float32(tmp_signal.imag)
            else:
                f2.real = 2 * beta * np.cos(tt) * np.float32(d2.real) - (beta ** 2) * np.float32(g2.real) + np.float32(
                    tmp_signal.real)
                f2.imag = 2 * beta * np.cos(tt) * np.float32(d2.imag) - (beta ** 2) * np.float32(g2.imag) + np.float32(
                    tmp_signal.imag)

        # g3 = np.copy(d3)
        # d3 = np.copy(g3)
        #
        # if cnt == rand_fr:
        #     f3 = tmp_signal
        #     d3 = np.ones((1024, 1920), dtype=np.complex_)
        #     d3.imag = np.ones((1024, 1920))
        #
        # else:
        #     if cnt == rand_fr + 1:
        #         f3.real = 2 * beta * np.cos(tt) * np.float32(d3.real) + np.float32(f2.real)
        #         f3.imag = 2 * beta * np.cos(tt) * np.float32(d3.imag) + np.float32(f2.imag)
        #     else:
        #         f3.real = 2 * beta * np.cos(tt) * np.float32(d3.real) - (beta ** 2) * np.float32(g3.real) + np.float32(
        #             f2.real)
        #         f3.imag = 2 * beta * np.cos(tt) * np.float32(d3.imag) - (beta ** 2) * np.float32(g3.imag) + np.float32(
        #             f2.imag)
        #
        # c = np.cos(tt * cnt) * np.float32(f3.real) + np.sin(tt * cnt) * np.float32(f3.imag)
        # s = np.cos(tt * cnt) * np.float32(f3.imag) - np.sin(tt * cnt) * np.float32(f3.real)

        c = np.cos(tt * cnt) * np.float32(f2.real) + np.sin(tt * cnt) * np.float32(f2.imag)
        s = np.cos(tt * cnt) * np.float32(f2.imag) - np.sin(tt * cnt) * np.float32(f2.real)

        # c = np.cos(tt * cnt) * np.float32(yc) + np.sin(tt * cnt) * np.float32(ys)
        # s = np.cos(tt * cnt) * np.float32(ys) - np.sin(tt * cnt) * np.float32(yc)

        try:
            fi = np.where(c < 0, np.arctan((s / c)) + np.pi,
                          np.where(s >= 0, np.arctan((s / c)), np.arctan((s / c)) + 2 * np.pi))
        except ZeroDivisionError:
            fi = np.full(f.shape, 255)
        fi = np.nan_to_num(fi)
        fi = np.where(fi < -np.pi / 4, fi + 2 * np.pi, fi)
        fi = np.where(fi > 9 * np.pi / 4, fi - 2 * np.pi, fi)

        wm = 255 * fi / 2 / math.pi

        wm[wm > 255] = 255
        wm[wm < 0] = 0

        a1 = wm
        # # a1 = cv2.cvtColor(a1, cv2.COLOR_YCrCb2RGB)
        # img = Image.fromarray(big2small(a1).astype('uint8'))
        # img.save(r'D:/pythonProject/phase_wm\extract/wm/result' + str(cnt) + '.png')
        # bringing to the operating range

        # l_kadr = io.imread(r'D:/pythonProject/phase_wm\extract/wm/result' + str(cnt) + '.png')
        # compr= l_kadr==a1
        # fi = np.copy(l_kadr)
        fi = (a1 * np.pi * 2) / 255

        coord1 = np.where(fi < np.pi, (fi / np.pi * 2 - 1) * (-1), ((fi - np.pi) / np.pi * 2 - 1))
        coord2 = np.where(fi < np.pi / 2, (fi / np.pi / 2),
                          np.where(fi > 3 * np.pi / 2, ((fi - 1.5 * np.pi) / np.pi * 2) - 1,
                                   ((fi - 0.5 * np.pi) * 2 / np.pi - 1) * (-1)))

        # noinspection PyTypeChecker
        hist, bin_centers = histogram(coord1, normalize=False)
        # noinspection PyTypeChecker
        hist2, bin_centers2 = histogram(coord2, normalize=False)

        mx_sp = np.arange(bin_centers2[0], bin_centers2[-1], bin_centers2[1] - bin_centers2[0])
        ver = hist2 / np.sum(hist)
        mo = np.sum(bin_centers2 * ver)
        dis = np.abs(mo - mx_sp)
        pr1 = np.min(dis)

        mx_sp2 = np.arange(bin_centers2[0], bin_centers2[-1], bin_centers2[1] - bin_centers2[0])
        ver2 = hist2 / np.sum(hist2)
        mo = np.sum(bin_centers2 * ver2)
        dis2 = np.abs(mo - mx_sp2)
        x = np.min(dis2)

        idx = np.argmin(np.abs(dis2 - x))
        pr2 = bin_centers2[idx]

        moment = np.where(pr1 < 0, np.arctan((pr2 / pr1)) + np.pi,
                          np.where(pr2 >= 0, np.arctan((pr2 / pr1)), np.arctan((pr2 / pr1)) + 2 * np.pi))

        if np.pi / 4 <= moment <= np.pi * 2 - np.pi / 4:
            fi_tmp = fi - moment + 0.5 * np.pi * 0.5

        elif moment > np.pi * 2 - np.pi / 4:
            fi = np.where(fi < np.pi / 4, fi + 2 * np.pi, fi)
            fi_tmp = fi - moment + 0.5 * np.pi * 0.5

        else:
            fi_tmp = fi - 2 * np.pi - moment + 0.5 * np.pi * 0.5

        fi_tmp = np.where(fi_tmp < -np.pi / 4, fi_tmp + 2 * np.pi, fi_tmp)
        fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)
        fi_tmp[fi_tmp < 0] = 0
        fi_tmp[fi_tmp > np.pi] = np.pi
        l_kadr = fi_tmp * 255 / np.pi

        l_kadr = 255 * (l_kadr - np.min(l_kadr)) / (np.max(l_kadr) - np.min(l_kadr))

        img = Image.fromarray(l_kadr.astype('uint8'))
        img.save(r"D:/pythonProject/phase_wm\extract/after_normal_phas/result" + str(cnt) + ".png")

        test_var = np.zeros((size_wm, size_wm))
        for row_ind in range(0, l_kadr.shape[0] - size_wm, size_wm):
            for col_ind in range(0, l_kadr.shape[1] - size_wm, size_wm):
                test_var += l_kadr[row_ind:row_ind + size_wm, col_ind:col_ind + size_wm] / 6

        variance.append(np.var(test_var - img_wm))
        bin_qr_spector = np.zeros((49, 49))
        count_quadr = 0
        if cnt % 1 == 0:
            # ser6 = []
            spector = np.zeros((size_wm, size_wm))
            for row_ind in range(0, l_kadr.shape[0] - size_wm + 1, 16):
                for col_ind in range(0, l_kadr.shape[1] - size_wm + 1, 16):
                    # print(row_ind, col_ind, l_kadr.shape)
                    spector += check_spatial2spectr(l_kadr[row_ind:row_ind + size_wm, col_ind:col_ind + size_wm]) / 8

                    count_quadr += 1
            # tmp_bin_spector = binarize_qr(spector, shift_qr)
            # ser6.append(compare_qr(tmp_bin_spector,
            #                        io.imread(
            #                            r"D:\pythonProject/Phase_WM_Clear/data/check_ifft_wm_1024_shift_0_49"
            #                            r".png"),
            #                        shift_qr))
            # bin_qr_spector += tmp_bin_spector

            # bin_qr_spector = np.where(bin_qr_spector > np.mean(bin_qr_spector), 255, 0)
            stop_kadr1.append(round(
                compare_qr(spector,
                           io.imread(
                               r"D:\pythonProject/Phase_WM_Clear/data/attempt_new_check_ifft_wm_1024_shift_40_49.png"),
                           shift_qr, cnt), 5))

            if cnt % 10 == 9:
                v = vot_by_variance(r"D:/pythonProject/phase_wm\extract\after_normal_phas_bin", max(0, cnt - 2000), cnt,
                                    0.045)
                vot_sp.append(round(max(v, 1 - v), 5))
                if cnt % 100 == 99:
                    print(ampl, alf, cnt, stop_kadr1)
                    print("after voting", cnt, tt, vot_sp)

            # if cnt % 200 == 199:
            #     # print(max(ser6), min(ser6), np.mean(ser6))
            #     # print(ser6)
            #     img = Image.fromarray(spector.astype('uint8'))
            #     img.save(r"D:/pythonProject/phase_wm\extract/after_normal_phas_bin/result" + str(cnt) + ".png")
            #     print(ampl, alf, cnt, stop_kadr1)
            # stop_kadr1.append(ser6)

        cnt += 1
    # print("Difference 100", diff100)

    return variance, stop_kadr1


def generate_video(bitr, image_folder):
    """
    Sequence of frames transform to compress video
    :param image_folder: folder for output frames

    :param bitr: bitrate of output video
    """

    if bitr != "orig":
        video_name = 'need_video.mp4'
    else:
        video_name = "RB_codec.mp4"
    os.chdir(image_folder)

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".png")]
    sort_name_img = sort_spis(images, "frame")[:total_count]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    # fourcc = cv2.VideoWriter_fourcc(*'H264')

    video = cv2.VideoWriter(video_name, 0, 29.97, (width, height))

    cnt = 0
    for image in sort_name_img:
        # if cnt % 300 == 0:

        video.write(cv2.imread(os.path.join(image_folder, image)))
        if cnt % 799 == 0:
            print(cnt)
        cnt += 1
    cv2.destroyAllWindows()
    video.release()

    if bitr != "orig":
        print("Codec worked")
        os.system(f"ffmpeg -y -i D:/pythonProject/phase_wm/frames_after_emb/need_video.mp4 -b:v {bitr}M -vcodec"
                  f" libx264  D:/pythonProject/phase_wm/frames_after_emb/RB_codec.mp4")


def vot_by_variance(path_imgs, start, end, treshold):
    var_list = csv2list(r"D:/pythonProject/\phase_wm/RB_disp.csv")[start:end]
    sum_matrix = np.zeros((49, 49))
    np_list = np.array(var_list)
    need_ind = [i for i in range(len(np_list)) if np_list[i] > treshold]
    i = start
    count = 0
    while i < end:
        c_qr = io.imread(path_imgs + r"/result" + str(i) + ".png")
        c_qr[c_qr == 255] = 1
        if (i - start) not in need_ind:
            sum_matrix += c_qr
            count += 1
        else:
            i += 1
        i += 1

    sum_matrix[sum_matrix <= count * 0.5] = 0
    sum_matrix[sum_matrix > count * 0.5] = 255
    print(np.count_nonzero(sum_matrix))
    # img1 = Image.fromarray(sum_matrix.astype('uint8'))
    # img1.save(r"D:/pythonProject/phase_wm\voting" + ".png")
    orig_qr = io.imread(r"D:\pythonProject\Phase_WM_Clear/data/test_qr_49_49.png")
    orig_qr = np.where(orig_qr > 127, 255, 0)

    sr_matr = orig_qr == sum_matrix
    k = np.count_nonzero(sr_matr)
    comp = k / sr_matr.size

    return comp


if __name__ == '__main__':
    total_count = 308
    # l_fr = []
    ampl = 1
    teta = 2.9
    alfa = 0.005
    betta = 0.999
    # teta = 2.6
    # bitr = 20
    shift = 40
    input_folder = "D:/pythonProject/phase_wm/synthesis_video/"
    # input_folder = "D:/pythonProject/phase_wm/frames_orig_video/"
    output_folder = "D:/pythonProject/phase_wm/frames_after_emb/"
    PATH_IMG = r"D:\pythonProject/Phase_WM_Clear/data/attempt_new_spatial_spectr_1024_in_shift_40_wm_49.png"

    img_wm = io.imread(PATH_IMG)

    # count = read_video(r'D:/pythonProject/phase_wm/Road.mp4',
    #                    input_folder, total_count)
    #

    bitr = "orig"
    for ampl in [1, 2, 3, 4]:
        embed(input_folder, output_folder, PATH_IMG, ampl, teta)
        psnr_full = 0
        for i in range(50):
            image1 = cv2.imread("D:\pythonProject\phase_wm/frames_after_emb/frame" + str(i) + ".png")
            image2 = cv2.imread("D:\pythonProject\phase_wm/frames_orig_video/frame" + str(i) + ".png")

            psnr_full += (cv2.PSNR(image1, image2))

        print("A = ", ampl, "PSNR: ", psnr_full / 50)

        if bitr != 50:
            generate_video(bitr, output_folder)
        rand_k = 0
        vot_sp = []
        stop_kadr1 = []
        # stop_kadr2 = []
        # stop_kadr1_bin = []
        # stop_kadr2_bin = []

        # total_count = 2997

        var_list, ext_values = extract(alfa, betta, teta, img_wm.shape[0], rand_k, shift)

        # with open(
        #         r'D:/pythonProject/Phase_WM_Clear\data/var_list_49_1024_no_smooth_union_on_%d_center_' % shift + str(
        #             ampl) + '_bitr' + str(
        #             bitr) + "_shift" + str(shift) + '.txt',
        #         'w') as file:
        #     for var in var_list:
        #         file.write(str(var) + "\n")
        #
        # with open(
        #         r'D:/pythonProject/Phase_WM_Clear\data/acc_list_49_1024_no_smooth_union_on_%d_center_' % shift + str(
        #             ampl) + '_bitr' + str(
        #             bitr) + "_shift" + str(shift) + '.txt',
        #         'w') as file:
        #     for val in ext_values:
        #         file.write(str(val) + "\n")

    # plt.plot(var_list)
    # plt.grid(True)
    # plt.show()

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
from helper_methods import csv2list, bit_voting, compare_qr
from reedsolomon import extract_RS, Nbit
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
    sort_name_img = sort_spis(images, "frame")[:total_count]
    cnt = 0

    while cnt < len(sort_name_img):
        # Reads in BGR format
        imgg = cv2.imread(folder_orig_image + sort_name_img[cnt])
        # translation to the YCrCb space
        a = cv2.cvtColor(imgg, cv2.COLOR_BGR2YCrCb)
        # a = a.astype(float)

        temp = fi * st_qr
        # A*sin(m * teta + fi)
        wm = np.array((amplitude * np.sin(cnt * tt + temp)))

        # if my_i == 1:
        #     wm = np.where(wm>0,1,-1)
        # Embedding in the Y-channel
        a[0:512, 0:512, 0] = np.where(np.float32(a[0:512, 0:512, 0] + wm) > 255, 255,
                                      np.where(a[0:512, 0:512, 0] + wm < 0, 0, np.float32(a[0:512, 0:512, 0] + wm)))

        # a[20:1060, 440:1480, 0] = np.where(np.float32(a[20:1060, 440:1480, 0] + wm[:, :, 0]) > 255, 255,
        #                                    np.where(a[20:1060, 440:1480, 0] + wm[:, :, 0] < 0, 0,
        #                                             np.float32(a[20:1060, 440:1480, 0] + wm[:, :, 0])))
        tmp = cv2.cvtColor(a, cv2.COLOR_YCrCb2BGR)

        # Converting the YCrCb matrix to BGR
        img_path = os.path.join(folder_to_save)
        cv2.imwrite(img_path + "frame" + str(cnt) + ".png", tmp)

        if cnt % 700 == 0:
            print("wm embed", cnt)

        cnt += 1


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


def extract(alf, beta, tt, size_wm, rand_fr):
    """
    Procedure embedding
    :param alf: primary smoothing parameter
    :param beta: primary smoothing parameter
    :param tt:reference frequency
    :param size_wm: side of embedding watermark
    :param rand_fr: the frame from which the extraction begins
    :return: the path to the final image
    """
    PATH_VIDEO = r'D:/pythonProject/phase_wm\frames_after_emb\RB_codec.mp4'

    count = read_video(PATH_VIDEO, 'D:/pythonProject/phase_wm/extract/')

    cnt = int(rand_fr)
    g = np.asarray([])
    f = g.copy()
    f1 = f.copy()

    while cnt < total_count:
        arr = io.imread(r"D:/pythonProject/phase_wm\extract/frame" + str(cnt) + ".png")

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
        img = Image.fromarray(f1.astype('uint8'))
        if cnt % 700 == 0:
            print("first smooth", cnt)
        img.save(r'D:/pythonProject/phase_wm\extract\first_smooth/result' + str(cnt) + '.png')

        cnt += 1

    variance = []
    cnt = int(rand_fr)
    g = np.asarray([])
    f = g.copy()
    d = g.copy()

    g2 = np.zeros((512, 512), dtype=np.complex_)
    f2 = np.zeros((512, 512), dtype=np.complex_)
    d2 = np.zeros((512, 512), dtype=np.complex_)
    # count = total_count

    # reading a shuffled object

    # subtracting the average
    while cnt < count:

        arr = np.float32(cv2.imread(r"D:/pythonProject/phase_wm/extract/first_smooth/result" + str(cnt) + ".png"))
        # arr = np.float32(cv2.imread(r"D:/pythonProject/phase_wm/extract/frame" + str(cnt) + ".png"))
        a = cv2.cvtColor(arr, cv2.COLOR_BGR2YCrCb)

        f1 = np.float32(
            cv2.imread(r"D:/pythonProject/phase_wm\extract\frame" + str(cnt) + ".png"))
        f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2YCrCb)
        a1 = np.where(a < f1, f1 - a, a - f1)

        a1 = a1[0:512, 0:512, 0]
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
            d = np.ones((size_wm, size_wm))

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

        tmp_signal = np.zeros((512, 512), dtype=np.complex_)
        tmp_signal.real = yc
        tmp_signal.imag = ys

        if cnt == rand_fr:
            f2 = tmp_signal
            d2 = np.ones((size_wm, size_wm), dtype=np.complex_)
            d2.imag = np.ones((size_wm, size_wm))

        else:
            if cnt == rand_fr + 1:
                f2.real = 2 * beta * np.cos(tt) * np.float32(d2.real) + np.float32(tmp_signal.real)
                f2.imag = 2 * beta * np.cos(tt) * np.float32(d2.imag) + np.float32(tmp_signal.imag)
            else:
                f2.real = 2 * beta * np.cos(tt) * np.float32(d2.real) - (beta ** 2) * np.float32(g2.real) + np.float32(
                    tmp_signal.real)
                f2.imag = 2 * beta * np.cos(tt) * np.float32(d2.imag) - (beta ** 2) * np.float32(g2.imag) + np.float32(
                    tmp_signal.imag)

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
        variance.append(np.var(l_kadr - img_wm))
        if cnt % 20 == 19:

            spector = check_spatial2spectr(l_kadr)
            stop_kadr1.append(compare_qr(
                spector, io.imread("D:\pythonProject/Phase_WM_Clear/data/check_ifft_wm.png")))
            if cnt % 500 == 499:
                print(ampl, cnt, stop_kadr1)

        cnt += 1

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
    sort_name_img = sort_spis(images, "frame")
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
        os.system(f"ffmpeg -y -i D:/pythonProject/phase_wm/frames_after_emb/need_video.mp4 -b:v {bitr}M -vcodec"
                  f" libx264  D:/pythonProject/phase_wm/frames_after_emb/RB_codec.mp4")


def vot_by_variance(path_imgs, start, end, treshold):
    var_list = csv2list(r"D:/pythonProject/\phase_wm/RB_disp.csv")[start:end]
    sum_matrix = np.zeros((int(img_wm.shape[0] / 16), int(img_wm.shape[1] / 16)))
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
    img1 = Image.fromarray(sum_matrix.astype('uint8'))
    img1.save(r"D:/pythonProject/phase_wm\voting" + ".png")
    comp = compare_qr(r"D:/pythonProject/phase_wm\voting" + ".png", io.imread(PATH_IMG))
    print(count)
    print(comp)
    # extract_RS(sum_matrix, rsc, Nbit)

    return comp


if __name__ == '__main__':

    # l_fr = []
    ampl = 1
    teta = 2.9
    alfa = 0.0005
    betta = 0.999
    # teta = 2.6
    bitr = "orig"
    input_folder = "D:/pythonProject/phase_wm/frames_orig_video/"
    output_folder = "D:/pythonProject/phase_wm/frames_after_emb/"
    PATH_IMG = r"D:\pythonProject/Phase_WM_Clear/data/spatial_spectr_wm_65.png"

    img_wm = io.imread(PATH_IMG)

    # count = read_video(r'D:/pythonProject/phase_wm/cut_RealBarca120.mp4',
    #                   input_folder)
    for ampl in [1]:
        rand_k = 0
        vot_sp = []
        stop_kadr1 = []
        # stop_kadr2 = []
        # stop_kadr1_bin = []
        # stop_kadr2_bin = []

        total_count = 2997

        embed(input_folder, output_folder, PATH_IMG, ampl, teta)
        generate_video(bitr, output_folder)
        var_list, ext_values = extract(alfa, betta, teta, img_wm.shape[0], rand_k)

        # print("Variance", var_list)
        with open(
                r'D:/pythonProject/Phase_WM_Clear\data/var_list_no_smooth_' + str(ampl) + '_bitr' + str(bitr) + '.txt',
                'w') as file:
            for var in var_list:
                file.write(str(var) + "\n")

        with open(
                r'D:/pythonProject/Phase_WM_Clear\data/acc_list_no_smooth_' + str(ampl) + '_bitr' + str(bitr) + '.txt',
                'w') as file:
            for val in ext_values:
                file.write(str(val) + "\n")
    # plt.plot(var_list)
    # plt.grid(True)
    # plt.show()

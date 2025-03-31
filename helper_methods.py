import numpy as np
import re
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
import csv
import cv2
from PIL import Image

# from reedsolomon import Nbit

size_quadr = 16


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


def read_video(path, path_to_save, final_frame):
    vidcap = cv2.VideoCapture(path)
    count_frame = 0
    success = True
    pix100 = []
    while success and count_frame < final_frame:
        success, image = vidcap.read()

        if success:
            cv2.imwrite(path_to_save + "/frame%d.png" % count_frame, image)
            pix100.append(image[100, 100, 0])
        if count_frame % 25 == 24:
            print("записан кадр", count_frame, )

        if cv2.waitKey(10) == 27:
            break
        count_frame += 1
    return count_frame, pix100


def sort_spis(sp, keyword):
    """

    :param sp: list of images
    :param keyword: name of these images
    :return: sorted by number list of images
    """
    spp = []
    spb = []
    res = []
    for i in sp:
        spp.append("".join(re.findall(r'\d', i)))
        spb.append(keyword)
    result = [int(item) for item in spp]
    result.sort()

    result1 = [str(item) for item in result]
    for k in range(len(sp)):
        res.append(spb[k] + result1[k] + ".png")
    return res


def img2bin(img):
    """

    :param img: grayscale image
    :return: binary image with average by squares
    """
    k = 0

    our_avg = np.mean(img)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            tmp = img[i, j]

            if tmp > our_avg:
                img[i, j] = 255
            else:
                img[i, j] = 0

            k += 1
    return img


def big2small(st_qr):
    """

    :param st_qr: big qr with square 16x16
    :return: qr with suqare 1x1 with average by 16x16
    """
    qr = np.zeros((int(st_qr.shape[0] / 16), int(st_qr.shape[1] / 16)))

    for i in range(0, st_qr.shape[0], size_quadr):
        for j in range(0, st_qr.shape[1], size_quadr):
            mean_now = np.mean(st_qr[i:i + size_quadr, j:j + size_quadr])
            qr[int(i / size_quadr), int(j / size_quadr)] = mean_now

    return qr


def small2big(sm_qr):
    """

    :param sm_qr: small qr with square 1x1
    :return: big qr with square 16x16
    """
    qr = np.zeros((sm_qr.shape[0] * 16, sm_qr.shape[1] * 16))

    for i in range(0, int(sm_qr.shape[0])):
        for j in range(0, int(sm_qr.shape[1])):
            tmp = sm_qr[i, j]
            qr[i * 16:i * 16 + 16, j * 16:j * 16 + 16].fill(tmp)
    qr = np.where(qr == 1, 255, 0)
    return qr


def disp(path, word='/frame', name_of_doc='LG_disp.csv'):
    cnt = 0
    arr = np.array([])

    total_count = len(list(Path(path).iterdir()))

    list_diff = []
    while cnt < 1377:
        tmp = np.copy(arr)
        arr = io.imread(path + word + str(cnt) + ".png").astype(float)
        if cnt == 0:
            list_diff.append(0)

        else:
            diff_img = np.abs(arr - tmp)

            list_diff.append(np.mean(diff_img))
        if cnt % 100 == 0:
            print(cnt)
        cnt += 1

    max_val_list = max(list_diff)
    list_diff = list_diff / max_val_list / 2

    with open(name_of_doc, 'w') as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(list_diff)

    plt.plot(list_diff)
    plt.show()

    avg = sum(list_diff) / len(list_diff)

    upd_start = []
    for i in range(len(list_diff)):
        if abs((list_diff[i])) > (4 * avg):
            upd_start.append(i)

    return list_diff


def csv2list(path_csv):
    """

    :param path_csv:
    :return:  parsing csv
    """
    with open(path_csv, newline='') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        l = []
        for row in readCSV:
            l.append(row)
    need_list = l[0]
    result = [float(item) for item in need_list]
    return result


def bit_voting(image, count):
    vec_np = np.reshape(image, image.shape[0] * image.shape[1])[0:count]
    vec_np[vec_np == 255] = 1
    vec_vot = np.zeros(int(count / 7))
    for i in range(len(vec_vot)):
        for j in range(0, 7):
            vec_vot[i] += vec_np[j * len(vec_vot) + i]
    vec_vot[vec_vot < (7 / 2)] = 0
    vec_vot[vec_vot > (7 / 2)] = 255

    long_vec = np.zeros(vec_np.shape)
    for i in range(0, 7):
        tmp = i * len(vec_vot)
        tmp2 = (i + 1) * len(vec_vot)
        long_vec[tmp:tmp2] = vec_vot
    for i in range(image.shape[0] * image.shape[1] - count):
        vec_np = np.append(vec_np, 0)
    matr_aft_vot = np.reshape(vec_np, (image.shape[0], image.shape[1]))
    matr_aft_vot[matr_aft_vot == 1] = 255
    return matr_aft_vot


def compar_before_after_saving(folderb4, foldera5):
    b4_00 = []
    a5_00 = []
    for i in range(2997):
        b4_00.append(io.imread(folderb4 + "/result" + str(i) + ".png")[0, 0, 0])
        a5_00.append(io.imread(foldera5 + "/frame" + str(i) + ".png")[0, 0, 0])
        print(i)

    print(b4_00)
    print(a5_00)


def disp_pix(path, coord_x, coord_y, total_count):
    cnt = 1
    arr = np.array([])

    list_diff = []
    while cnt < total_count:
        tmp = np.copy(arr[coord_x, coord_y])
        arr = cv2.imread(path + str(cnt) + ".png").astype(float)[coord_x, coord_y]

        diff_pix = np.abs(arr - tmp)
        print(np.mean(diff_pix), " frame ", cnt)
        list_diff.append(np.mean(diff_pix))
        cnt += 1

    avg = sum(list_diff) / len(list_diff)
    max_list = max(list_diff)
    print(avg)
    for i in range(len(list_diff)):
        list_diff[i] /= max_list

    return list_diff


def create_gray_bg():
    frame = np.full((1080, 1920), 127)
    orig_qr = io.imread(r"D:\pythonProject\\phase_wm/some_qr.png")

    for cnt in range(1000):
        wm = np.asarray((-1) ** (cnt + orig_qr / 255))
        print(np.max(wm), np.min(wm))
        final = np.copy(frame)
        final[20:1060, 440:1480] = frame[20:1060, 440:1480] + wm
        imgc = Image.fromarray(final.astype('uint8'))
        imgc.save(
            r"D:/phase_wm_graps/BBC/gray_background/result" + str(cnt) + ".png")


def decode_wm(wm, path_to_save):
    decoding_qr = np.zeros(wm.shape)

    wm = np.where(wm == 255, 1, 0)
    for i in range(wm.shape[0]):
        for j in range(wm.shape[1]):
            if j != 0:
                if wm[i, j] == 0:
                    decoding_qr[i, j] = abs(decoding_qr[i, j - 1] - 1)
                else:
                    decoding_qr[i, j] = decoding_qr[i, j - 1]
            else:
                if wm[i, j] == 0:
                    decoding_qr[i, j] = 1
                else:
                    decoding_qr[i, j] = 0

    decoding_qr = np.where(decoding_qr == 1, 255, 0)
    img = Image.fromarray(decoding_qr.astype('uint8'))

    img.convert('RGB').save(path_to_save)

    return decoding_qr

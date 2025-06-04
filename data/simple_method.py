import cv2
import numpy as np
import os
from PIL import Image, ImageFile
from skimage import io
from skimage.exposure import histogram
# from qrcode_1 import read_qr, correct_qr
from helper_methods import big2small, sort_spis, img2bin, small2big
from helper_methods import csv2list, decode_wm

# from reedsolomon import extract_RS, rsc, Nbit

ImageFile.LOAD_TRUNCATED_IMAGES = True
size_quadr = 16


def read_video(path, path_to_save, final_frame):
    vidcap = cv2.VideoCapture(path)
    count_frame = 0
    success = True
    while success and (count_frame < final_frame):
        success, image = vidcap.read()
        if success:
            cv2.imwrite(path_to_save + "/frame%d.png" % count_frame, image)

        if count_frame % 25 == 24:
            print("записан кадр", count_frame)

        if cv2.waitKey(10) == 27:
            break
        count_frame += 1
    return count_frame


def embed(my_i, count, var):
    cnt = 0

    PATH_IMG = r"D:\pythonProject\Phase_WM_Clear\data/test_qr_89_89.png"

    st_qr = cv2.imread(PATH_IMG)
    st_qr = cv2.cvtColor(st_qr, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    data_length = st_qr.size
    shuf_order = np.arange(data_length)

    np.random.seed(42)
    np.random.shuffle(shuf_order)

    # Expand the binary image into a string
    st_qr_1d = st_qr.ravel()
    shuffled_data = st_qr_1d[shuf_order]  # Shuffle the original data

    # 1d-string in the image
    pict = np.resize(shuffled_data, (1057, 1920))
    # the last elements are uninformative. Therefore, we make zeros
    pict[-1, 256 - 1920:] = 0

    while cnt < count:
        imgg = cv2.imread(r"D:\phase_wm_graps\BBC\frames_orig_video/frame%d.png" % cnt)

        a = cv2.cvtColor(imgg, cv2.COLOR_BGR2YCrCb)

        temp = np.where(pict == 255, 1, 0)

        wm_n = np.asarray(my_i * ((-1) ** (cnt + temp)))

        a[0:1057, :, 0] = np.where(np.float32(a[0:1057, :, 0] + wm_n) > 255, 255,
                                   np.where(a[0:1057, :, 0] + wm_n < 0, 0, np.float32(a[0:1057, :, 0] + wm_n)))

        tmp = cv2.cvtColor(a, cv2.COLOR_YCrCb2RGB)
        img = Image.fromarray(tmp.astype('uint8'))

        img.save(r"D:/phase_wm_graps/BBC\frames_after_emb\result" + str(cnt) + ".png")
        if cnt % 100 == 0:
            print("wm embed", cnt)
        cnt += 1


def read2list(file):
    # открываем файл в режиме чтения utf-8
    file = open(file, 'r', encoding='utf-8')

    # читаем все строки и удаляем переводы строк
    lines = file.readlines()
    lines = [line.rstrip('\n') for line in lines]

    file.close()

    return lines


def extract(rand_fr, tresh):
    size_wm = 1424

    path_of_video = r'D:/phase_wm_graps/BBC/frames_after_emb\RB_codec.mp4'
    vidcap = cv2.VideoCapture(path_of_video)
    vidcap.open(path_of_video)
    alf = 0.01
    list00 = []
    beta = 0.99

    # count = 0
    read_video(path_of_video, r"D:/phase_wm_graps/BBC/extract", total_count)
    print("pixels after saving", list00)
    f1 = np.zeros((1080, 1920))
    cnt = int(rand_fr)

    while cnt < total_count:
        filename = f"D:/phase_wm_graps/BBC/extract/frame{cnt}.png"
        if not os.path.exists(filename):
            print(f"[Ошибка] Файл не найден: {filename}")
            continue
        arr = io.imread(filename)

        if cnt == rand_fr:
            f1 = arr
            diff = np.clip(arr, -2, 2)
            x1 = np.zeros((1080, 1920))

        else:
            # diff = np.clip(np.float32(arr) - np.float32(x1), -2, 2)
            diff = np.float32(arr) - np.float32(x1)
            f1 = np.float32(diff) - np.float32(arr) * (alf)

        x1 = np.copy(arr)

        # f1[f1 > 255] = 255
        # f1[f1 < 0] = 0
        img = Image.fromarray(f1.astype('uint8'))
        if cnt % 80 == 0:
            print("first smooth", cnt)
        img.save(r'D:/phase_wm_graps/BBC/extract/first_smooth/result' + str(cnt) + '.png')

        cnt += 1

    g = np.asarray([])
    f = g.copy()

    d = g.copy()
    vot_sp = []

    sp0 = []
    sp1 = []
    cnt = int(rand_fr)
    count = total_count
    shuf_order = read2list(r'D:\pythonProject\\phase_wm\shuf.txt')
    shuf_order = [eval(i) for i in shuf_order]
    # вычитание усреднённого
    while cnt < total_count:
        orig_arr = np.float32(cv2.imread(r"D:/phase_wm_graps/BBC\extract/frame" + str(cnt) + ".png"))

        f1 = cv2.cvtColor(orig_arr[:1057, :], cv2.COLOR_BGR2YCrCb)

        # f1 = f1[:, :, 0]
        arr = np.float32(io.imread(r"D:/phase_wm_graps/BBC\extract/first_smooth/result" + str(cnt) + ".png"))
        a = cv2.cvtColor(arr[:1057, :], cv2.COLOR_BGR2YCrCb)

        a1 = a - f1
        a1 = a1[:, :, 0]

        res_1d = np.ravel(a1)[:256 - 1920]
        start_qr = np.resize(res_1d, (size_wm, size_wm))

        unshuf_order = np.zeros_like(shuf_order)
        unshuf_order[shuf_order] = np.arange(start_qr.size)
        unshuffled_data = np.ravel(start_qr)[unshuf_order]
        matr_unshuf = np.resize(unshuffled_data, (size_wm, size_wm))
        a1 = matr_unshuf

        # извлечение ЦВЗ

        # tmp = a_main*0.5 + a_main_1*0.5
        # tmp = a_main
        if cnt == rand_fr:
            f = np.copy(a1)
            d = np.zeros((1424, 1424))

        else:
            f = -beta * np.float32(d) + np.float32(a1)
            sp0.append(f[120, 0])
            sp1.append(f[0, 0])
            d = np.copy(f)

        a1 = f

        small_frame = big2small(a1)
        if cnt != 0:
            if cnt % 2 == 1:
                wm = np.where(small_frame >= 0, 255, 0)
            else:
                wm = np.where(small_frame >= 0, 0, 255)
        else:
            wm = np.zeros(small_frame.shape)

        img = Image.fromarray(wm.astype('uint8'))
        img.save(r'D:/phase_wm_graps/BBC\extract/wm/result' + str(cnt) + '.png')

        if cnt % 5 == 4:
            v = vot_by_variance(r"D:/phase_wm_graps/BBC\extract/wm", 0, cnt, tresh)
            vot_sp.append(np.round(max(v, 1 - v), 4))
            # if extract_RS(io.imread(
            #         r"D:/phase_wm_graps/BBC\extract/after_normal_phas_bin/result" + str(cnt) + ".png"),
            #         rsc, Nbit) == b'':
            #
            #     stop_kadr1_bin.append(0)
            # else:
            #     stop_kadr1_bin.append(0)
            #
            stop_kadr1.append(np.round(max(compare(
                r"D:/phase_wm_graps/BBC\extract/wm/result" + str(cnt) + ".png"), 1 - compare(
                r"D:/phase_wm_graps/BBC\extract/wm/result" + str(cnt) + ".png")), 4))
            if cnt % 50 == 49:
                print(cnt, stop_kadr1)
                print(vot_sp)

        cnt += 1

    return r"D:/phase_wm_graps/BBC\extract/after_normal_phas_bin/result" + str(total_count - 1) + ".png"


def generate_video(bitr, image_folder):
    """
    Sequence of frames transform to compress video
    :param image_folder: folder for output frames

    :param bitr: bitrate of output video
    """

    os.chdir(image_folder)

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".png")]
    sort_name_img = sort_spis(images, "result")[:total_count]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    if bitr != "orig":
        video_name = 'need_video.mp4'
    else:
        video_name = "RB_codec.mp4"

    # fourcc = cv2.VideoWriter_fourcc(*'H264')

    video = cv2.VideoWriter(video_name, 0, 29.97, (width, height))

    cnt = 0
    for image in sort_name_img:
        video.write(cv2.imread(os.path.join(image_folder, image)))
        if cnt % 799 == 0:
            print(cnt)
        cnt += 1
    cv2.destroyAllWindows()
    video.release()

    if bitr != "orig":
        print("Codec worked")
        os.system(f"ffmpeg -y -i D:/phase_wm_graps/BBC/frames_after_emb/need_video.mp4 -b:v {bitr}M -vcodec"
                  f" libx264 D:/phase_wm_graps/BBC/frames_after_emb/RB_codec.mp4")


def compare(path):  # сравнивание извлечённого QR с исходным
    orig_qr = io.imread(r"D:\pythonProject\Phase_WM_Clear\data/test_qr_89_89.png")
    orig_qr = np.where(orig_qr > 127, 255, 0)
    small_qr = big2small(orig_qr)
    sr_matr = np.zeros((1424, 1424, 3))
    myqr = io.imread(path)
    myqr = np.where(myqr > 127, 255, 0)

    k = 0
    mas_avg = []
    for i in range(0, 89):
        for j in range(0, 89):

            if np.mean(small_qr[i, j]) == np.mean(myqr[i, j]):
                sr_matr[i, j] = 1
                mas_avg.append(1)
            else:
                sr_matr[i, j] = 0
                mas_avg.append(0)

    for i in mas_avg:
        if i == 1:
            k += 1
    return k / len(mas_avg)


def diff_pix_between_neugb(qr1, qr2):
    k = 0
    mas_avg = []
    for i in range(0, 89):
        for j in range(0, 89):

            if qr1[i, j] == qr2[i, j]:
                mas_avg.append(1)
            else:
                mas_avg.append(0)

    for i in mas_avg:
        if i == 0:
            k += 1
    return k


def vot_by_variance(path_imgs, start, end, treshold):
    var_list = csv2list(r"D:\pythonProject\\phase_wm/RB_disp.csv")[start:end]
    # var_list = [0, 36, 77, 82, 120, 136, 184, 243, 278, 285, 290, 291, 308, 345, 348, 365, 375, 394, 403, 467]

    sum_matrix = np.zeros((89, 89))
    np_list = np.array(var_list)
    need_ind = [i for i in range(len(np_list)) if np_list[i] > treshold]
    i = start
    count = 0
    while i < end:
        c_qr = io.imread(path_imgs + r"/result" + str(i) + ".png")

        if len(c_qr.shape) == 3:
            c_qr = c_qr[:, :, 0]
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
    img1.save(r"D:/phase_wm_graps/BBC/voting/vot" + str(count) + ".png")
    comp = compare(r"D:/phase_wm_graps/BBC/voting/vot" + str(count) + ".png")
    # print(count)
    # print(comp)
    if comp < 0.5:
        sum_matrix = np.where(sum_matrix == 255, 0, 255)
    img1 = Image.fromarray(sum_matrix.astype('uint8'))
    img1.save(r"D:/phase_wm_graps/BBC/voting/vot" + str(count) + ".png")
    # extract_RS(sum_matrix, rsc, Nbit)

    return comp


if __name__ == '__main__':

    PATH_VIDEO = "D:/pythonProject/phase_wm/cut_RealBarca120.mp4"
    output_folder = "D:/phase_wm_graps/BBC/frames_after_emb/"

    rand_k = 0
    total_count = 107

    hm_list = []
    alfa = 0.01
    sp = []

    ampl = 1
    embed(ampl, total_count, 0)
    for bitr in [7]:
        generate_video(bitr, output_folder)
        stop_kadr1 = []
        stop_kadr1_bin = []
        stop_kadr2_bin = []
        print('GEN')
        path_extract_code = extract(rand_k, 0.045)
        print("all")
        print("brate = ", bitr, "random frame = ", rand_k, alfa, "current percent", stop_kadr1)
    hand_made = [0, 118, 404, 414, 524, 1002, 1391, 1492, 1972, 2393, 2466, total_count]

    print(alfa, "current percent", stop_kadr1)

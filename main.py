# -*- coding: utf-8 -*-
import math
from skimage import io
import psutil
# from reedsolo import RSCodec
from skimage.exposure import histogram
import cv2
import os
import gc
import numpy as np
from PIL import Image, ImageFile
# from qrcode_1 import read_qr, correct_qr
from helper_methods import small2big, big2small, sort_spis, read_video
from helper_methods import csv2list, bit_voting, read2list


def psnr_hvs_m_simple(img1, img2, weights=(0.299, 0.587, 0.114)):
    """
    Упрощённая версия PSNR-HVS-M.

    :param img1: numpy array, эталонное изображение (RGB или grayscale)
    :param img2: numpy array, искажённое изображение
    :param weights: веса для RGB каналов (по умолчанию ITU-R BT.601)
    :return: PSNR_HVS_M значение в дБ
    """
    # Проверка размеров
    if img1.shape != img2.shape:
        raise ValueError("Изображения должны быть одинакового размера")

    # Если изображения цветные — переводим в оттенки серого с учётом весов
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        diff = np.abs(img1.astype('float32') - img2.astype('float32'))
        weighted_diff = np.dot(diff, weights)
    else:
        diff = np.abs(img1.astype('float32') - img2.astype('float32'))
        weighted_diff = diff

    # Среднеквадратичная ошибка с учетом веса
    mse = np.mean(weighted_diff ** 2)

    if mse == 0:
        return float('inf')

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


# from reedsolomon import extract_RS, Nbit

ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm


def is_image_valid(filepath):
    """Проверяет, можно ли корректно прочитать изображение."""
    try:
        image = cv2.imread(filepath)
        if image is None:
            return False
        return True
    except Exception as e:
        print(f"Ошибка при чтении {filepath}: {e}")
        return False


def reextract_corrupted_frames(video_path, image_dir, max_frame=None):
    """Перезаписывает повреждённые кадры из видео."""
    # Получаем список всех frame файлов
    frame_files = sorted(
        [f for f in os.listdir(image_dir) if f.startswith("frame") and f.endswith(".png")],
        key=lambda x: int(x[5:-4])  # Сортировка по номеру кадра
    )

    if not frame_files:
        print("Нет кадров для проверки.")
        return

    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Не удалось открыть видеофайл.")

    # Подготавливаем путь для сохранения
    corrupted_count = 0

    for filename in frame_files:
        frame_idx = int(filename[5:-4])
        if max_frame is not None and frame_idx >= max_frame:
            break

        filepath = os.path.join(image_dir, filename)

        # Проверяем целостность изображения
        if not is_image_valid(filepath):
            print(f"Кадр {filename} повреждён. Перезаписываю...")

            # Перематываем видео до нужного кадра
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, image = cap.read()

            if success:
                cv2.imwrite(filepath, image)
                print(f"Кадр {filename} успешно перезаписан.")
                corrupted_count += 1
            else:
                print(f"Не удалось извлечь кадр {frame_idx} из видео.")


def embed(folder_orig_image, folder_to_save, binary_image, amplitude, tt, count, var):
    """
    Procedure embedding
    :param count: count of frames for embedding
    :param binary_image: embedding code
    :param folder_orig_image: the folder from which the original images are taken
    :param folder_to_save: the folder where the images from the watermark are saved
    :param amplitude: embedding amplitude
    :param tt: reference frequency parameter
    """

    fi = math.pi / 2 / 255
    st_qr = cv2.imread(binary_image)
    st_qr = cv2.cvtColor(st_qr, cv2.COLOR_BGR2YCrCb)

    lst_100 = []

    data_length = st_qr[:, :, 0].size
    shuf_order = np.arange(data_length)

    np.random.seed(42)
    np.random.shuffle(shuf_order)

    # Expand the binary image into a string
    st_qr_1d = st_qr[:, :, 0].ravel()
    shuffled_data = st_qr_1d[shuf_order]  # Shuffle the original data

    # 1d-string in the image
    pict = np.resize(shuffled_data, (1057, 1920))
    # the last elements are uninformative. Therefore, we make zeros
    pict[-1, 256 - 1920:] = 0

    images = [img for img in os.listdir(folder_orig_image)
              if img.endswith(".png")]

    # The list should be sorted by numbers after the name
    sort_name_img = sort_spis(images, "frame")[:count]
    cnt = 0

    while cnt < len(sort_name_img):

        try:
            imgg = cv2.imread(folder_orig_image + sort_name_img[cnt])
            if imgg is None:
                raise ValueError(f"Изображение не загружено: {folder_orig_image + sort_name_img[cnt]}")
            a = cv2.cvtColor(imgg, cv2.COLOR_BGR2YCrCb)
        except Exception as e:
            print(f"[Ошибка] {e} в файле {folder_orig_image + sort_name_img[cnt]}")

        # translation to the YCrCb space
        # a = cv2.cvtColor(imgg, cv2.COLOR_BGR2YCrCb)
        # a = a.astype(float)

        temp = fi * pict
        # A*sin(m * teta + fi)
        wm = np.array((amplitude * np.sin(cnt * tt + temp)))

        copy_a = np.copy(a)
        if amplitude == 1:
            wm = np.where(wm > 0, 1, -1)
        # Embedding in the Y-channel
        a[0:1057, :, 0] = np.where(np.float32(a[0:1057, :, 0] + wm) > 255, 255,
                                   np.where(a[0:1057, :, 0] + wm < 0, 0, np.float32(a[0:1057, :, 0] + wm)))

        lst_100.append(a[100, 100, 0] - copy_a[100, 100, 0])
        # print(lst_100)
        # a[20:1060, 440:1480, 0] = np.where(np.float32(a[20:1060, 440:1480, 0] + wm[:, :, 0]) > 255, 255,
        #                                    np.where(a[20:1060, 440:1480, 0] + wm[:, :, 0] < 0, 0,
        #                                             np.float32(a[20:1060, 440:1480, 0] + wm[:, :, 0])))
        tmp = cv2.cvtColor(a, cv2.COLOR_YCrCb2RGB)

        row, col, ch = tmp.shape
        mean = 0
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(tmp.shape)
        noisy = np.clip(tmp + gauss, 0, 255)
        # Converting the YCrCb matrix to BGR
        img_path = os.path.join(folder_to_save)
        img = Image.fromarray(noisy.astype('uint8'))

        img.save(img_path + "frame" + str(cnt) + ".png")
        if cnt % 100 == 0:
            print("wm embed", cnt)

        cnt += 1

        with open('diff_wm.txt', 'w') as f:
            for line in lst_100:
                f.write(f"{line}\n")


import cv2


def extract(alf, beta, tt, size_wm, rand_fr, count):
    """
    Procedure embedding
    :param count: count of frames for extracting
    :param alf: primary smoothing parameter
    :param beta: primary smoothing parameter
    :param tt:reference frequency
    :param size_wm: side of embedding watermark
    :param rand_fr: the frame from which the extraction begins
    :return: the path to the final image
    """
    PATH_VIDEO = r'D:/pythonProject/phase_wm\frames_after_emb\RB_codec.mp4'

    read_video(PATH_VIDEO, 'D:/pythonProject/phase_wm/extract/', count)
    # reextract_corrupted_frames(PATH_VIDEO, 'D:/pythonProject/phase_wm/extract/', total_count)
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

    pix100_smooth = []
    gc.collect()

    while cnt < count:
        if cnt % 250 == 249:
            print('After create dataset The CPU usage is: ', psutil.virtual_memory().percent)

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

            # Гарантируем корректные значения
            f1 = np.clip(f1, 0, 255)

            # Проверяем данные перед сохранением
            if np.any(np.isnan(f1)) or np.any(f1 < 0) or np.any(f1 > 255):
                print(f"Invalid data in frame {cnt} - min: {np.min(f1)}, max: {np.max(f1)}")
                f1 = np.nan_to_num(f1)
                f1 = np.clip(f1, 0, 255)

            try:
                img = Image.fromarray(f1.astype('uint8'))
                # Альтернативный вариант сохранения
                img.save(r'D:/pythonProject/phase_wm\extract\first_smooth/result' + str(cnt) + '.png',
                         compress_level=6, optimize=True)
            except Exception as e:
                print(f"Error saving frame {cnt}: {e}")
                # Попробуем сохранить через OpenCV
                try:

                    cv2.imwrite(r'D:/pythonProject/phase_wm\extract\first_smooth/result' + str(cnt) + '.png',
                                cv2.cvtColor(f1.astype('uint8'), cv2.COLOR_RGB2BGR))
                except Exception as e2:
                    print(f"Also failed with OpenCV: {e2}")

        if cnt % 300 == 0:
            print("first smooth", cnt)

        del arr
        gc.collect()

        cnt += 1

    cnt = int(rand_fr)
    g = np.asarray([])
    f = g.copy()
    d = g.copy()

    # reading a shuffled object
    shuf_order = read2list(r'D:/pythonProject/phase_wm\shuf.txt')
    shuf_order = [eval(i) for i in shuf_order]
    # subtracting the average
    while cnt < count:
        if is_image_valid(r"D:/pythonProject/phase_wm/extract/first_smooth/result" + str(cnt) + ".png"):
            arr = np.float32(cv2.imread(r"D:/pythonProject/phase_wm/extract/first_smooth/result" + str(cnt) + ".png"))
            a = cv2.cvtColor(arr[:1057, :], cv2.COLOR_BGR2YCrCb)
        else:
            arr = cv2.imread(r"D:/pythonProject/phase_wm\extract\frame" + str(cnt) + ".png")
            a = cv2.cvtColor(arr[:1057, :], cv2.COLOR_BGR2YCrCb)
        # a = a[:, :, 0]

        try:
            f1 = cv2.imread(r"D:/pythonProject/phase_wm\extract\frame" + str(cnt) + ".png")
            if f1 is None:
                raise ValueError(
                    f"Изображение не загружено: D:/pythonProject/phase_wm\extract/frame" + str(cnt) + ".png")
            f1 = cv2.cvtColor(f1[:1057, :], cv2.COLOR_BGR2YCrCb)
        except Exception as e:
            print(f"[Ошибка] {e} в файле D:/pythonProject/phase_wm\extract/frame" + str(cnt) + ".png")

        # a1 = np.where(a < f1, f1 - a, a - f1)
        # a1 = np.where(a < f1, f1 - a, 0)
        a1 = a - f1

        a1 = a1[:, :, 0]
        res_1d = np.ravel(a1)[:256 - 1920]
        start_qr = np.resize(res_1d, (size_wm, size_wm))

        unshuf_order = np.zeros_like(shuf_order)
        unshuf_order[shuf_order] = np.arange(start_qr.size)
        unshuffled_data = np.ravel(start_qr)[unshuf_order]
        matr_unshuf = np.resize(unshuffled_data, (size_wm, size_wm))
        a = matr_unshuf
        # extraction of watermark
        # a = a1[20:1060, 440:1480, 0]
        g = np.copy(d)
        d = np.copy(f)

        if cnt == rand_fr:
            f = np.copy(matr_unshuf)
            d = np.ones((size_wm, size_wm))

        else:
            if cnt == rand_fr + 1:
                f = 2 * beta * np.cos(tt) * np.float32(d) + np.float32(a)

            else:
                f = 2 * beta * np.cos(tt) * np.float32(d) - (beta ** 2) * np.float32(g) + np.float32(a)

        yc = np.float32(f) - beta * np.cos(tt) * np.float32(d)
        ys = beta * np.sin(tt) * np.float32(d)

        c = np.cos(tt * cnt) * np.float32(yc) + np.sin(tt * cnt) * np.float32(ys)
        s = np.cos(tt * cnt) * np.float32(ys) - np.sin(tt * cnt) * np.float32(yc)

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

        # img = Image.fromarray(wm.astype('uint8'))
        # img.save(r"D:/pythonProject/phase_wm\extract/before_normalize/result" + str(cnt) + ".png")

        a1 = wm

        fi = (a1 * np.pi * 2) / 255

        # if cnt > 63:
        #     # loc_hist = np.histogram(a1.flatten(), 255, (0, 255))
        #     plt.hist(fi.flatten(), bins=255)
        #     plt.xlabel("Значение полученной фазы", fontsize=20)
        #     plt.ylabel("Количество пикселей", fontsize=20)
        #     plt.show()
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

        img = Image.fromarray(l_kadr.astype('uint8'))
        img.save(r"D:/pythonProject/phase_wm\extract/after_normal_phas/result" + str(cnt) + ".png")

        l_kadr = io.imread(
            r'D:/pythonProject/phase_wm\extract/after_normal_phas/result' + str(cnt) + '.png').astype(
            float)
        cp = big2small(l_kadr.copy())

        our_avg = np.mean(cp)
        cp = np.where(cp > our_avg, 255, 0)

        # cp = bit_voting(cp, Nbit)
        imgc = Image.fromarray(cp.astype('uint8'))

        imgc.save(
            r"D:/pythonProject/phase_wm\extract/after_normal_phas_bin/result" + str(cnt) + ".png")

        if len(vot_sp) >= 10 and all(x > 0.99 for x in vot_sp[-10:]):
            return stop_kadr1, vot_sp

        if cnt % 5 == 4:
            v = vot_by_variance(r"D:/pythonProject/phase_wm\extract\after_normal_phas_bin", max(0, cnt - 400), cnt,
                                0.045)
            vot_sp.append(np.round(max(v, 1 - v), 4))
            # extract_RS(cp,
            #            106, 127, Nbit)
            stop_kadr1.append(np.round(max(compare(
                r"D:/pythonProject/phase_wm\extract/after_normal_phas_bin/result" + str(cnt) + ".png",
                io.imread(PATH_IMG)),
                1 - compare(
                    r"D:/pythonProject/phase_wm\extract/after_normal_phas_bin/result" + str(
                        cnt) + ".png", io.imread(PATH_IMG))), 4))
            if cnt % 20 == 19:
                print(tt, alf, cnt, stop_kadr1)
                print("after voting", tt, alf, vot_sp)

        cnt += 1

    return stop_kadr1, vot_sp


def generate_video(bitr, image_folder, st_frame=0):
    """
    Sequence of frames transform to compress video
    :param st_frame: frame which start extraction
    :param image_folder: Folder which save all pictures after embedding
    :param bitr: bitrate of output video
    """

    if bitr != "orig":
        video_name = 'need_video.avi'
    else:
        video_name = "RB_codec.mp4"
    os.chdir(r"D:/pythonProject/phase_wm\frames_after_emb")

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".png")]
    sort_name_img = sort_spis(images, "frame")[st_frame:total_count + st_frame]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    video = cv2.VideoWriter(video_name, fourcc, 29.97, (width, height))

    cnt = 0
    for image in sort_name_img:
        video.write(cv2.imread(os.path.join(image_folder, image)))
        if cnt % 299 == 0:
            print(cnt)
        cnt += 1
    cv2.destroyAllWindows()
    video.release()

    if bitr != "orig":
        os.system(
            f"ffmpeg -y -i D:/pythonProject/phase_wm/frames_after_emb/need_video.avi -b:v {bitr}M -c:v libx264 "
            f"D:/pythonProject/phase_wm/frames_after_emb/RB_codec.mp4")

    return "D:/pythonProject/phase_wm/frames_after_emb/RB_codec.mp4"


def compare(path, orig_qr):
    """
     Comparing the extracted QR with the original one
    :param path: path to code for comparison
    :return: percentage of similarity
    """

    orig_qr = np.where(orig_qr > 127, 255, 0)
    small_qr = big2small(orig_qr)

    myqr = io.imread(path)
    myqr = np.where(myqr > 127, 255, 0)

    sr_matr = small_qr == myqr
    k = np.count_nonzero(sr_matr)
    return k / sr_matr.size


def vot_by_variance(path_imgs, start, end, treshold):
    var_list = csv2list(r"D:/pythonProject/phase_wm/RB_disp.csv")[start:end]
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
    comp = compare(r"D:/pythonProject/phase_wm\voting" + ".png", io.imread(PATH_IMG))

    return comp


def reduce_to_blocks(img, block_size=16):
    h, w = img.shape
    return img.reshape(h // block_size, block_size, w // block_size, block_size).mean(axis=(1, 3)) > 0.5


def vot_weighted(path_imgs, start, end, threshold=0.5):
    H_BLOCKS, W_BLOCKS = 89, 89
    sum_matrix = np.zeros((H_BLOCKS, W_BLOCKS), dtype=np.float32)
    frame_weights = []

    total_frames = end - start
    for i in tqdm(range(start, end)):
        try:
            img = io.imread(f"{path_imgs}/result{i}.png")
            if img.ndim == 3:
                img = img[:, :, 0]  # grayscale if RGB

            # Вес: линейно растущий
            weight = (i - start + 1) / total_frames
            frame_weights.append(weight)

            sum_matrix += img.astype(np.float32) * weight
        except Exception as e:
            print(f"Ошибка при чтении {i}: {e}")

    # Максимальный возможный вес (сумма всех весов)
    total_weight = sum(frame_weights)

    # Порог — голосование: если набрано больше threshold * total_weight → 1
    vote_result = (sum_matrix >= threshold * 255 * total_weight).astype(np.uint8) * 255

    # Сохраняем результат
    Image.fromarray(vote_result).save(os.path.join(path_imgs, "voting.png"))
    comp = compare(os.path.join(path_imgs, "voting.png"), io.imread(PATH_IMG))

    return comp


if __name__ == '__main__':
    l_fr = []
    ampl = 1
    alfa = 0.001
    betta = 0.999
    teta = 2.9
    bitr = "orig"
    total_count = 297
    input_folder = "D:/pythonProject/phase_wm/frames_orig_video/"
    output_folder = "D:/pythonProject/phase_wm/frames_after_emb/"
    # PATH_IMG = r"D:/pythonProject//phase_wm\qr_ver18_H.png"
    PATH_IMG = r"D:\pythonProject\Phase_WM_Clear\data/test_qr_89_89.png"
    img_wm = io.imread(PATH_IMG)
    for vid_name in ["cut_RealBarca120", "IndiDance", "Road"]:

        read_video(r'D:/pythonProject/phase_wm/' + vid_name + '.mp4',
                   input_folder, total_count)
        embed(input_folder, output_folder, PATH_IMG, ampl, teta, total_count, 0)
        for rand_frame in [25, 50, 100]:

            generate_video(bitr, output_folder, rand_frame)

            vot_sp = []
            stop_kadr1 = []

            stop_kadr, vot_sp_final = extract(alfa, betta, teta, img_wm.shape[0], 0, total_count)

            print("Acc-cy of last frame", stop_kadr[-1])

            # Запись vot_sp_final в файл
            with open(f'a{ampl}_vot_sp_final_{vid_name}_no_c_alf_rf_mjpg.txt', 'a') as f_vot:
                f_vot.write(f"rand_frame={rand_frame}: {vot_sp_final}\n")

            # Запись stop_kadr1 в файл
            with open(f'a{ampl}_stop_kadr_{vid_name}_no_c_alf_001_var_mjpg.txt', 'a') as f_stop:
                f_stop.write(f"rand_frame={rand_frame}: {stop_kadr}\n")

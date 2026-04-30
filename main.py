# -*- coding: utf-8 -*-
import math
from skimage import io
import psutil
import time
from skimage.exposure import histogram
import cv2
import os
import gc
import numpy as np
from PIL import Image, ImageFile
# from qrcode_1 import read_qr, correct_qr
from helper_methods import small2big, big2small, sort_spis, read_video
from helper_methods import csv2list, bit_voting, read2list

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


def embed(folder_orig_image, folder_to_save, binary_image, amplitude, tt, count, var, rand_seed=42, L=64):
    """
    Adaptive embedding for any frame size.
    """
    fi = math.pi / (2 * 255)

    # Load and preprocess QR code (grayscale Y channel)
    st_qr = cv2.imread(binary_image)
    qr_y = st_qr[:, :, 0]  # Shape: (H_qr, W_qr)

    # Get dimensions of the first frame to determine target size
    images = [img for img in os.listdir(folder_orig_image) if img.endswith(".png")]
    if not images:
        raise ValueError("No frames found in input folder")
    first_img_path = os.path.join(folder_orig_image, sort_spis(images, "frame")[0])
    first_frame = cv2.imread(first_img_path)
    if first_frame is None:
        raise ValueError("Failed to read first frame")
    frame_h, frame_w = first_frame.shape[:2]

    # Now reshape shuffled data to match the actual frame region we'll embed into

    temp_phase = fi * qr_y

    # Process frames
    sort_name_img = sort_spis(images, "frame")[:count - 1]

    for cnt, img_name in enumerate(sort_name_img):
        img_path = os.path.join(folder_orig_image, img_name)
        imgg = cv2.imread(img_path)
        if imgg is None:
            print(f"[Ошибка] Изображение не загружено: {img_path}")
            continue

        a = cv2.cvtColor(imgg, cv2.COLOR_BGR2YCrCb).astype(np.float32)

        # Generate watermark pattern (same size as embedded region)
        wm = amplitude * np.sin(tt * cnt + temp_phase)

        if amplitude == 1:
            wm = np.where(wm > 0, 1.0, -1.0)

        # Embed into top-left region of Y channel
        y_region = a[:, :, 0]
        new_y = y_region + wm
        np.clip(new_y, 0, 255, out=new_y)
        a[:, :, 0] = new_y

        # Convert back and add noise
        tmp = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_YCrCb2RGB).astype(np.float32)

        if var > 0:
            noise = np.random.normal(0, np.sqrt(var), tmp.shape)
            tmp = np.clip(tmp + noise, 0, 255)

        output_img = Image.fromarray(tmp.astype(np.uint8))
        output_img.save(os.path.join(folder_to_save, f"frame{cnt}.png"))

        if cnt % 100 == 0:
            print("wm embed", cnt)


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
    PATH_VIDEO = r'D:/python_projects/phase_wm\frames_after_emb\RB_codec.avi'

    read_video(PATH_VIDEO, 'D:/python_projects/phase_wm/extract/', count)

    cnt = int(rand_fr)
    g = np.asarray([])
    f = g.copy()
    f1 = f.copy()

    pix100_smooth = []
    gc.collect()

    while cnt < count:
        if cnt % 250 == 249:
            print('After create dataset The CPU usage is: ', psutil.virtual_memory().percent)

        arr = io.imread(r"D:/python_projects/phase_wm\extract/frame" + str(cnt) + ".png")

        d1 = f1
        if cnt == rand_fr:
            f1 = arr.astype('float32')
            d1 = np.zeros((240, 320))
        # elif cnt == change_sc[scene-1] + 1:
        else:
            f1 = np.float32(d1) * alf + np.float32(arr) * (1 - alf)
            # else:
            #     f1 = (1-alf)*(1-alf)*a+(1-alf)*alf*d1+alf*g1

            # Гарантируем корректные значения
        f1 = np.clip(f1, 0, 255)

        try:
            img = Image.fromarray(f1.astype('uint8'))
            # Альтернативный вариант сохранения
            img.save(r'D:/python_projects/phase_wm\extract\first_smooth/result' + str(cnt) + '.png')
        except Exception as e:
            print(f"Error saving frame {cnt}: {e}")

        if cnt % 300 == 0:
            print("first smooth", cnt)

        del arr
        gc.collect()

        cnt += 1

    cnt = int(rand_fr)
    g = np.asarray([])
    f = g.copy()
    d = g.copy()
    fi_list = []
    c_list = []
    s_list = []
    yc_list = []
    ys_list = []
    qr_true = io.imread(binary_img_path)

    # subtracting the average
    while cnt < count:

        arr = cv2.imread(r"D:/python_projects/phase_wm/extract/first_smooth/result" + str(cnt) + ".png")
        a = cv2.cvtColor(arr[:, :], cv2.COLOR_BGR2YCrCb)
        # a = a[:, :, 0]

        try:
            f1 = cv2.imread(r"D:/python_projects/phase_wm\extract\frame" + str(cnt) + ".png")
            if f1 is None:
                raise ValueError(
                    f"Изображение не загружено: D:/python_projects/phase_wm\extract/frame" + str(cnt) + ".png")
            f1 = cv2.cvtColor(f1[:, :], cv2.COLOR_BGR2YCrCb)
        except Exception as e:
            print(f"[Ошибка] {e} в файле D:/python_projects/phase_wm\extract/frame" + str(cnt) + ".png")

        # a1 = np.where(a < f1, f1 - a, a - f1)
        # a1 = np.where(a < f1, f1 - a, 0)
        a1 = a - f1
        a1 = a1[:, :, 0]

        a = a1
        # extraction of watermark
        # a = a1[20:1060, 440:1480, 0]
        g = np.copy(d)
        d = np.copy(f)

        if cnt == rand_fr:
            f = np.copy(a1)
            d = np.ones((qr_true.shape[0], qr_true.shape[1]))

        else:
            if cnt == rand_fr + 1:
                f = 2 * beta * np.cos(tt) * np.float32(d) + np.float32(a)

            else:
                f = 2 * beta * np.cos(tt) * np.float32(d) - (beta ** 2) * np.float32(g) + np.float32(a)

        yc = np.float32(f) - beta * np.cos(tt) * np.float32(d)
        ys = beta * np.sin(tt) * np.float32(d)

        c = np.cos(tt * (cnt % 200)) * np.float32(yc) + np.sin(tt * (cnt % 200)) * np.float32(ys)
        s = np.cos(tt * (cnt % 200)) * np.float32(ys) - np.sin(tt * (cnt % 200)) * np.float32(yc)

        yc_list.append(yc[100, 100])
        ys_list.append(ys[100, 100])

        c_list.append(c[100, 100])
        s_list.append(s[100, 100])

        try:
            fi = np.where(c < 0, np.arctan((s / c)) + np.pi,
                          np.where(s >= 0, np.arctan((s / c)), np.arctan((s / c)) + 2 * np.pi))
        except ZeroDivisionError:
            fi = np.full(f.shape, 255)
        fi = np.nan_to_num(fi)
        fi = np.where(fi < -np.pi / 4, fi + 2 * np.pi, fi)
        fi = np.where(fi > 9 * np.pi / 4, fi - 2 * np.pi, fi)

        fi_list.append(fi[100, 100])
        wm = 255 * fi / 2 / math.pi

        wm[wm > 255] = 255
        wm[wm < 0] = 0

        # img = Image.fromarray(wm.astype('uint8'))
        # img.save(r"D:/python_projects/phase_wm\extract/before_normalize/result" + str(cnt) + ".png")

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
        img.save(r"D:/python_projects/phase_wm\extract/after_normal_phas/result" + str(cnt) + ".png")

        l_kadr = io.imread(
            r'D:/python_projects/phase_wm\extract/after_normal_phas/result' + str(cnt) + '.png').astype(
            float)
        cp = big2small(l_kadr.copy())

        our_avg = np.mean(cp)
        cp = np.where(cp > our_avg, 255, 0)

        # cp = bit_voting(cp, Nbit)
        imgc = Image.fromarray(cp.astype('uint8'))

        imgc.save(
            r"D:/python_projects/phase_wm\extract/after_normal_phas_bin/result" + str(cnt) + ".png")

        if len(vot_sp) >= 10 and all(x > 0.99 for x in vot_sp[-10:]):
            return stop_kadr1, vot_sp

        if cnt % 5 == 4:
            v = vot_by_variance(r"D:/python_projects/phase_wm\extract\after_normal_phas_bin", max(0, cnt - 400), cnt,
                                0.045)
            vot_sp.append(np.round(max(v, 1 - v), 4))
            # extract_RS(cp,
            #            106, 127, Nbit)
            stop_kadr1.append(np.round(max(compare(
                r"D:/python_projects/phase_wm\extract/after_normal_phas_bin/result" + str(cnt) + ".png",
                io.imread(binary_img_path)),
                1 - compare(
                    r"D:/python_projects/phase_wm\extract/after_normal_phas_bin/result" + str(
                        cnt) + ".png", io.imread(binary_img_path))), 4))
            # if cnt % 20 == 19:
            #     print(tt, alf, cnt, stop_kadr1)
            #     print("after voting", tt, alf, vot_sp)

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
        video_name = "RB_codec.avi"
    os.chdir(r"D:/python_projects/phase_wm\frames_after_emb")

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
            f"ffmpeg -y -loglevel quiet -nostats -i D:/python_projects/phase_wm/frames_after_emb/need_video.avi -b:v {bitr}M -c:v libx264 "
            f"D:/python_projects/phase_wm/frames_after_emb/RB_codec.avi")

    return "D:/python_projects/phase_wm/frames_after_emb/RB_codec.avi"


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
    var_list = csv2list(r"D:/python_projects/Phase_WM_Clear/RB_disp.csv")[start:end]
    sum_matrix = np.zeros((15, 20))
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
    img1.save(r"D:/python_projects/phase_wm\voting" + ".png")
    comp = compare(r"D:/python_projects/phase_wm\voting" + ".png", io.imread(binary_img_path))

    return comp


def create_binary_watermark(bit_matrix, block_size=16, output_path=None):
    """
    Создаёт изображение из битовой матрицы, где каждый бит = block_size × block_size пикселей.

    :param bit_matrix: 2D numpy array of shape (H_bits, W_bits), dtype=bool or int (0/1)
    :param block_size: размер блока в пикселях (по умолчанию 16)
    :param output_path: путь для сохранения PNG (опционально)
    :return: np.ndarray of shape (H_px, W_px), dtype=uint8 (0 или 255)
    """
    # Убедимся, что bit_matrix — булево или 0/1
    bit_matrix = np.array(bit_matrix, dtype=bool)
    bits_h, bits_w = bit_matrix.shape

    # Размер в пикселях
    px_h = bits_h * block_size
    px_w = bits_w * block_size

    # Создаём изображение: 0 = чёрный (бит 0), 255 = белый (бит 1)
    img = np.zeros((px_h, px_w), dtype=np.uint8)

    for i in range(bits_h):
        for j in range(bits_w):
            val = 255 if bit_matrix[i, j] else 0
            # Заполняем блок размером block_size × block_size
            img[i * block_size:(i + 1) * block_size,
            j * block_size:(j + 1) * block_size] = val

    # Сохраняем, если указан путь
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(img).save(output_path)
        print(f"Binary watermark saved to: {output_path}")
        print(f"Size: {img.shape} ({bits_h}×{bits_w} bits, {block_size}×{block_size} per bit)")

    return img


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
    comp = compare(os.path.join(path_imgs, "voting.png"), io.imread(binary_img_path))

    return comp


def contrast_video(video_path, alpha=1.5, beta=0):
    """
    Применяет контрастирование к видеофайлу.
    """
    if not os.path.exists(video_path):
        print(f"Файл не найден: {video_path}")
        return

    # Генерируем имя временного файла корректно для любого расширения
    base, ext = os.path.splitext(video_path)
    temp_path = f"{base}_temp{ext}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        return

    # Получаем параметры видео
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Пробуем использовать тот же кодек, что и у исходника, или mp4v по умолчанию
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(temp_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Применяем контраст и яркость
        adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        out.write(adjusted_frame)
        frame_count += 1

    cap.release()
    out.release()

    # Проверяем, что временный файл действительно создан и не пуст
    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
        # Удаляем оригинал
        os.remove(video_path)
        # Переименовываем временный в оригинал
        os.rename(temp_path, video_path)
        print(f"Контрастирование завершено. Обработано кадров: {frame_count}")
    else:
        print(f"Ошибка: Временный файл {temp_path} не был создан корректно.")
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == '__main__':
    l_fr = []
    ampl = 1
    alfa = 0.1
    betta = 0.999
    teta = 3
    bitr = "orig"
    total_count = 450
    input_folder = "D:/python_projects/phase_wm/frames_orig_video/"
    output_folder = "D:/python_projects/phase_wm/frames_after_emb/"
    # PATH_IMG = r"D:/python_projects//phase_wm\qr_ver18_H.png"
    PATH_IMG = r"D:/python_projects/Phase_WM_Clear/data/RS_cod89x89.png"
    img_wm = io.imread(PATH_IMG)

    import os


    def get_video_files(folder_path):
        video_extensions = {'.avi', '.mp4'}
        video_files = []
        for file in os.listdir(folder_path):
            _, ext = os.path.splitext(file)
            if ext.lower() in video_extensions:
                video_files.append(os.path.join(folder_path, file))
        return sorted(video_files)


    def get_subfolders(directory):
        """Возвращает список путей ко всем непосредственным подпапкам."""
        return [
            os.path.join(directory, item)
            for item in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, item))
        ]


    # Пример
    folder = r"D:/python_projects/Phase_WM_Clear/dataset/UCF-101"
    subfolders = get_subfolders(folder)
    for sf in subfolders:

        # Пример использования
        folder = sf
        videos = get_video_files(folder)
        for v in videos:
            print(v)
        last_acc = []
        for vid_name in videos:
            qr_bits = np.random.randint(0, 2, size=(15, 20))  # замените на реальный QR

            # Создаём изображение 89*16 × 89*16 = 1424×1424
            binary_img_path = r"D:/python_projects/phase_wm/qr_15x20_16px.png"
            create_binary_watermark(qr_bits, block_size=16, output_path=binary_img_path)

            c_frame, _ = read_video(vid_name, input_folder, total_count)
            print(c_frame)
            for variance in [10]:
                # start = time.perf_counter()
                embed(input_folder, output_folder, binary_img_path, ampl, teta, c_frame, variance)
                # end = time.perf_counter()
                # print(f"Время выполнения: {end - start:.6f} секунд")
                generate_video(bitr, output_folder, 0)
                output_video_path = r'D:/python_projects/phase_wm\frames_after_emb\RB_codec.avi'
                if os.path.exists(output_video_path):
                    # alpha=1.5 увеличит контраст на 50%. Подберите значение под задачу.
                    contrast_video(output_video_path, alpha=1.5, beta=1)
                else:
                    print(f"Предупреждение: Файл {output_video_path} не найден для контрастирования.")
                # ---------------------
                vot_sp = []
                stop_kadr1 = []

                stop_kadr, vot_sp_final = extract(alfa, betta, teta, img_wm.shape[0], 0, c_frame)

                print("Acc-cy of last frame", stop_kadr[-1])
                last_acc.append(stop_kadr1[-1])

                # # Запись vot_sp_final в файл
                # with open(f'a{ampl}_vot_sp_final_{vid_name}_no_c_alf_noise_mjpg.txt', 'a') as f_vot:
                #     f_vot.write(f"variance={variance}: {vot_sp_final}\n")
                #
                # # Запись stop_kadr1 в файл
                # with open(f'a{ampl}_stop_kadr_{vid_name}_no_c_alf_001_noise_mjpg.txt', 'a') as f_stop:
                #     f_stop.write(f"variance={variance}: {stop_kadr}\n")
        print("list of acc", last_acc, sf)

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


def disp_pix(path, coord_x, coord_y):
    cnt = 1
    arr = np.array([])

    list_diff = []
    while cnt < total_count:
        tmp = np.copy(arr[coord_x, coord_y])
        arr = io.imread(path + str(cnt) + ".png").astype(float)[coord_x, coord_y]

        diff_pix = np.abs(arr - tmp)
        print(np.mean(diff_pix), " frame ", cnt)
        list_diff.append(np.mean(diff_pix))
        cnt += 1

    avg = sum(list_diff) / len(list_diff)
    print(avg)
    upd_start = []
    for i in range(len(list_diff)):
        if abs((list_diff[i])) > (4 * avg):
            upd_start.append(i)

    print("frame with change scene", upd_start)
    return list_diff


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


def diff(fold_BBC, fold_orig):
    for cnt in range(0, 3000):
        img_path = os.path.join(fold_orig, f"result{cnt}.png")
        imgg = cv2.imread(img_path)
        img_BBC = os.path.join(fold_BBC, f"result{cnt}.png")
        BBC = cv2.imread(img_BBC)
        cur_diff = np.abs(BBC - imgg)
        # print(diff)
        img_path1 = os.path.join("D:/phase_wm_graps/BBC/diff_BBC-orig", f"result{cnt}.png")
        cv2.imwrite(img_path1, cur_diff)


def smoothing(path, filename, path_to_save, coef):
    stop_kadr2 = []

    cnt = 2
    g2 = io.imread(
        path + "/" + filename + str(cnt - 1) + ".png")
    f = np.copy(g2)
    # alf2 = 0.13

    # for i in range(1, count):
    #
    #     curr_img = io.imread(path + "/" + filename + str(i) + ".png")
    #
    #     if np.max(curr_img) != 0:
    #         mid = (np.max(curr_img) / 2 + np.min(curr_img) / 2)
    #         if mid > 127.5:
    #             new_img = curr_img - (mid - 127.5)
    #         else:
    #             new_img = curr_img + (127.5 - mid)
    #         # norm_img = curr_img/ np.max(curr_img)
    #         #
    #         # norm_img*=255
    #         img = Image.fromarray(new_img.astype('uint8'))
    #         img.save(path + "/normalize/" + filename + str(i) + ".png")

    while cnt < total_count:
        arr = io.imread(
            path + "/" + filename + str(cnt) + ".png")
        # g2 - y(n-1)
        y_step_1 = f

        # y(n)=alfa*y(n-1)+x(n)*(1-alfa)
        f = y_step_1 * coef + arr * (1 - coef)

        sum_over_border = np.sum(f > 255)
        if sum_over_border > 0:
            print("Sum>255", sum_over_border)
        f[f > 255] = 255
        # print(cnt, np.max(f), np.min(f))
        img = Image.fromarray(f.astype('uint8'))
        img.save(path_to_save + "/" + filename + str(cnt) + ".png")
        if cnt % 300 == 0:
            print("wm 2 smooth", cnt)
        cnt += 1

    vot_sp = []

    cnt = 0
    while cnt < total_count:
        c_qr = io.imread(path_to_save + "/" + filename + str(cnt) + ".png")
        c_qr = img2bin(c_qr)
        # c_qr = correct_qr(c_qr)
        img1 = Image.fromarray(c_qr.astype('uint8'))
        img1.save(r"D:/phase_wm_graps/BBC\extract/wm_after_2_smooth_bin/result" + str(cnt) + ".png")
        if cnt % 300 == 0:
            print("2 smooth bin", cnt)
        if cnt % 100 == 96:
            v = vot_by_variance(r"D:/phase_wm_graps/BBC\extract\after_normal_phas_bin", 0, cnt, 0.045)
            vot_sp.append(max(v, 1 - v))
            # if extract_RS(io.imread(
            #         r"D:/phase_wm_graps/BBC\extract/wm_after_2_smooth_bin/result" + str(cnt) + ".png"),
            #         rsc, Nbit) != b'':
            #     stop_kadr2_bin.append(1)
            # else:
            #     stop_kadr2_bin.append(0)
            stop_kadr2.append(max(
                compare(
                    r"D:/phase_wm_graps/BBC\extract/wm_after_2_smooth_bin/result" + str(cnt) + ".png"), 1 - compare(
                    r"D:/phase_wm_graps/BBC\extract/wm_after_2_smooth_bin/result" + str(cnt) + ".png")))
            print("2 voting", v)
        cnt += 1

    print("variance ON", vot_sp)
    print(stop_kadr2)


def embed(my_i, count, var):
    cnt = 0

    PATH_IMG = r"D:/pythonProject//phase_wm\some_qr.png"

    st_qr = cv2.imread(PATH_IMG)
    st_qr = cv2.cvtColor(st_qr, cv2.COLOR_RGB2YCrCb)[:, :, 0]

    # small_qr = big2small(st_qr[:,:,0])
    # coding_qr = np.zeros(small_qr.shape)
    #
    # for i in range(small_qr.shape[0]):
    #     for j in range(small_qr.shape[1]):
    #         if j!=0:
    #             coding_qr[i][j]= small_qr[i][j]==small_qr[i][j-1]
    #         else:
    #             coding_qr[i][j] = small_qr[i][j] == 0
    # coding_qr[:,0] = 1
    # big_qr = small2big(small_qr)

    img = Image.fromarray(st_qr.astype('uint8'))

    img.convert('RGB').save(r"D:/phase_wm_graps/BBC/reformat_qr.png")

    while cnt < count:
        imgg = cv2.imread(r"D:\phase_wm_graps\BBC\frames_orig_video/frame%d.png" % cnt)
        a = imgg
        a = cv2.cvtColor(imgg, cv2.COLOR_RGB2YCrCb)

        temp = np.where(st_qr == 255, 1, 0)
        # wm = np.asarray(my_i * ((-1) ** cnt) * temp)
        wm_n = np.asarray(my_i * ((-1) ** (cnt + temp)))

        a[20:1060, 440:1480, 0] = np.where(np.float32(a[20:1060, 440:1480, 0] + wm_n) > 255, 255,
                                           np.where(a[20:1060, 440:1480, 0] + wm_n < 0, 0,
                                                    np.float32(a[20:1060, 440:1480, 0] + wm_n)))
        tmp = cv2.cvtColor(a, cv2.COLOR_YCrCb2BGR)

        row, col, ch = tmp.shape
        mean = 0
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(tmp.shape)
        # gauss[gauss < 0] = 0
        noisy = tmp + gauss

        # tmp=a

        img = Image.fromarray(noisy.astype('uint8'))

        img.convert('RGB').save(r"D:/phase_wm_graps/BBC\frames_after_emb\result" + str(cnt) + ".png")
        if cnt % 100 == 0:
            print("wm embed", cnt)
        cnt += 1
    # print(shuf_order)


def read2list(file):
    # открываем файл в режиме чтения utf-8
    file = open(file, 'r', encoding='utf-8')

    # читаем все строки и удаляем переводы строк
    lines = file.readlines()
    lines = [line.rstrip('\n') for line in lines]

    file.close()

    return lines


def extract(rand_fr, tresh):
    # path_of_video = r'D:/phase_wm_graps/BBC/frames_after_emb\need_video.mp4'
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
        arr = io.imread(r"D:/phase_wm_graps/BBC/extract/frame" + str(cnt) + ".png")
        # a = arr
        #
        # d1 = f1
        # if cnt == rand_fr:
        #     f1 = a.copy()
        #     d1 = np.zeros((1080, 1920))
        #
        # else:
        #     f1 = np.float32(d1) * (1-alf) + np.float32(a) * (alf)

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
    # shuf_order = read2list(r'D:\pythonProject\\phase_wm\shuf.txt')
    # shuf_order = [eval(i) for i in shuf_order]
    # вычитание усреднённого
    while cnt < total_count:
        orig_arr = np.float32(cv2.imread(r"D:/phase_wm_graps/BBC\extract/frame" + str(cnt) + ".png"))
        if orig_arr.shape[0] < 1040 or orig_arr.shape[1] < 1480:
            print("cur cnt", cnt)

        f1 = cv2.cvtColor(orig_arr[20:1060, 440:1480], cv2.COLOR_BGR2YCrCb)

        # f1 = f1[:, :, 0]
        arr = np.float32(io.imread(r"D:/phase_wm_graps/BBC\extract/first_smooth/result" + str(cnt) + ".png"))
        a = cv2.cvtColor(arr[20:1060, 440:1480], cv2.COLOR_BGR2YCrCb)

        a1 = a - f1
        a1 = a1[:, :, 0]

        # извлечение ЦВЗ

        # tmp = a_main*0.5 + a_main_1*0.5
        # tmp = a_main
        if cnt == rand_fr:
            f = np.copy(a1)
            d = np.zeros((1040, 1040))
            # d = np.copy(a_main-a_main_1)
        # else:
        #     if cnt == rand_fr + 1:
        #         f = - 2 * beta * np.float32(d) + np.float32(a)

        else:
            f = -beta * np.float32(d) + np.float32(a1)
            sp0.append(f[120, 0])
            sp1.append(f[0, 0])
            d = np.copy(f)

        a1 = f
        small_frame = (big2small(a1))
        if cnt != 0:
            if cnt % 2 == 1:
                wm = np.where(small_frame >= 0, 255, 0)
            else:
                wm = np.where(small_frame >= 0, 0, 255)
        else:
            wm = np.zeros(small_frame.shape)

        img = Image.fromarray(wm.astype('uint8'))
        img.save(r'D:/phase_wm_graps/BBC\extract/wm/result' + str(cnt) + '.png')

        # decode_wm(wm,r'D:/phase_wm_graps/BBC\extract/after_normal_phas/result' + str(cnt) + '.png')
        #
        # img = Image.fromarray(decoding_qr.astype('uint8'))
        # img.save(r'D:/phase_wm_graps/BBC\extract/after_normal_phas/result' + str(cnt) + '.png')

        # if cnt % 100 == 99:
        #     print("0 spisok ",sp0)
        #     print("1 spisok ",sp1)
        # a1 = cv2.cvtColor(a1, cv2.COLOR_YCrCb2RGB)

        """
        # # привдение к рабочему диапазону
        #
        l_kadr = io.imread(r'D:/phase_wm_graps/BBC\extract/wm/result' + str(cnt) + '.png')
        if cnt!=0:
            l_kadr_n1 = io.imread(r'D:/phase_wm_graps/BBC\extract/wm/result' + str(cnt-1) + '.png')
            avg_lkadr = l_kadr*0.5+l_kadr_n1*0.5
            fi = avg_lkadr
        else:
            fi = l_kadr
        # fi_tmp = np.copy(fi)
        # fi = (l_kadr * np.pi * 2) / 255
        #
        # dis = []
        # coord1 = np.copy(fi)
        #
        # coord2 = np.copy(fi)
        # coord1 = np.where(fi < np.pi, (fi / np.pi * 2 - 1) * (-1),
        #                   np.where(fi > np.pi, ((fi - np.pi) / np.pi * 2 - 1), fi))
        # # list001.append(coord1[0,0])
        # coord2 = np.where(fi < np.pi / 2, (fi / np.pi / 2),
        #                   np.where(fi > 3 * np.pi / 2, ((fi - 1.5 * np.pi) / np.pi * 2) - 1,
        #                            ((fi - 0.5 * np.pi) * 2 / np.pi - 1) * (-1)))
        # # list_phas.append(coord2[0, 0])
        # hist, bin_centers = histogram(coord1, normalize=False)
        # hist2, bin_centers2 = histogram(coord2, normalize=False)
        a1 = fi
        fi = (a1 * np.pi * 2) / 255

        coord1 = np.where(fi < np.pi, (fi / np.pi * 2 - 1) * (-1), ((fi - np.pi) / np.pi * 2 - 1))
        coord2 = np.where(fi < np.pi / 2, (fi / np.pi / 2),
                          np.where(fi > 3 * np.pi / 2, ((fi - 1.5 * np.pi) / np.pi * 2) - 1,
                                   ((fi - 0.5 * np.pi) * 2 / np.pi - 1) * (-1)))

        # noinspection PyTypeChecker
        hist, bin_centers = histogram(coord1, normalize=False)
        # noinspection PyTypeChecker
        hist2, bin_centers2 = histogram(coord2, normalize=False)

        # ver = []
        # ver2 = []
        # mx_sp = np.arange(bin_centers[0], bin_centers[-1], bin_centers[1] - bin_centers[0])
        # for i in range(len(hist)):
        #     ver.append(hist[i] / sum(hist))
        # mo = moment = 0
        # for i in range(len(hist)):
        #     mo += bin_centers[i] * ver[i]
        # for mx in mx_sp:
        #     dis.append(abs(mo - mx))
        #
        # pr1 = 0
        # pr2 = 0
        # for i in range(len(dis)):
        #     if min(dis) == dis[i]:
        #         pr1 = bin_centers[i]

        # GPT
        mx_sp = np.arange(bin_centers2[0], bin_centers2[-1], bin_centers2[1] - bin_centers2[0])
        ver = hist2 / np.sum(hist)
        mo = np.sum(bin_centers2 * ver)
        dis = np.abs(mo - mx_sp)
        pr1 = np.min(dis)

        # dis2 = []
        # mx_sp2 = np.arange(bin_centers2[0], bin_centers2[-1], bin_centers2[1] - bin_centers2[0])
        # for i in range(len(hist2)):
        #     ver2.append(hist2[i] / sum(hist2))
        # mo = 0
        # for i in range(len(hist2)):
        #     mo += bin_centers2[i] * ver2[i]
        # for mx in mx_sp2:
        #     dis2.append(abs(mo - mx))
        #
        # x = min(dis2)

        # dis2 = []
        # mx_sp2 = np.arange(bin_centers2[0], bin_centers2[-1], bin_centers2[1] - bin_centers2[0])
        # for i in range(len(hist2)):
        #     ver2.append(hist2[i] / sum(hist2))
        # mo = 0
        # for i in range(len(hist2)):
        #     mo += bin_centers2[i] * ver2[i]
        # for mx in mx_sp2:
        #     dis2.append(abs(mo - mx))
        #
        # x = min(dis2)

        # GPT
        mx_sp2 = np.arange(bin_centers2[0], bin_centers2[-1], bin_centers2[1] - bin_centers2[0])
        ver2 = hist2 / np.sum(hist2)
        mo = np.sum(bin_centers2 * ver2)
        dis2 = np.abs(mo - mx_sp2)
        x = np.min(dis2)

        # for i in range(len(dis2)):
        #     if x == dis2[i]:
        #         pr2 = bin_centers2[i]

        # GPT
        idx = np.argmin(np.abs(dis2 - x))
        pr2 = bin_centers2[idx]

        moment = np.where(pr1 < 0, np.arctan((pr2 / pr1)) + np.pi,
                          np.where(pr2 >= 0, np.arctan((pr2 / pr1)), np.arctan((pr2 / pr1)) + 2 * np.pi))

        # if np.pi / 4 <= moment <= np.pi * 2 - np.pi / 4:
        #     fi_tmp = fi - moment + 0.5 * np.pi * 0.5
        #     fi_tmp = np.where(fi_tmp < -np.pi / 4, fi_tmp + 2 * np.pi, fi_tmp)
        #     fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)
        #
        # elif moment > np.pi * 2 - np.pi / 4:
        #     fi = np.where(fi < np.pi / 4, fi + 2 * np.pi, fi)
        #     fi_tmp = fi - moment + 0.5 * np.pi * 0.5
        #     fi_tmp = np.where(fi_tmp < -np.pi / 4, fi_tmp + 2 * np.pi, fi_tmp)
        #     fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)
        #
        # elif moment < np.pi / 4:
        #     fi_tmp = fi - 2 * np.pi - moment + 0.5 * np.pi * 0.5
        #     fi_tmp = np.where(fi_tmp < -np.pi / 4, fi_tmp + 2 * np.pi, fi_tmp)
        #     fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)

        if np.pi / 4 <= moment <= np.pi * 2 - np.pi / 4:
            fi_tmp = fi - moment + 0.5 * np.pi * 0.5

        elif moment > np.pi * 2 - np.pi / 4:
            fi = np.where(fi < np.pi / 4, fi + 2 * np.pi, fi)
            fi_tmp = fi - moment + 0.5 * np.pi * 0.5

        else:
            fi_tmp = fi - 2 * np.pi - moment + 0.5 * np.pi * 0.5

        fi_tmp = np.where(fi_tmp < -np.pi / 4, fi_tmp + 2 * np.pi, fi_tmp)
        fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)

        # list001.append(fi_tmp[0,0])
        fi_tmp = np.where(fi_tmp > np.pi, np.pi, np.where(fi_tmp < 0, 0, fi_tmp))
        l_kadr = fi_tmp * 255 / np.pi

        small_frame = big2small(l_kadr)
        img = Image.fromarray(small_frame.astype('uint8'))
        img.save(r"D:/phase_wm_graps/BBC\extract/after_normal_phas/result" + str(cnt) + ".png")

        l_kadr = io.imread(
            r'D:/phase_wm_graps/BBC\extract/after_normal_phas/result' + str(cnt) + '.png').astype(
            float)
        cp = l_kadr.copy()
        our_avg = np.mean(cp)
        cp = np.where(cp > our_avg, 255, 0)

        # cp = bit_voting(cp, Nbit)

        imgc = Image.fromarray(cp.astype('uint8'))

        imgc.save(
            r"D:/phase_wm_graps/BBC\extract/after_normal_phas_bin/result" + str(cnt) + ".png")

        if compare(r"D:/phase_wm_graps/BBC\extract/after_normal_phas_bin/result" + str(cnt) + ".png") < 0.5:
            cp = np.where(cp == 0, 255, 0)
            imgc = Image.fromarray(cp.astype('uint8'))

            imgc.save(
                r"D:/phase_wm_graps/BBC\extract/after_normal_phas_bin/result" + str(cnt) + ".png")
        """
        # print("wm extract", cnt)
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

    # smoothing(r"D:/phase_wm_graps/BBC\extract/wm", "result", "D:/phase_wm_graps/BBC/extract"
    #                                                                         "/wm_after_2_smooth", 0.13)
    """
    count = total_count

    cnt = int(rand_fr)
    g2 = np.asarray([])
    f = np.copy(g2)
    alf2 = 0.13

    while cnt < count:

        arr = io.imread(
            r"D:/phase_wm_graps/BBC/extract/after_normal_phas/result" + str(cnt) + ".png")
        # g2 - y(n-1)
        y_step_1 = f
        if cnt == rand_fr:
            f = arr.copy()
            f_step_1 = np.zeros((89, 89))
        else:
            # y(n)=alfa*y(n-1)+x(n)*(1-alfa)
            f = y_step_1 * alf2 + arr * (1 - alf2)
            f[f > 255] = 255

        img = Image.fromarray(f.astype('uint8'))
        img.save(r"D:/phase_wm_graps/BBC/extract/wm_after_2_smooth/result" + str(cnt) + ".png")
        if cnt % 300 == 0:
            print("wm 2 smooth", cnt)
        cnt += 1

    count = total_count
    cnt = int(rand_fr)
    while cnt < count:
        c_qr = io.imread(r"D:/phase_wm_graps/BBC/extract/wm_after_2_smooth/result" + str(cnt) + ".png")
        c_qr = img2bin(c_qr)
        # c_qr = correct_qr(c_qr)
        img1 = Image.fromarray(c_qr.astype('uint8'))
        img1.save(r"D:/phase_wm_graps/BBC/extract/wm_after_2_smooth_bin/result" + str(cnt) + ".png")
        if cnt % 300 == 0:
            print("2 smooth bin", cnt)
        if cnt % 100 == 96:
            v = vot_by_variance(r"D:/phase_wm_graps/BBC/extract/after_normal_phas_bin", 0, cnt, 0.045)
            vot_sp.append(max(v, 1 - v))
            # if extract_RS(io.imread(
            #         r"D:/phase_wm_graps/BBC/extract/wm_after_2_smooth_bin/result" + str(cnt) + ".png"),
            #         rsc, Nbit) != b'':
            #     stop_kadr2_bin.append(1)
            # else:
            #     stop_kadr2_bin.append(0)
            stop_kadr2.append(max(
                compare(
                    r"D:/phase_wm_graps/BBC/extract/wm_after_2_smooth_bin/result" + str(cnt) + ".png"), 1 - compare(
                    r"D:/phase_wm_graps/BBC/extract/wm_after_2_smooth_bin/result" + str(cnt) + ".png")))
            print("2 voting", v)
        cnt += 1
    """

    return r"D:/phase_wm_graps/BBC\extract/after_normal_phas_bin/result" + str(2996) + ".png"


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
        # if cnt % 300 == 0:

        video.write(cv2.imread(os.path.join(image_folder, image)))
        if cnt % 799 == 0:
            print(cnt)
        cnt += 1
    cv2.destroyAllWindows()
    video.release()

    if bitr != "orig":
        print("Codec worked")
        os.system(f"ffmpeg -y -i D:/phase_wm_graps/BBC/frames_after_emb/need_video.mp4 -b:v {bitr}M -vcodec"
                  f" mpeg2video  D:/phase_wm_graps/BBC/frames_after_emb/RB_codec.mp4")


def compare(path):  # сравнивание извлечённого QR с исходным
    orig_qr = io.imread(r"D:\pythonProject\\phase_wm/some_qr.png")
    orig_qr = np.where(orig_qr > 127, 255, 0)
    small_qr = big2small(orig_qr)
    sr_matr = np.zeros((1424, 1424, 3))
    myqr = io.imread(path)
    myqr = np.where(myqr > 127, 255, 0)

    k = 0
    mas_avg = []
    for i in range(0, 65):
        for j in range(0, 65):

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

    sum_matrix = np.zeros((65, 65))
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
    my_exit = []
    my_exit1 = []
    my_exit2 = []

    squ_size = 4
    for_fi = 6

    # графики-сравнения по различныи параметрам

    PATH_VIDEO = "D:/pythonProject/phase_wm/Road.mp4"
    input_folder = "D:/pythonProject/phase_wm/frames_orig_video/"
    output_folder = "D:/phase_wm_graps/BBC/frames_after_emb/"
    # with open('change_sc.csv', 'r') as f:
    #     change_sc = list(csv.reader(f))[0]
    #
    # change_sc = [eval(i) for i in change_sc]

    rand_k = 0
    total_count = 507

    hm_list = []

    # smoothing(r"D:/phase_wm_graps/BBC\extract/after_normal_phas", "result", "D:/phase_wm_graps/BBC\extract"
    #                                                                          "/wm_after_2_smooth", 0.1)
    alfa = 0.01
    sp = []

    # for tr in np.arange(0,206,20):

    # for st in range(96, 3000, 100):
    #     v = vot_by_variance(r"D:/phase_wm_graps/BBC\extract\after_normal_phas_bin", 0, st, 0.045)
    #     vot_sp.append(v)
    #     print("frame", st)
    #     print("Start ", vot_sp)

    bitr = "orig"
    ampl = 1

    # count_of_frames = read_video(PATH_VIDEO, "D:/phase_wm_graps/BBC/frames_orig_video", total_count)
    # embed(ampl, total_count, 0)
    for bitr in [25, 10, 5]:
        generate_video(bitr, output_folder)
        stop_kadr1 = []
        stop_kadr1_bin = []
        stop_kadr2_bin = []
        print('GEN')
        path_extract_code = extract(rand_k, 0.045)
        print("all")
        print("brate = ", bitr, "random frame = ", rand_k, alfa, "current percent", stop_kadr1)
    hand_made = [0, 118, 404, 414, 524, 1002, 1391, 1492, 1972, 2393, 2466, total_count]
    exit_list = []
    res = np.zeros((65, 65))
    res_bin = np.zeros((65, 65))

    print(alfa, "current percent", stop_kadr1)
    # print("bitrate", bitr, ampl, teta, alfa, "current percent", stop_kadr1_bin)
    # print("bitrate", bitr, ampl, teta, alfa, "current percent", stop_kadr2_bin)

    # teta += 0.1

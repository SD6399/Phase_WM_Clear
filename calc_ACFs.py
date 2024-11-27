import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from skimage import io

SIZE = 2048
SIZE_HALF = int(SIZE / 2)
mean_of_all = 72.6219124403117


def avg(lst):
    return sum(lst) / len(lst)


def read_video(path, coord_x, coord_y):
    vidcap = cv2.VideoCapture(path)
    count = 0
    list00 = []
    dsp00 = []
    temp00 = []

    success = True
    while success:
        success, image = vidcap.read()
        if success:
            p00 = None
            if count != 0:
                temp00.append(image[coord_x, coord_y, 0] * p00)
            p00 = int(image[coord_x, coord_y, 0])
            list00.append(p00)
            dsp00.append(p00 * p00)

        print("записан кадр", count)

        if cv2.waitKey(10) == 27:
            break
        count += 1
    mog00 = avg(list00)

    avg00_2 = avg(dsp00)
    av_2_00 = avg(temp00)
    print("MO", mog00)
    print("MO^2", avg00_2, )
    print("Temporary", av_2_00, )
    print("Variance", avg00_2 - mog00 * mog00)
    print("ACF", av_2_00 - mog00 * mog00)

    return mog00, avg00_2 - mog00 * mog00, av_2_00 - mog00 * mog00


def bracket_1():
    p = 0.01
    alf = 0.0016
    betta = 0.01
    i, j = np.indices((1920, 1080))
    r = np.sqrt(i ** 2 + j ** 2)
    new_matr = (p * np.exp(-alf * r)) + ((1 - p) * np.exp(-betta * r))

    return new_matr


def bracket_2():
    p = 0.01
    alf = 0.0016
    betta = 0.01
    i = np.indices((3000, 1))
    new_matr = (p * math.e ** (-alf * i[0, :, 0]) + (1 - p) * math.e ** (-betta * i[0, :, 0]))
    new_matr = np.ravel(new_matr)

    return new_matr


def DFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(e, x)

    return X


# ACF(x, y) = ∑[I(i, j) * I(i + x, j + y)]
def ACF_by_periodogram(lst):
    lst_pix = np.array(lst)

    # fft = np.fft.fft2(image_array)
    fft = np.fft.fft(lst_pix)
    # print("DFT [0][0]",fft[0][0])
    # fft[0]=0
    # fft[0][0]=0
    abs_fft = np.power(np.abs(fft), 2)
    ifft = np.fft.ifft(abs_fft)
    ifft = np.abs(ifft) / fft.size

    return ifft


def ACF_by_periodogram2(lst):
    lst_pix = np.array(lst)
    fft = np.fft.fft2(lst_pix)
    print("DFT[0][0] ", fft[0][0])
    # fft[0]=0
    # fft[0][0]=0
    abs_fft = np.power(np.abs(fft), 2)
    print("DFT abs[0][0] ", abs_fft[0][0])
    ifft = np.fft.ifft2(abs_fft)

    ifft = np.abs(ifft) / fft.size

    return ifft


def signal_by_ACF(ACF):
    ACF *= ACF.size
    fft = np.fft.fft(ACF)
    sqrt_fft = np.sqrt(np.abs(fft))

    ifft = np.fft.ifft(sqrt_fft)
    ifft = np.abs(ifft)

    return ifft


def calc_ACF2(p, betta, alf, x, y):
    # p = 0.001
    # alf = 0.0066
    # betta = 0.072
    # r = math.sqrt(coord_x * coord_x + coord_y * coord_y)

    # R = (p * math.e ** (-alf * r) + (1 - p) * math.e ** (-betta * r)) * \
    R = (p * math.e ** (-alf * math.sqrt(x ** 2 + y ** 2)) + (1 - p) * math.e ** (-betta * math.sqrt(x ** 2 + y ** 2)))

    return R


def calc_ACF(p, betta, alf, numb_frame):
    # p = 0.001
    # alf = 0.0066
    # betta = 0.072
    # r = math.sqrt(coord_x * coord_x + coord_y * coord_y)

    # R = (p * math.e ** (-alf * r) + (1 - p) * math.e ** (-betta * r)) * \
    R = (p * math.e ** (-alf * numb_frame) + (1 - p) * math.e ** (-betta * numb_frame))

    return R


def plot_ACF(img):
    mean_by_mean = np.mean(img[:, :])
    # print(img.shape, len(pair_lst), mean_by_mean)

    ifft_matr = ACF_by_periodogram2(img[:, :])

    check_row = (ifft_matr[0, :])
    check_row -= mean_by_mean ** 2

    return check_row


def plot_ACF_video(path_video, list_random_pixels, top_treshold):
    matr_time = np.zeros((len(list_random_pixels), top_treshold))

    count = 0
    vidcap = cv2.VideoCapture(path_video)
    # matr_time = np.zeros(( vidcap.read()[1].shape[0], 2048))

    success = True
    while success and count < top_treshold:
        success, image = vidcap.read()

        if count == 1:
            print("shape image", image.shape)

        image = image[:1080, :1920, 0]
        # image = image[300:1080, 300:1920, 0]
        print(count, np.mean(image[:, :]), np.var(image[:, :]), top_treshold)
        if count == 0:
            print(image.shape)
        for i in range(len(list_random_pixels)):
            matr_time[i, count] = image[list_random_pixels[i][0], list_random_pixels[i][1]]
        count += 1

    mean_by_mean = np.mean(matr_time)

    print("MBM", mean_by_mean)
    # ifft_matr = np.zeros(matr_time.shape)
    ifft_matr = np.zeros((len(list_random_pixels), top_treshold))
    for i in range(matr_time.shape[0]):
        # print(len(matr_time[i, :]))
        matr_time[i, 0] = 0
        ifft_matr[i, :] = ACF_by_periodogram(matr_time[i, :])
        ifft_matr[i, :] = ifft_matr[i, :] - np.mean(matr_time[i, :]) ** 2
        # ifft_matr[i, :] /= np.max(ifft_matr[i, :])

    mean_ifft = np.mean(ifft_matr, axis=0)

    print("ACF", mean_ifft[:10])

    return list(mean_ifft)


def reconstruct_signal(acf):
    N = len(acf) // 2  # Длина исходного сигнала
    R = np.zeros((N, N))  # Матрица автокорреляционных коэффициентов

    # Заполнение матрицы автокорреляционных коэффициентов
    for i in range(N):
        for j in range(N):
            R[i, j] = acf[N - 1 + abs(i - j)]

    # Вычисление вектора автокорреляции
    r = acf[N:2 * N]

    # Вычисление коэффициентов регрессии
    a = np.linalg.inv(R) @ r

    # Восстановление сигнала
    reconstructed_signal = np.zeros(N)
    for i in range(N):
        for j in range(N - i):
            reconstructed_signal[i] += a[j] * reconstructed_signal[i + j]

    return reconstructed_signal


def gener_field(list_ACF2, seed):
    var2 = np.abs(np.fft.fft2(list_ACF2))

    var2 = np.sqrt(var2)

    np.random.seed(seed)
    var1 = np.random.rand(SIZE, SIZE)

    var1 = np.fft.fft2(var1)
    un_var = var1 * var2

    final_res = np.fft.ifft2(un_var)
    print("MO of synthes", np.mean(final_res))
    final_res -= np.mean(final_res)
    new_arr = np.mod(np.real(final_res), 256)
    print(np.where(new_arr == np.min(new_arr)), np.where(new_arr == np.max(new_arr)))

    img2 = Image.fromarray(np.abs(new_arr).astype('uint8'))
    img2.save(r"D:/pythonProject/phase_wm\new_simtez_image_real_11-06-24" + str(seed) + ".png")
    mo2, varance2 = np.mean(new_arr), np.var(new_arr)
    print("MO/variance", mo2, varance2)
    print("MIMIMAX", np.min(final_res), np.max(final_res))
    print("MIMIMAX2", np.min(new_arr), np.max(new_arr))

    return np.real(final_res)


def calc_sensor_noise(img121, img122):
    # img121 = io.imread("D:/pythonProject/phase_wm/RB/frame123.png")[-350:-200, -600:-200, 0]
    # img122 = io.imread("D:/pythonProject/phase_wm/RB/frame122.png")[-350:-200, -600:-200, 0]
    cv2.imshow("Image ", io.imread(r"D:\pythonProject\phase_wm\RB/frame" + str(122) + ".png")[-350:-200, -600:-200, :])
    # print("121 ",plot_ACF(img121[:, :, 0]))
    # print("122 ",plot_ACF(img121[:, :, 0]))
    diff_img = img122.astype(int) - img121.astype(int)

    plt.plot(plot_ACF(diff_img))
    print("max and min ", np.max(diff_img), np.min(diff_img))
    plt.title("ACF of the difference image(RealBarca)")
    plt.show()

    graph_video_orig = plot_ACF_video(r"IndiDance.mp4", my_square, 2048)
    relation_list = []
    for i in range(400):
        relation_list.append(graph_video_orig[i + 1] / graph_video_orig[i])
    print(np.mean(relation_list[15:400]))
    plt.plot(relation_list)

    plt.title("B(n+1) / B(n). LutGaya")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    """
    np.random.seed(42)
    # rand_list = np.random.choice(1080, 350, replace=False)
    np.random.seed(43)
    # rand_list2 = np.random.choice(1080, 350, replace=False)

    rand_list = [i for i in range(400, 800)]
    rand_list2 = [i for i in range(600, 900)]

    pair_lst = []
    for ii in rand_list:
        for jj in rand_list2:
            pair_lst.append([ii, jj])
            pair_lst.append([jj, ii])
    my_square = pair_lst

    video_orig_acf = plot_ACF_video(r"D:/pythonProject/phase_wm/IndiDance.mp4", my_square, 2048)
    print(video_orig_acf[0:50])

    plt.plot(video_orig_acf[:200], label="Отношение АКФ соседних кадров")
    plt.legend()
    plt.show()

    relation_list_acf_video = []
    for i in range(50):
        # if 0.5 < (graph_video_orig[i + 1] / graph_video_orig[i]) < 1.5:
        relation_list_acf_video.append(video_orig_acf[i + 1] / video_orig_acf[i])
    print(relation_list_acf_video[:50])
    print(np.mean(relation_list_acf_video[25:50]))

    plt.plot(relation_list_acf_video[0:50], label="Отношение АКФ соседних кадров")
    plt.plot([np.mean(relation_list_acf_video[25:50])] * 50)
    # plt.plot([0.9242108257512562] * ind_tresh)
    plt.xlabel("The number of frame", fontsize=20)
    plt.ylabel("The ratio of adjacent ACF counts", fontsize=20)
    plt.title("LutGaya", fontsize=20)
    plt.grid(True)
    plt.xticks(np.arange(0, 51 + 1, 5))
    plt.show()
    # graph_video_model = plot_ACF_video(r"D:/pythonProject/phase_wm/Road.mp4", my_square, 2048)
    """
    # plt.plot(graph_video_model[:200], label="Synthesis video ACF")
    # plt.show()

    """
    d = [0, 121, 196, 404, 414, 772, 1418, 2363]
    full_ACF = np.zeros((len(d), 1920))
    for cnt in range(len(d)):
        image_orig = io.imread("D:/pythonProject/phase_wm/RB/frame" + str(d[cnt]) + ".png")[:, :, 0]
        # image_orig = io.imread("D:/pythonProject/phase_wm/mosaics/mosaic" + str(cnt) + ".png")
        full_ACF[cnt, :] = plot_ACF(image_orig)

    avg_ACF = np.mean(full_ACF, axis=0)
    # avg_ACF  = plot_ACF(io.imread("D:\pythonProject\phase_wm/frames_orig_video/frame111"+ ".png")[:, :, 0])
    plt.plot(avg_ACF[:100])
    plt.xlabel("The ACF argument", fontsize=18)
    plt.ylabel("ACF Value", fontsize=18)
    plt.title("ACF of the average 'LutGaya' frame", fontsize=18)
    plt.grid(True)
    plt.show()

    relation_list_ACF = []
    for ind in range(100):
        relation_list_ACF.append(avg_ACF[ind + 1] / avg_ACF[ind])

    print("cCor coef", np.mean(relation_list_ACF[0:20]))

    print("Cor coef texture", np.mean(relation_list_ACF[50:100]))
    print("relation lsit", relation_list_ACF)
    plt.plot(relation_list_ACF, )
    plt.plot([np.mean(relation_list_ACF[30:100])] * 100)
    plt.yticks(np.arange(0.96, 1.000001, 0.005))
    plt.xlabel("The ACF argument", fontsize=20)
    plt.ylabel("R=B(x+1)/B(x)", fontsize=20)
    plt.title("LutGaya", fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    """
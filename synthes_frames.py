import math
from multiprocessing import Pool
import multiprocessing
import numpy as np
from skimage import io
from PIL import Image
import os
import re
import cv2
from calc_ACFs import gener_field, calc_ACF2, SIZE, SIZE_HALF, plot_ACF
from model_of_moving import FieldGenerator
import matplotlib.pyplot as plt
from helper_methods import disp

hc_const = 6100
alf = 0.995
ro = 0.959
var_disp = 500

per_of_jumps = int(1 / (1 - alf))

count_jump = math.ceil(500 / 25)

rand_jump = np.random.choice(range(500), count_jump, replace=False)
rand_jump = np.append(rand_jump, 0)


# rand_jump = [0, 36, 77, 82, 120, 136, 184, 243, 278, 285, 290, 291, 308, 345, 348, 365, 375, 394, 403, 467]
# rand_jump = [ 11,  16,  70,  79, 100, 120, 190, 212, 218, 224, 229, 254, 256, 272, 315, 363, 433, 458, 460, 471]

def generate_video(path):
    image_folder = path  # make sure to use your folder
    video_name = "47change_ro=" + str(ro) + "_" + str(var_disp) + ".mp4"
    os.chdir(path)

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".png")]
    sort_name_img = sort_spis(images, "need_sum")
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, -1, 29.97, (width, height))

    cnt = 0
    for image in sort_name_img:
        if cnt % 100 == 0:
            print(cnt)
        video.write(cv2.imread(os.path.join(image_folder, image)))
        cnt += 1
    cv2.destroyAllWindows()
    video.release()


def sort_spis(sp, word):
    spp = []
    spb = []
    res = []
    for i in sp:
        spp.append("".join(re.findall(r'\d', i)))
        spb.append(word)
    result = [int(item) for item in spp]
    result.sort()

    result1 = [str(item) for item in result]
    for k in range(len(sp)):
        res.append(spb[k] + result1[k] + ".png")
    return res


def add_noise(img, mean, var, seed):
    row, col = img.shape
    img = img.astype(float)
    sigma = var ** 0.5
    np.random.seed(seed)
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)

    # print("Gauss max and min",np.max(gauss), np.min(gauss))
    new_img = np.where(img + gauss > 255, 255, np.where(img + gauss < 0, 0, img + gauss))
    # new_img = img + gauss

    return new_img


def process_range(first_proc, finish_proc, queue_of_proc, ACF_quadr):
    results = []
    for i in range(first_proc, finish_proc):
        sq = gener_field(ACF_quadr, rand_jump[i])
        synth_mosaic = FieldGenerator.draw_mosaic_field(20, ro, 1080, 1920, 0, var_disp, i)
        synth_mosaic_aft_noise = add_noise(synth_mosaic, 0, 49, 42 * i)
        synth_texture_aft_noise = add_noise(sq, 0, 100, rand_jump[i])
        synth_final_frame = np.where(synth_texture_aft_noise[:1080, :1920] + synth_mosaic_aft_noise[:1080, :1920] > 255,
                                     255,
                                     np.where(synth_texture_aft_noise[:1080, :1920] + synth_mosaic_aft_noise[:1080,
                                                                                      :1920] < 0, 0,
                                              synth_texture_aft_noise[:1080, :1920] + synth_mosaic_aft_noise[:1080,
                                                                                      :1920]))
        img = Image.fromarray(synth_final_frame.astype('uint8'))
        img.save(r"D:/pythonProject/phase_wm/sum_mosaic" + str(rand_jump[i]) + ".png")
        results.append(final_frame)
    queue_of_proc.put(results)


def func_sum_noise(folder_to_img, fold_to_save):
    list_of_change_frame = []
    for file in os.listdir(folder_to_img):
        if file.endswith(".png"):
            if "sum_mosaic" in file:
                digit_indices = [file[index] for index, char in enumerate(file) if char.isdigit()]
                result_string = ''.join(digit_indices)
                list_of_change_frame.append(int(result_string))
    sort_list_img = (np.sort(list_of_change_frame))
    print(sort_list_img)
    cnt = 0
    for ind in sort_list_img[:-1]:
        img = io.imread(folder_to_img + "/sum_mosaic" + str(ind) + ".png")
        print(sort_list_img[list(sort_list_img).index(ind) + 1])

        while cnt < sort_list_img[list(sort_list_img).index(ind) + 1]:
            texture_with_noise = add_noise(img, 0, 100, cnt)

            img = Image.fromarray(texture_with_noise.astype('uint8'))
            img.save(fold_to_save + "/need_sum" + str(cnt) + ".png")
            print(cnt, ind)
            cnt += 1

            # cnt += 1
            # img = io.imread(file)
            # texture_aft_noise = add_noise(img, 0, 100, cnt)


# # need_params2 = (0.5, 0.01, 0.014)
# # need_params2 = (0.5, 0.3, 0.1) # - works correctly 25.09 at morning and was so good graphic by spatial

# need_params2 = (0.5, 0.1, 0.32)  # - RealBarca
need_params2 = (0.5, 0.1, 0.1)  # - Road


# need_params3 = (0.5, 0.004, 0.05)
# need_params4 = (0.5, 0.004, 0.1)
# need_params5 = (0.5, 0.005, 0.04)
# need_params6 = (0.5, 0.09, 0.007)  # - LG


def synt_tex(need_param):
    list_ACF2 = np.zeros((SIZE, SIZE))
    tmp_matr = np.zeros((SIZE_HALF, SIZE_HALF))
    for x in range(0, SIZE_HALF):
        for y in range(0, SIZE_HALF):
            tmp_matr[x][y] = (hc_const * calc_ACF2(need_param[0], need_param[1], need_param[2], x, y))

    list_ACF2[SIZE_HALF:, SIZE_HALF:] = tmp_matr[:SIZE_HALF, :SIZE_HALF]
    # for x in range(0, 64):
    #     for y in range(0, 64):
    for x in range(SIZE_HALF):
        for y in range(SIZE_HALF):
            list_ACF2[SIZE_HALF - x, SIZE_HALF - y] = tmp_matr[x, y]
            list_ACF2[SIZE_HALF + x, SIZE_HALF - y] = tmp_matr[x, y]
            list_ACF2[SIZE_HALF - x, SIZE_HALF + y] = tmp_matr[x, y]

    # for i in range(SIZE):
    #     list_ACF2[0][i] = list_ACF2[i][0]= 64

    list_ACF2 = np.fft.fftshift(list_ACF2)
    return list_ACF2


# # rand_jump = np.random.randint(3000, size=int(3000 / per_of_jumps))
# rand_jump=np.sort(rand_jump)
#
# print(rand_jump)
# #
# image_orig = io.imread("D:/pythonProject/phase_wm/LG/frame0.png")[:, :, 0]
# d = [0, 121, 196, 404, 414, 772, 1418, 2363]
d = [i for i in range(100, 2000, 200)]
full_ACF = np.zeros((len(d), 1920))
for numb in range(len(d)):
    image_orig = io.imread("D:/pythonProject/phase_wm/Road/frame" + str(d[numb]) + ".png")[:, :, 0]
    # image_orig = io.imread("D:/pythonProject/phase_wm/mosaics/mosaic" + str(numb) + ".png")
    full_ACF[numb, :] = plot_ACF(image_orig)
    # graph_orig = plot_ACF(image_orig)
    # plt.plot(graph_orig[:200], label=d[numb])

# graph_orig = plot_ACF(io.imread("D:/pythonProject/phase_wm/Road/frame" + str(100) + ".png")[:, :, 0])
# plt.legend()
# plt.show()
graph_orig = np.mean(full_ACF, axis=0)

i = 42
# mosaic = FieldGenerator.draw_mosaic_field(20, 0.9953, 1080, 1920, 0, 1920, i)  # - RB
# mosaic = FieldGenerator.draw_mosaic_field(20, 0.9955, 1080, 1920, 0, 2811, i)  # - LG
mosaic = FieldGenerator.draw_mosaic_field(20, 0.9917, 1080, 1920, 0, 1000, i)  # - Road
# mosaic = np.zeros((1080, 1920))


list_ACF_attempt_2 = synt_tex(need_params2)
# list_ACF_attempt_3 = synt_tex(need_params3)
# list_ACF_attempt_4 = synt_tex(need_params4)
# list_ACF_attempt_5 = synt_tex(need_params5)
# list_AC_attempt_F6 = synt_tex(need_params6)

texture = gener_field(list_ACF_attempt_2, i)[:1080, :1920]
acf_text = plot_ACF(texture)[:200]

# texture3 = gener_field(list_ACF3, i)[:1080, :1920]
# texture4 = gener_field(list_ACF4, i)[:1080, :1920]
# texture5 = gener_field(list_ACF5, i)[:1080, :1920]
# texture6 = gener_field(list_ACF6, i)[:1080, :1920]
print("Analys of texture", np.mean(texture), np.var(texture))
if i == 42:
    # txt = plot_ACF(texture)
    # print("VARIANCe of texture", txt[0])
    gauss_noise = np.random.normal(0, 16 ** 0.5, (1080, 1920))

    texture_aft_noise = texture + gauss_noise
    # texture_aft_noise = add_noise(texture, 0, 30, 42)
    print("Analys of texture noise", np.mean(texture_aft_noise), np.var(texture_aft_noise))
    plot_tan = plot_ACF(texture_aft_noise)

    count = 0
    all_mse = []
    params = []

    print("Analys of mosaic", np.mean(mosaic), np.var(mosaic))
    mosaic += 62

    print(mosaic.shape)
    print(np.var(mosaic))
    img1 = Image.fromarray(mosaic.astype('uint8'))
    img1.save(r"D:/pythonProject/phase_wm/mosaic" + str(i) + ".png")
    print("--------------")

    print("--------------")
    print("Noise ACF")
    # noise = plot_ACF(gauss)
    # noise2 = plot_ACF(gauss2)
    txt_acf = plot_ACF(texture)[:200]
    msc_acf = plot_ACF(mosaic)[:200]
    # msc_acf -= 200

    plt.plot(txt_acf, label="ACF of texture")
    plt.plot(msc_acf, label="ACF of mosaics")

    # mosaic_aft_noise = add_noise(mosaic, 0, 4, 42 * i)

    # gauss = np.random.normal(0, 4 ** 0.5, (1080, 1920))
    mosaic_aft_noise = mosaic + gauss_noise
    print("Analys of mosaic noise", np.mean(mosaic_aft_noise), np.var(mosaic_aft_noise))
    final_frame = texture_aft_noise[:1080, :1920] + mosaic_aft_noise[:1080, :1920]
    # final_frame = np.where(texture_aft_noise[:1080, :1920] + mosaic_aft_noise[:1080, :1920] > 255, 255,
    #                        np.where(texture_aft_noise[:1080, :1920] + mosaic_aft_noise[:1080, :1920] < 0, 0,
    #                                 texture_aft_noise[:1080, :1920] + mosaic_aft_noise[:1080, :1920]))

    txt_n_acf = plot_ACF(texture_aft_noise)[:200]
    msc_n_acf = plot_ACF(mosaic_aft_noise)[:200]

    # msc_n_acf -= 200
    plt.plot(txt_n_acf, label="ACF of texture + noise")
    # plt.plot(plot_ACF(texture3+mosaic_aft_noise)[:200] , label="Texture + noise2 need_params2 " + str(need_params3))
    # plt.plot(plot_ACF(texture4+mosaic_aft_noise)[:200] , label="Texture + noise2 need_params2 " + str(need_params4))
    # plt.plot(plot_ACF(texture5+mosaic_aft_noise)[:200] , label="Texture + noise2 need_params2 " + str(need_params5))
    # plt.plot(plot_ACF(texture6+mosaic_aft_noise)[:200] , label="Texture + noise2 need_params2 " + str(need_params6))
    plt.plot(msc_n_acf, label="ACF mosaics + noise")
    plt.plot(graph_orig[:200], label="ACF of the original image")
    # img1 = Image.fromarray(final_frame.astype('uint8'))
    # print(np.mean(final_frame), np.var(final_frame))
    # img1.save(r"D:/pythonProject/phase_wm/fold_model_video/final_img_article_LG_" + str(i) + ".png")
    final_acf = msc_n_acf + txt_n_acf
    plt.plot(final_acf[:200], label="ACF of the synthesized image")
    plt.xlabel("The ACF argument", fontsize=18)
    plt.ylabel("ACF Value", fontsize=18)
    plt.title("Road", fontsize=18)
    plt.legend()
    plt.show()
    # Текстурный шум

if __name__ == '__main__':
    mosaic = FieldGenerator.draw_mosaic_field(20, ro, 1080, 1920, 0, var_disp, 43)
    num_processes = multiprocessing.cpu_count()
    result_queue = multiprocessing.Queue()
    jobs = []

    for i in range(num_processes):
        start = i * (len(rand_jump) // num_processes)
        end = (i + 1) * (len(rand_jump) // num_processes)
        process = multiprocessing.Process(target=process_range, args=(start, end, result_queue, list_ACF_attempt_2))
        jobs.append(process)
        process.start()

    func_sum_noise(r"D:\pythonProject\phase_wm", r"D:\pythonProject\phase_wm\BBC_method_sintez")
    generate_video(r"D:\pythonProject\phase_wm\BBC_method_sintez")
# disp(r"D:\pythonProject\phase_wm\fold_model_video",word='/final_img',name_of_doc="variance_Hand_Make.csv")

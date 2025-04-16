import numpy as np
import re
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
import csv
import cv2
from PIL import Image
from scpetrcal_halftone import check_spatial2spectr


# from reedsolomon import Nbit


def read_video(path, path_to_save, final_frame):
    vidcap = cv2.VideoCapture(path)
    count_frame = 0
    success = True
    while success and (count_frame < final_frame):
        success, image = vidcap.read()

        if success:
            cv2.imwrite(path_to_save + "/frame%d.png" % count_frame, image)
        if count_frame % 500 == 499:
            print("записан кадр", count_frame, )

        if cv2.waitKey(10) == 27:
            break
        count_frame += 1
    return count_frame


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


# def img2bin(img):
#     k = 0
#
#     our_avg = np.mean(img)
#     for i in range(0, img.shape[0]):
#         for j in range(0, img.shape[1]):
#             tmp = img[i, j]
#
#             if tmp > our_avg:
#                 img[i, j] = 255
#             else:
#                 img[i, j] = 0
#
#             k += 1
#     return img


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


def compare_qr(myqr, orig_qr, shift, cnt):
    """
     Comparing the extracted QR with the original one
    :param path: path to code for comparison
    :return: percentage of similarity
    """
    size_wm = 65
    # orig_qr = io.imread(r"data/RS_cod89x89.png")
    orig_cut = np.zeros((size_wm, size_wm))
    orig_qr = np.where(orig_qr > 127, 255, 0)
    orig_cut[:, :int(size_wm / 2)] = orig_qr[1 + shift:size_wm + 1 + shift,
                                     1 + shift:np.ceil(size_wm / 2).astype(int) + shift]
    if shift != 0:
        orig_cut[:, int(size_wm / 2):] = orig_qr[1 + shift:size_wm + 1 + shift,
                                         -1 * (np.ceil(size_wm / 2)).astype(int) - shift:-shift]
    else:
        orig_cut[:, int(size_wm / 2):] = orig_qr[1 + shift:size_wm + 1 + shift,
                                         -1 * (np.ceil(size_wm / 2).astype(int)) - shift:]
    # small_qr = big2small(orig_qr)
    # sr_matr = np.zeros((1424, 1424, 3))
    # myqr = io.imread(path)

    # myqr_cut = myqr
    myqr_cut = np.zeros((size_wm, size_wm))

    myqr_cut[:, :int(size_wm / 2)] = myqr[1 + shift:size_wm + 1 + shift,
                                     1 + shift:np.ceil(size_wm / 2).astype(int) + shift]
    if shift== 0:
        myqr_cut[:, int(size_wm / 2):] = myqr[1 + shift:size_wm + 1 + shift,
                                         -1 * (np.ceil(size_wm / 2).astype(int)) - shift:]
    else:
        myqr_cut[:, int(size_wm / 2):] = myqr[1 + shift:size_wm + 1 + shift,
                                     -1 * (np.ceil(size_wm / 2).astype(int)) - shift:-shift]
    myqr_cut = np.where(myqr_cut > np.mean(myqr_cut), 255, 0)

    img = Image.fromarray(myqr_cut.astype('uint8'))
    img.save(r"D:/pythonProject/phase_wm\extract/after_normal_phas_bin/result" + str(cnt) + ".png")

    sr_matr = orig_cut == myqr_cut
    k = np.count_nonzero(sr_matr)
    return k / sr_matr.size


def binarize_qr(myqr, shift):
    """
     Comparing the extracted QR with the original one
    :param path: path to code for comparison
    :return: percentage of similarity
    """

    myqr_cut = np.zeros((49, 49))
    # for i in range(0, 230, 4):
    #     for j in range(0, 230, 4):
    #         myqr_cut[int(i / 4), int(j / 4)] = np.mean(myqr[1 + i + shift: i + shift + 4 +1, -33 + j - shift:-33+j - shift+4])

    myqr_cut[:, :24] = myqr[1 + shift:50 + shift, 1 + shift:25 + shift]
    if shift != 0:
        myqr_cut[:, 24:] = myqr[1 + shift:50 + shift, -25 - shift:-shift]
    else:
        myqr_cut[:, 24:] = myqr[1 + shift:50 + shift, -25 - shift:]
    myqr_cut = np.where(myqr_cut > np.mean(myqr_cut), 255, 0)
    myqr_cut[myqr_cut == 255] = 1

    return myqr_cut


# create_gray_bg()
# compar_before_after_saving("C:/Users/user/PycharmProjects/phase_wm/frames_after_emb", "C:/Users/user/PycharmProjects/phase_wm/extract")

# bit_voting(io.imread(r"D:\dk\university\nirs\extract/wm_after_2_smooth_bin/result" + str(456) + ".png"), 7112)
# print(disp("C:/Users/user/PycharmProjects/phase_wm/frames_orig_video"))
# csv2list('LG_disp.csv')

"""
a5 = [0.514792899408284, 0.5417751479289941, 0.9531360946745562, 0.9337278106508876, 0.7753846153846153,
      0.6596449704142012, 0.9995266272189349, 1.0, 1.0, 1.0, 0.8866272189349113, 0.9898224852071006,
      0.7162130177514793, 0.9510059171597633, 0.994792899408284, 0.9905325443786982, 0.9981065088757396,
      0.8830769230769231, 0.9995266272189349, 0.9912426035502958, 0.5394082840236687, 0.5121893491124261,
      0.5112426035502958, 0.5105325443786982, 0.5107692307692308, 0.5095857988165681, 0.5095857988165681,
      0.5095857988165681, 0.5107692307692308, 0.5100591715976331, 0.5117159763313609, 0.5100591715976331,
      0.5140828402366864, 0.5124260355029586, 0.5114792899408284, 0.5150295857988165, 0.5223668639053255,
      0.5732544378698224, 0.5644970414201184, 0.5898224852071006, 0.5910059171597634, 0.6288757396449705,
      0.6584615384615384, 0.6307692307692307, 0.6686390532544378, 0.6885207100591716, 0.6584615384615384,
      0.5848520710059172, 0.5900591715976331, 0.5460355029585798, 0.5689940828402367, 0.597396449704142,
      0.6257988165680474, 0.663905325443787, 0.7422485207100592, 0.7654437869822486, 0.7557396449704142,
      0.8511242603550296, 0.8861538461538462, 0.8989349112426035, 0.8764497041420118, 0.9119526627218935,
      0.9242603550295858, 0.9100591715976332, 0.8925443786982249, 0.9297041420118343, 0.9382248520710059,
      0.9583431952662722, 0.9453254437869822, 0.9656804733727811, 0.9808284023668639, 0.9879289940828402,
      0.9931360946745562, 0.9910059171597633, 0.9938461538461538, 0.9933727810650888, 0.9924260355029586,
      0.9936094674556213, 0.9931360946745562, 0.9966863905325444, 0.9957396449704142, 0.9966863905325444,
      0.9969230769230769, 0.994792899408284, 0.9914792899408285, 0.9914792899408285, 0.9805917159763313,
      0.9528994082840236, 0.970887573964497, 0.9730177514792899, 0.9564497041420118, 0.9528994082840236,
      0.8901775147928994, 0.9226035502958579, 0.9429585798816568, 0.9403550295857989, 0.6837869822485207,
      0.6892307692307692, 0.7332544378698225, 0.7545562130177514, 0.7737278106508876, 0.7952662721893491,
      0.8175147928994083, 0.8274556213017752, 0.8684023668639054, 0.8731360946745562, 0.9079289940828402,
      0.9157396449704142, 0.9453254437869822, 0.9457988165680473, 0.951715976331361, 0.9465088757396449,
      0.9609467455621302, 0.965207100591716, 0.9723076923076923, 0.9611834319526628, 0.965207100591716,
      0.9673372781065088, 0.9706508875739644, 0.9784615384615385, 0.9820118343195267, 0.9850887573964497,
      0.9862721893491124, 0.9933727810650888, 0.9940828402366864, 0.9981065088757396, 0.9983431952662722,
      0.9983431952662722, 0.9985798816568048, 0.9985798816568048, 0.9985798816568048, 0.9988165680473373,
      0.9988165680473373, 0.9988165680473373, 0.9995266272189349, 0.9990532544378699, 0.9992899408284024,
      0.9995266272189349, 0.9995266272189349, 0.9997633136094675, 0.9995266272189349, 0.9995266272189349,
      0.9995266272189349, 0.9997633136094675, 0.9992899408284024, 0.9992899408284024, 0.9995266272189349,
      0.9995266272189349, 0.9997633136094675, 0.9997633136094675]

a3 = [0.5081656804733727, 0.5079289940828402, 0.5460355029585798, 0.556923076923077, 0.5169230769230769,
      0.5110059171597633, 0.9559763313609467, 0.9358579881656804, 0.9753846153846154, 0.7881656804733728,
      0.5327810650887574, 0.7060355029585799, 0.5197633136094675, 0.5881656804733728, 0.7254437869822485,
      0.7105325443786982, 0.8499408284023668, 0.5782248520710059, 0.855621301775148, 0.7697041420118343,
      0.5131360946745562, 0.5102958579881657, 0.5100591715976331, 0.5093491124260355, 0.5102958579881657,
      0.5098224852071006, 0.5095857988165681, 0.5100591715976331, 0.5102958579881657, 0.5102958579881657,
      0.5114792899408284, 0.5105325443786982, 0.5143195266272189, 0.5143195266272189, 0.5128994082840237,
      0.5138461538461538, 0.5162130177514793, 0.525680473372781, 0.5261538461538462, 0.5287573964497041,
      0.5315976331360946, 0.5398816568047338, 0.5472189349112426, 0.570414201183432, 0.5642603550295858,
      0.5581065088757396, 0.5431952662721894, 0.525680473372781, 0.5259171597633137, 0.5197633136094675,
      0.5228402366863906, 0.5259171597633137, 0.5268639053254438, 0.5318343195266272, 0.5410650887573965,
      0.5427218934911242, 0.5450887573964497, 0.5727810650887574, 0.5905325443786982, 0.6089940828402367,
      0.6255621301775148, 0.6570414201183432, 0.6745562130177515, 0.6724260355029585, 0.6627218934911243,
      0.6956213017751479, 0.7133727810650887, 0.7318343195266273, 0.703905325443787, 0.7275739644970414,
      0.7765680473372781, 0.7879289940828402, 0.8527810650887574, 0.8134911242603551, 0.8660355029585799,
      0.8927810650887574, 0.882603550295858, 0.9084023668639053, 0.9048520710059171, 0.9110059171597633,
      0.874792899408284, 0.8901775147928994, 0.8942011834319527, 0.8823668639053255, 0.8267455621301775,
      0.8435502958579881, 0.8113609467455621, 0.7656804733727811, 0.7997633136094675, 0.796923076923077,
      0.7715976331360946, 0.7564497041420118, 0.6771597633136095, 0.7005917159763314, 0.7330177514792899,
      0.7566863905325444, 0.5637869822485208, 0.5635502958579882, 0.5805917159763314, 0.5850887573964497,
      0.5945562130177515, 0.6002366863905325, 0.6044970414201184, 0.6120710059171598, 0.6248520710059171,
      0.6423668639053255, 0.674792899408284, 0.6792899408284023, 0.7285207100591716, 0.7218934911242604,
      0.7555029585798817, 0.7394082840236686, 0.7706508875739645, 0.7713609467455621, 0.7810650887573964,
      0.7396449704142012, 0.7507692307692307, 0.7512426035502958, 0.7647337278106509, 0.7801183431952663,
      0.7978698224852071, 0.8165680473372781, 0.803076923076923, 0.8563313609467456, 0.8700591715976331,
      0.9188165680473372, 0.9280473372781065, 0.9401183431952663, 0.9562130177514793, 0.9585798816568047,
      0.963076923076923, 0.9701775147928994, 0.9784615384615385, 0.9820118343195267, 0.9872189349112426,
      0.9881656804733728, 0.9853254437869823, 0.9848520710059172, 0.978224852071006, 0.9711242603550296,
      0.9706508875739644, 0.9661538461538461, 0.9661538461538461, 0.965207100591716, 0.9611834319526628,
      0.9730177514792899, 0.9585798816568047, 0.9715976331360947, 0.9618934911242604, 0.9706508875739644]

a2 = [0.5081656804733727, 0.5079289940828402, 0.5268639053254438, 0.5491124260355029, 0.5110059171597633,
      0.5093491124260355, 0.7640236686390532, 0.736094674556213, 0.7150295857988166, 0.5195266272189349,
      0.5192899408284024, 0.6804733727810651, 0.5216568047337278, 0.573491124260355, 0.693491124260355,
      0.7031952662721893, 0.8009467455621302, 0.6345562130177514, 0.8196449704142011, 0.8066272189349113,
      0.5102958579881657, 0.5100591715976331, 0.5098224852071006, 0.509112426035503, 0.5088757396449705,
      0.5086390532544379, 0.5088757396449705, 0.509112426035503, 0.5095857988165681, 0.5098224852071006,
      0.5100591715976331, 0.5100591715976331, 0.5112426035502958, 0.5124260355029586, 0.5128994082840237,
      0.5126627218934912, 0.5126627218934912, 0.5131360946745562, 0.5131360946745562, 0.5136094674556213,
      0.5140828402366864, 0.5143195266272189, 0.5157396449704142, 0.5209467455621302, 0.5204733727810651,
      0.5271005917159763, 0.5228402366863906, 0.5152662721893491, 0.5162130177514793, 0.5133727810650888,
      0.5133727810650888, 0.5138461538461538, 0.5145562130177515, 0.5159763313609468, 0.517396449704142,
      0.5176331360946745, 0.5185798816568047, 0.5249704142011834, 0.5294674556213018, 0.5358579881656804,
      0.5469822485207101, 0.5514792899408284, 0.5588165680473373, 0.5649704142011834, 0.5699408284023668,
      0.5751479289940828, 0.5836686390532544, 0.5926627218934911, 0.5782248520710059, 0.5971597633136094,
      0.6220118343195267, 0.6253254437869823, 0.6471005917159763, 0.642130177514793, 0.6695857988165681,
      0.7015384615384616, 0.698698224852071, 0.7237869822485207, 0.7297041420118343, 0.7128994082840237,
      0.6814201183431953, 0.7003550295857989, 0.7001183431952662, 0.6958579881656805, 0.6444970414201183,
      0.6520710059171597, 0.6397633136094675, 0.6255621301775148, 0.6492307692307693, 0.6463905325443787,
      0.6359763313609468, 0.6257988165680474, 0.589585798816568, 0.6018934911242604, 0.615621301775148,
      0.6324260355029586, 0.5346745562130177, 0.5396449704142012, 0.5346745562130177, 0.5519526627218935,
      0.5538461538461539, 0.5571597633136095, 0.5557396449704142, 0.5592899408284023, 0.5689940828402367,
      0.5727810650887574, 0.5822485207100592, 0.5843786982248521, 0.610887573964497, 0.6049704142011835,
      0.618224852071006, 0.6130177514792899, 0.62698224852071, 0.6362130177514793, 0.62698224852071, 0.6115976331360947,
      0.6170414201183432, 0.6196449704142012, 0.6229585798816568, 0.6307692307692307, 0.6352662721893491,
      0.6433136094674556, 0.6449704142011834, 0.6710059171597633, 0.6802366863905326, 0.7602366863905325,
      0.7988165680473372, 0.8208284023668639, 0.84, 0.829112426035503, 0.8345562130177515, 0.8473372781065088,
      0.8847337278106508, 0.8942011834319527, 0.9062721893491125, 0.9081656804733728, 0.8989349112426035,
      0.8958579881656805, 0.9057988165680473, 0.895621301775148, 0.8996449704142012, 0.8970414201183432,
      0.89301775147929, 0.8989349112426035, 0.8960946745562131, 0.8998816568047338, 0.8532544378698225,
      0.9140828402366864, 0.8759763313609468, 0.8743195266272189]

a1 = [0.5081656804733727, 0.5079289940828402, 0.5218934911242603, 0.5401183431952663, 0.5084023668639053,
      0.5081656804733727, 0.5342011834319527, 0.5938461538461538, 0.6684023668639053, 0.5086390532544379,
      0.5164497041420119, 0.5817751479289941, 0.5287573964497041, 0.527810650887574, 0.5171597633136095,
      0.5680473372781065, 0.5749112426035503, 0.5105325443786982, 0.5207100591715976, 0.6044970414201184,
      0.5088757396449705, 0.5088757396449705, 0.5086390532544379, 0.5084023668639053, 0.5084023668639053,
      0.5081656804733727, 0.5084023668639053, 0.5084023668639053, 0.5086390532544379, 0.5081656804733727,
      0.5086390532544379, 0.5088757396449705, 0.5095857988165681, 0.5098224852071006, 0.5095857988165681,
      0.5095857988165681, 0.5095857988165681, 0.5100591715976331, 0.5100591715976331, 0.5100591715976331,
      0.5107692307692308, 0.5119526627218934, 0.5119526627218934, 0.5124260355029586, 0.5119526627218934,
      0.5107692307692308, 0.5110059171597633, 0.5105325443786982, 0.5112426035502958, 0.5107692307692308,
      0.5105325443786982, 0.5102958579881657, 0.5105325443786982, 0.5110059171597633, 0.5112426035502958,
      0.5114792899408284, 0.5110059171597633, 0.5110059171597633, 0.5114792899408284, 0.5124260355029586,
      0.5126627218934912, 0.5136094674556213, 0.5145562130177515, 0.5138461538461538, 0.5169230769230769,
      0.5176331360946745, 0.5181065088757396, 0.5138461538461538, 0.5126627218934912, 0.5289940828402366,
      0.5337278106508876, 0.5372781065088758, 0.5294674556213018, 0.5275739644970414, 0.5195266272189349,
      0.5228402366863906, 0.5275739644970414, 0.5299408284023669, 0.530414201183432, 0.5356213017751479,
      0.5453254437869822, 0.5519526627218935, 0.5588165680473373, 0.5441420118343195, 0.5396449704142012,
      0.5431952662721894, 0.5446153846153846, 0.541301775147929, 0.5453254437869822, 0.5479289940828402,
      0.5446153846153846, 0.5457988165680473, 0.5332544378698225, 0.5377514792899408, 0.5405917159763314,
      0.5403550295857988, 0.5164497041420119, 0.5124260355029586, 0.5124260355029586, 0.5249704142011834,
      0.5268639053254438, 0.5266272189349113, 0.5133727810650888, 0.5214201183431952, 0.5299408284023669,
      0.5315976331360946, 0.5195266272189349, 0.5249704142011834, 0.5327810650887574, 0.5315976331360946, 0.52,
      0.5133727810650888, 0.5320710059171597, 0.538698224852071, 0.5342011834319527, 0.5318343195266272,
      0.5353846153846153, 0.5344378698224852, 0.5351479289940828, 0.5351479289940828, 0.5346745562130177,
      0.5379881656804734, 0.5372781065088758, 0.5441420118343195, 0.5457988165680473, 0.5396449704142012,
      0.5604733727810651, 0.5739644970414202, 0.5879289940828403, 0.5829585798816568, 0.5843786982248521,
      0.5758579881656805, 0.5765680473372781, 0.5798816568047337, 0.5992899408284024, 0.6205917159763313,
      0.6149112426035503, 0.6210650887573964, 0.6142011834319526, 0.6127810650887574, 0.6104142011834319,
      0.6125443786982249, 0.6094674556213018, 0.6328994082840237, 0.6115976331360947, 0.6272189349112426,
      0.6364497041420119, 0.6437869822485207, 0.6444970414201183, 0.5713609467455621]

plt.plot([i for i in range(19, 3000, 20)], a5, label="A = 5")
plt.plot([i for i in range(19, 3000, 20)], a3, label="A = 3")
plt.plot([i for i in range(19, 3000, 20)], a2, label="A = 2")
plt.plot([i for i in range(19, 3000, 20)], a1, label="A = 1")
plt.legend()
plt.grid(True)
plt.show()
"""

# import os
# video_name = 'test_video_after_norn.mp4'
# image_folder = r"D:\pythonProject\phase_wm\extract\after_normal_phas"
# os.chdir(image_folder)
#
# images = [img for img in os.listdir(image_folder)
#           if img.endswith(".png")]
# sort_name_img = sort_spis(images, "result")
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape
# # fourcc = cv2.VideoWriter_fourcc(*'H264')
#
# video = cv2.VideoWriter(video_name, 0, 50, (width, height))
#
# cnt = 0
# for image in sort_name_img:
#     # if cnt % 300 == 0:
#
#     video.write(cv2.imread(os.path.join(image_folder, image)))
#     if cnt % 799 == 0:
#         print(cnt)
#     cnt += 1
# cv2.destroyAllWindows()
# video.release()

"""
list_v = [3]
for i in list_v:
    my_file1_acc = open(r'D:/pythonProject/Phase_WM_Clear\data/acc_list_49_1024_no_smooth_union_on_%d_center_' % 10 + str(
            3) + '_bitr' + str(
            5) + "_shift" + str(10) + '.txt', "r")
    # my_file1_var = open("data/var_list_no_smooth_union_on_%d_center_" % 0 + str(2) + '_bitr' + str(10) + '.txt', "r")
    my_file2_acc = open(r'D:/pythonProject/Phase_WM_Clear\data/acc_list_49_1024_no_smooth_union_on_%d_center_' % 0 + str(
        3) + '_bitr' + str(
        5) + "_shift" + str(0) + '.txt', "r")
    # my_file2_var = open("data/var_list_2_bitr20.txt", "r")
    # reading the file
    # data_var = my_file1_var.read()
    #
    # # replacing end of line('/n') with ' ' and
    # # splitting the text it further when '.' is seen.
    # data_into_list_var = data_var.replace('\n', ' ').split(" ")
    # list_split_var = np.array([float(words) for segments in data_into_list_var for words in segments.split()])
    # list_split_var[list_split_var > 1000] = 1000
    # list_split_var /= 1000
    # plt.plot(list_split_var, label="Variance. Embedding in all picture")

    # data_var2 = my_file2_var.read()
    # #
    # # # replacing end of line('/n') with ' ' and
    # # # splitting the text it further when '.' is seen.
    # data_into_list_var2 = data_var2.replace('\n', ' ').split(" ")
    # list_split_var2 = np.array([float(words) for segments in data_into_list_var2 for words in segments.split()])
    # list_split_var2[list_split_var2 > 1000] = 1000
    # list_split_var2 /= 1000
    # plt.plot(list_split_var2, label="Variance. Embedding in corner")

    data1_acc = my_file1_acc.read()
    data_into_list1_acc = data1_acc.replace('\n', ' ').split(" ")
    list_split1_acc = np.array([float(words) for segments in data_into_list1_acc for words in segments.split()])
    #
    data2_acc = my_file2_acc.read()
    data_into_list2_acc = data2_acc.replace('\n', ' ').split(" ")
    list_split2_acc = np.array([float(words) for segments in data_into_list2_acc for words in segments.split()])
    # list_split2_acc[list_split2_acc > 1000] = 1000
    # print(len(list_split2_acc), len(list_split1_acc), len(range(19, 2980, 20)))
    plt.plot(range(19, 2980, 20), list_split2_acc, label="Size of QR = 49x49")
    plt.plot(range(19, 2980, 20), list_split1_acc, label="Size of QR = 65x65")
    # plt.plot(range(19, 2980, 20), list_split2_acc[:-1], label="Accuracy. Embedding in All Picture. Bitrate = 10")
    plt.title("Spectral Method. Embedding in All Picture. Bitrate = 5.A =%d" % i)
plt.legend()
plt.grid(True)
plt.show()
"""

a1 = [0.7571845064556435, 0.7592669720949604, 0.7551020408163265, 0.8575593502707205, 0.9058725531028738,
      0.9379425239483549, 0.9566847147022074, 0.9683465222823824, 0.9758433985839233, 0.978758850478967,
      0.9800083298625573, 0.9800083298625573, 0.9837567680133278, 0.9875052061640983, 0.9883381924198251,
      0.9891711786755518, 0.9908371511870054, 0.9916701374427322, 0.9920866305705955, 0.9925031236984589,
      0.9937526030820492, 0.9941690962099126, 0.9950020824656394, 0.9950020824656394, 0.9950020824656394,
      0.9950020824656394, 0.9950020824656394, 0.9950020824656394, 0.9950020824656394, 0.9950020824656394]

a2 = [0.8342357351103706, 0.8575593502707205, 0.8688046647230321, 0.9354435651811746, 0.9616826322365681,
      0.9762598917117867, 0.9845897542690546, 0.9866722199083715, 0.9891711786755518, 0.9895876718034152,
      0.9900041649312786, 0.9900041649312786, 0.9912536443148688, 0.9920866305705955, 0.9920866305705955,
      0.9933361099541858, 0.9933361099541858, 0.9950020824656394, 0.9958350687213661, 0.9979175343606831,
      0.9983340274885465, 0.9983340274885465, 0.9983340274885465, 0.9983340274885465, 0.9983340274885465,
      0.9983340274885465, 0.9983340274885465, 0.9983340274885465, 0.9983340274885465, 0.9983340274885465]

a3 = [0.8688046647230321, 0.8912952936276551, 0.9192003331945023, 0.9670970428987922, 0.9812578092461475,
      0.9858392336526447, 0.9895876718034152, 0.990420658059142, 0.9920866305705955, 0.9929196168263223,
      0.9929196168263223, 0.9929196168263223, 0.9929196168263223, 0.9937526030820492, 0.9950020824656394,
      0.9966680549770929, 0.9970845481049563, 0.9983340274885465, 0.9983340274885465, 0.9983340274885465,
      0.9983340274885465, 0.9987505206164098, 0.9987505206164098, 0.9987505206164098, 0.9987505206164098,
      0.9987505206164098, 0.9987505206164098, 0.9987505206164098, 0.9987505206164098, 0.9987505206164098]

a3_177 = [0.7763431903373594, 0.8029987505206164, 0.8842149104539775, 0.9421074552269888, 0.963765097875885,
          0.9754269054560599, 0.9816743023740109, 0.9837567680133278, 0.9866722199083715, 0.9875052061640983,
          0.9879216992919617, 0.9879216992919617, 0.9900041649312786, 0.9916701374427322, 0.9929196168263223,
          0.9937526030820492, 0.994585589337776, 0.9950020824656394, 0.9950020824656394, 0.9950020824656394,
          0.9966680549770929, 0.9970845481049563, 0.9975010412328197, 0.9975010412328197, 0.9975010412328197,
          0.9975010412328197, 0.9975010412328197, 0.9975010412328197, 0.9975010412328197, 0.9975010412328197]

a2_177 = [0.7259475218658892, 0.7392753019575177, 0.8096626405664307, 0.8679716784673053, 0.8962932111620159,
          0.9167013744273219, 0.9321116201582674, 0.9429404414827155, 0.9541857559350271, 0.9645980841316119,
          0.9650145772594753, 0.9650145772594753, 0.9687630154102457, 0.9725114535610162, 0.9766763848396501,
          0.9783423573511038, 0.9791753436068305, 0.9804248229904207, 0.9812578092461475, 0.9837567680133278,
          0.985006247396918, 0.9854227405247813, 0.9858392336526447, 0.9858392336526447, 0.9858392336526447,
          0.9858392336526447, 0.9858392336526447, 0.9858392336526447, 0.9858392336526447, 0.9858392336526447]

a1_177 = [0.6588921282798834, 0.6463973344439816, 0.6601416076634735, 0.7113702623906706, 0.7476051645147855,
          0.7896709704289879, 0.8179925031236984, 0.8400666389004582, 0.8554768846314036, 0.870887130362349,
          0.8717201166180758, 0.8717201166180758, 0.8821324448146606, 0.8937942523948355, 0.9037900874635568,
          0.9117034568929613, 0.9167013744273219, 0.9216992919616827, 0.9283631820074969, 0.9325281132861308,
          0.9379425239483549, 0.9400249895876718, 0.941274468971262, 0.941274468971262, 0.941274468971262,
          0.941274468971262, 0.941274468971262, 0.941274468971262, 0.941274468971262, 0.941274468971262]

# plt.plot(range(10, 308, 10), a1, label="A = 1", )
# plt.plot(range(10, 308, 10), a2, label="A = 2", )
# plt.plot(range(10, 308, 10), a3, label="A = 3", )
# plt.legend(fontsize=20)
# plt.grid(True)
# plt.xlabel("Номер кадра", fontsize=20)
# plt.ylabel("Процент корректно извлеченных битов", fontsize=20)
# plt.title("RealBarca", fontsize=20)
# plt.show()
from scpetrcal_halftone import energy_spector


def check_energ_spector():
    orig_wave = io.imread(r"D:\pythonProject/Phase_WM_Clear/data/spatial_spectr_wm_65.png").astype(int)
    dif_val = []
    qr_acc = []
    cnt = 1
    extr_wave = io.imread("D:/pythonProject/phase_wm/extract/after_normal_phas/result" + str(2950) + ".png").astype(int)
    for i in range(2951, 3000, 1):
        extr_wave += io.imread("D:/pythonProject/phase_wm/extract/after_normal_phas/result" + str(i) + ".png").astype(
            int)
        cnt += 1
        # extr_wave1 = io.imread("D:/pythonProject/phase_wm/extract/after_normal_phas/result" + str(i - 1) + ".png").astype(
        #     int)

        # spector = check_spatial2spectr(extr_wave)
        # qr_acc.append(compare_qr(
        #     spector, io.imread("data/check_ifft_wm.png")))
        # dif_val.append(np.var(dif_img))

    dif_img = 128 + orig_wave - extr_wave / cnt

    img2 = Image.fromarray(dif_img.astype('uint8'))
    img2.save(r"data/diff_extract_img.png")

    energ = energy_spector(dif_img)

    img2 = Image.fromarray(energ.astype('uint8'))
    img2.save(r"data/energ_spector_diff_img.png")
    dif_val = np.array(dif_val)
    dif_val /= np.max(dif_val)
    # plt.plot(dif_val)
    # plt.plot(qr_acc)
    # plt.show()

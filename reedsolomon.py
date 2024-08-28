import numpy as np
from reedsolo import RSCodec
from PIL import Image
from helper_methods import small2big, big2small
from collections import Counter


def create_RS_code(mes, nsym, nsize, len_side, output_path):
    """
    Convert text message into square binary image by RS-code
    :param mes: Message for coding
    :param nsym: number of ecc symbols
    :param nsize: length of each chunk
    :param len_side: The side of the square representation of the Reed Solomon code
    :param output_path: path for image saving
    :return: len of binary sequence
    """

    rsc = RSCodec(nsym=nsym, nsize=nsize)  # creation RS-coder
    cod_mes = rsc.encode(mes)  # bytearray of coding message

    num_elem_cod = [cod_mes[i] for i in range(len(cod_mes))]
    print(num_elem_cod)
    binary_cod_mes = [bin(byte)[2:] for byte in num_elem_cod]

    len_bin = 0  # length of final RS-code
    binstr = np.array([])
    for i in range(len(binary_cod_mes)):
        while len(binary_cod_mes[i]) < 8:  # each character is padded with up to 8 bit characters by '0'
            binary_cod_mes[i] = '0' + binary_cod_mes[i]
        len_bin += len(binary_cod_mes[i])  # increment len
        binstr = np.append(binstr, list(binary_cod_mes[i]))  # all RS-code in one Numpy

    binstr = binstr.astype(int)
    while len(binstr) < len_side * len_side:
        binstr = np.append(binstr, 0)  # expanded to match the size of the square code

    print(len_bin)
    print(len(mes))
    small_matrix = np.resize(binstr, (len_side, len_side))  # to square
    matr4embed = small2big(small_matrix)  # to 1424x1424
    matr4embed[matr4embed == 1] = 255
    img = Image.fromarray(matr4embed.astype('uint8'))
    img.convert('RGB').save(output_path)

    return len_bin


def extract_RS(image_RS, nsym, nsize, count):
    """
    Decode binary image in a text
    :param image_RS: output binary image
    :param nsym: number of ecc symbols
    :param nsize: length of each chunk
    :param count: length of embedded sequence
    :return: decoding text
    """

    rsc = RSCodec(nsym=nsym, nsize=nsize)
    side_code = image_RS.shape[0]
    final_extract = b""
    # matrbin = big2small(image_RS)  # convert to small image
    matrbin = np.copy(image_RS)
    matrbin[matrbin == 255] = 1
    listbin = np.reshape(matrbin, (side_code * side_code)).astype(int)  # square to sequence
    data_length = len(listbin) - (side_code * side_code - count)  # required sequence length
    mas_symb = []
    for i in range(0, data_length, 8):  # binary sequence to sequence of symbols
        tmp = ''.join(str(x) for x in listbin[i:i + 8])
        ch = bytes([(int(tmp, 2))])
        mas_symb.append(ch)

    for i in range(0, int(len(mas_symb) / 7)):  # voting by 7 candidates of symbol
        vot_list = []

        for j in range(i, i + len(mas_symb), int(len(mas_symb) / 7)):  # add 7 candidates
            vot_list.append(mas_symb[j])
        mpc = Counter(vot_list).most_common(1)[0][0]  # most popular candidate
        for j in range(i, i + len(mas_symb), int(len(mas_symb) / 7)):
            mas_symb[j] = mpc

    for ch in mas_symb:  # list to binary string
        final_extract += ch

    try:  # attempt to encoding message
        # print(final_extract)
        rmes1, rmesecc1, errata_pos1 = rsc.decode(final_extract)
        print(rmes1)

    except :
        rmes1 = ''
        print("Error of decoder")

    return rmes1


len_side_code = 89

mes = 7 * b'Correct extraction of'
print(len(mes))
Nbit = create_RS_code(mes, 106, 127, len_side_code, "data/RS_cod89x89.png")
print(Nbit)

# extr_RS = io.imread(r"D:\pythonProject\\phase_wm\RS_cod89x89.png")
# left = 0
# right = 138
# n = 89
# count = 0
# for sid in range(0, 100, 1):
#     extr_R = io.imread(r"D:\pythonProject\\phase_wm\RS_cod89x89.png")
#     random.seed(sid)
#     sampl = sample(list(combinations(range(0, n), 2)), right)
#     print(len(sampl))
#     for i in range(left, right):
#
#         extr_R[sampl[i][0] * 16:sampl[i][0] * 16 + 16, sampl[i][1] * 16:sampl[i][1] * 16 + 16] = np.where(
#             extr_R[sampl[i][0] * 16:sampl[i][0] * 16 + 16, sampl[i][1] * 16:sampl[i][1] * 16 + 16] == 255, 0, 255)
#
#     # extr_RS[0:1040,0:16*16]=0
#     comp=extr_RS==extr_R
#     at=(np.sum(comp==True))
#     # print(at/extr_R.size)
#     # print((right - left) / (89 * 89))
#     if (extract_RS2(extr_R, rsc, Nbit)) != '':
#         count += 1
#
# print(count)

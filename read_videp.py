import cv2
import pandas as pd


def read_video(path):
    vidcap = cv2.VideoCapture(path)
    count = 0
    success = True
    while success:
        n_fr = int(count // 29.97)
        print(n_fr)
        success, image = vidcap.read()
        if success:
            cv2.imwrite(r"D:\video_dataset\tmp_img\frame" + str(count) +"-"+ str(n_fr) + ".png", image)

        print("записан кадр", count)

        if cv2.waitKey(10) == 27:
            break
        count += 1
    return count


def parse_xlsx(path_xls):
    excel_data = pd.read_excel(path_xls)
    data = pd.DataFrame(excel_data)
    list_tmp=(data.iloc[1:,1].tolist())
    list_tmp2 = (data.iloc[1:, 0].tolist())
    list_mark=[int(x) for x in list_tmp]
    list_numb = [int(x) for x in list_tmp2]

    print(list_mark)
    print(list_numb)
    with open(r"D:\video_dataset\!final_mark4.txt", "w") as file:
        for i in range(len(list_numb)):
            file.write(str(list_numb[i]+1) + " " + str(list_mark[i]) + '\n')


def parse_txt(path_txt):
    list_mark = []
    file = open(path_txt, mode='r', encoding='utf-8-sig')
    lines = file.readlines()
    cnt=0

    list_numb=[i for i in range(1,235)]
    for line in lines:
        # print(cnt)

        ind=line.find("-")
        ind2=line.find(" ")
        print(cnt,int(line[ind+1:ind2])-int(line[:ind]))
        list_mark.extend((int(line[ind+1:ind2])-int(line[:ind])+1)*[int(line[ind2+1])])

        cnt += 1

    print(len(list_numb),len(list_mark))
    with open(r"D:\video_dataset\final_mark2.txt", "w") as file:
        for i in range(len(list_numb)):
            print(list_numb[i])
            file.write(str(list_numb[i]) + " " + str(list_mark[i]) + '\n')


# parse_txt("D:/video_dataset/marking2.txt")
parse_xlsx("D:/video_dataset/no_fight4.xlsx")

# read_video("D:/video_dataset/fight4.mp4")

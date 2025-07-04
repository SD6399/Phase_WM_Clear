import numpy as np
import matplotlib.pyplot as plt

x = np.arange(5, 2957, 5)  # от 5 до 3000 включительно с шагом 5

for vid_name in ["cut_RealBarca120", "IndiDance", "Road"]:
    for ampl in range(1, 3):
        base_filename = f'data/graphics/simple_method/mpeg/simple_a{ampl}_vot_sp_final_{vid_name}_mpeg_alf_001_lim2'
        with (open(f'{base_filename}.txt', 'r') as f_vot):
            for line in f_vot:
                s = [float(x.strip()) for x in line[8:].replace("]", "").replace("[", "").replace("\n", "").split(",")]
                plt.plot(x[:len(s)], s, label=f"Bitrate{line[4:7]}")

        if vid_name == "cut_RealBarca120":
            plot_name = "RealBarca"
        elif vid_name == "IndiDance":
            plot_name = "LutGaya"
        else:
            plot_name = "Road1"
        plt.title(plot_name + ". Амплитуда=" + str(ampl), fontsize=20)
        plt.legend(fontsize=20)
        plt.xlabel("Номер кадра", fontsize=20)
        plt.ylabel("Вероятность корректного извлечения", fontsize=20)
        plt.xticks(range(0, 2957, 100))
        plt.grid(True)
        plt.show()


# rand_k0 = [0.5064, 1.0, 1.0, 1.0, 1.0, 1.0]
# rand_k25 = [1.0, 1.0, 1.0, 1.0, 1.0]
# rand_k50 = [1.0, 1.0, 1.0, 1.0, 1.0]
# rand_k100 = [1.0, 1.0, 1.0, 1.0, 1.0]
#
# plt.title("RealBarca. Влияние начала извлечения на качество извлечения", fontsize=20)
# plt.plot(x[:len(rand_k0)], rand_k0, label=f"Извлечение с 0 кадра")
# plt.plot(x[5:len(rand_k25) + 5], rand_k25, label=f"Извлечение с 25 кадра")
# plt.plot(x[10:len(rand_k50) + 10], rand_k50, label=f"Извлечение с 50 кадра")
# plt.plot(x[20:len(rand_k100) + 20], rand_k100, label=f"Извлечение с 100 кадра")
# plt.xlabel("Номер кадра", fontsize=20)
# plt.ylabel("Вероятность корректного извлечения", fontsize=20)
# plt.xticks(range(0, 127, 10))
# plt.grid(True)
# plt.legend(fontsize=20)
# plt.show()

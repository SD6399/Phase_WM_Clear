import math
import random
import numpy as np


def rnd_gauss(mu, sigma, seed):
    # Gaussian random number generator
    np.random.seed(seed + 1)
    u1 = 1.0 - np.random.random()
    np.random.seed(seed + 2)
    u2 = 1.0 - np.random.random()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mu + z0 * sigma


class FieldGenerator:
    @staticmethod
    def draw_mosaic_field(N, ro, height, width, average, variance, seed):
        """
        :param N: количество направлений линий
        :param ro: желаемый коэффициент корреляции изображения
        :param height: высота изображения
        :param width:  ширина изображения
        :param average: ожидаемое мат ожидание изображения
        :param variance: ожидаемая дисперсия
        :param seed:
        :return:
        """
        fi = math.pi / N
        rays_number = N
        lambda_val = -math.log(ro) * np.tan(0.5 * math.pi / N)
        diameter = math.sqrt(width * width + height * height)
        K = int(diameter * lambda_val * rays_number) + 1

        image = np.zeros((height, width), dtype=int)
        color_count = 0

        # Marking areas when throwing a line
        for i in range(K):

            np.random.seed(seed + i)
            current_num = np.random.randint(0, rays_number)
            np.random.seed(seed + i)
            rtemp = np.random.random() * diameter - diameter / 2
            fi_current = fi * current_num
            np.random.seed(seed + i)
            increment = np.random.randint(0, 255)
            for x in range(width):
                for y in range(height):
                    x_condition = (x - width / 2.0) * math.cos(fi_current) + (-y + height / 2.0) * math.sin(
                        fi_current) + rtemp
                    if x_condition >= 0:
                        image[y, x] += (image[y, x] + increment) % 255
                        if color_count < image[y, x]:
                            color_count = image[y, x]

        # Generating brightness values
        colors = np.zeros(color_count + 1, dtype=float)
        sigma = math.sqrt(variance)
        for i in range(color_count + 1):
            colors[i] = rnd_gauss(average, sigma, i)

        # Coloring regions by number
        result = np.zeros((height, width), dtype=float)
        for x in range(width):
            for y in range(height):
                result[y, x] = colors[image[y, x]]

        print(result.shape)

        # result=np.where(result<0,result+255,result)
        # result = np.where(result > 255, result - 255, result)
        print("min/max", np.min(result), np.max(result))
        print(np.sum(result < 0))
        return result

    @staticmethod
    def draw_mosaic_field_non_isotropic(N, ro, angle, gamma, height, width, average, variance):
        """

        :param N:
        :param ro:
        :param angle: угол преимущественного направления корреляции (главная ось эллипса корфункции) вдоль этого
        направления будет больше линий генеритьс
        :param gamma: от 0 до 1 меняет соотношение осей эллипса кор функции, если 1 то линии уровня кор функции - круги,
         если уменьшается, то будут эллипсы
        :param height:
        :param width:
        :param average:
        :param variance:
        :return:
        """
        fi = math.pi / N
        lambdas = []
        Ks = []

        diameter = math.sqrt(width * width + height * height)
        sum_K = 0

        for i in range(N):
            alpha = 0.5 * (i * fi - math.pi * angle / 180)
            lambdas.append(-math.log(ro) * math.sqrt(gamma * gamma * math.cos(alpha) * math.cos(alpha) +
                                                     math.sin(alpha) * math.sin(alpha)) * math.tan(0.5 * math.pi / N))
            Ks.append(int(diameter * lambdas[i]) + 1)
            sum_K += Ks[i]

        randomizer = random.Random()
        x0 = width / 2.0
        y0 = height / 2.0

        image = np.zeros((height, width), dtype=int)

        fi_current = 0
        color_count = 0

        # Marking areas when throwing a line
        bar_value = 0
        for n in range(N):
            fi_current = fi * n
            for i in range(Ks[n]):
                rtemp = randomizer.random() * diameter - diameter / 2
                increment = randomizer.randint(0, 255)
                for x in range(width):
                    for y in range(height):
                        x_condition = (x - width / 2.0) * math.cos(fi_current) + (-y + height / 2.0) * math.sin(
                            fi_current) + rtemp
                        if x_condition >= 0:
                            image[y, x] += (image[y, x] + increment) % 255
                            if color_count < image[y, x]:
                                color_count = image[y, x]
                bar_value += 1

        # Generating brightness values
        colors = np.zeros(color_count + 1, dtype=float)
        sigma = math.sqrt(variance)
        for i in range(color_count + 1):
            colors[i] = rnd_gauss(average, sigma, i)

        # Computing correction coefficients for correct mean and variance
        sum_val = 0
        sum_squares = 0
        for x in range(width):
            for y in range(height):
                color = colors[image[y, x]]
                sum_val += color
                sum_squares += color * color
        sum_val /= (width * height)
        sum_squares /= (width * height)
        mpy_coef = math.sqrt(variance / (sum_squares - sum_val * sum_val))

        # Coloring regions by number
        result = np.zeros((height, width), dtype=float)
        for x in range(width):
            for y in range(height):
                result[y, x] = (colors[image[y, x]] - sum_val) * mpy_coef + average

        return result

    @staticmethod
    def draw_mosaic_field_non_isotropic_statistical(fis, lambdas, width, height, average, variance,
                                                    ):
        if len(fis) != len(lambdas):
            raise ValueError("not equal count of fis and lambdas")

        N = len(fis)
        Ks = []
        diameter = math.sqrt(width * width + height * height)
        sum_K = 0

        for i in range(N):
            Ks.append(int(diameter * lambdas[i]) + 1)
            sum_K += Ks[i]

        randomizer = random.Random()
        x0 = width / 2.0
        y0 = height / 2.0

        image = np.zeros((height, width), dtype=int)

        fi_current = 0
        color_count = 0

        # Marking areas when throwing a line
        bar_value = 0
        for n in range(N):
            fi_current = fis[n]
            for i in range(Ks[n]):
                rtemp = randomizer.random() * diameter - diameter / 2
                increment = randomizer.randint(0, 255)
                for x in range(width):
                    for y in range(height):
                        x_condition = (x - width / 2.0) * math.cos(fi_current) + (-y + height / 2.0) * math.sin(
                            fi_current) + rtemp
                        if x_condition >= 0:
                            image[y, x] += (image[y, x] + increment) % 255
                            if color_count < image[y, x]:
                                color_count = image[y, x]
                bar_value += 1

        # Generating brightness values
        colors = np.zeros(color_count + 1, dtype=float)
        sigma = math.sqrt(variance)
        for i in range(color_count + 1):
            colors[i] = rnd_gauss(average, sigma, i)

        # Computing correction coefficients for correct mean and variance
        sum_val = 0
        sum_squares = 0
        for x in range(width):
            for y in range(height):
                if (x % 1000 == 999) and (y % 1000 == 999):
                    print(x, y)
                color = colors[image[y, x]]
                sum_val += color
                sum_squares += color * color
        sum_val /= (width * height)
        sum_squares /= (width * height)
        mpy_coef = math.sqrt(variance / (sum_squares - sum_val * sum_val))

        # Coloring regions by number
        result = np.zeros((height, width), dtype=float)
        for x in range(width):
            for y in range(height):
                result[y, x] = (colors[image[y, x]] - sum_val) * mpy_coef + average

        return result

    @staticmethod
    def generate_energy_spectrum(r, dx, ver_size, hor_size, g, angle):
        cos_alpha = math.cos(angle)
        sin_alpha = math.sin(angle)
        result = np.zeros((ver_size, hor_size), dtype=float)
        coef = -2 * math.pi * math.log(r) * dx / g
        coef1 = math.log(r) ** 2
        two_pi = 2 * math.pi

        for i in range(-ver_size // 2, ver_size // 2 + 1):
            for j in range(-hor_size // 2, hor_size // 2 + 1):
                value = 0
                for l1 in range(-4, 4):
                    for l2 in range(-4, 4):
                        x = (cos_alpha * j + sin_alpha * i) / g
                        y = -sin_alpha * j + cos_alpha * i

                        value += (coef1 + (two_pi * (x / hor_size + l2 / g)) ** 2 + (
                                two_pi * (y / ver_size + l1)) ** 2) ** -1.5

                value *= coef
                result[(ver_size + i) % ver_size, (hor_size + j) % hor_size] = value

        return result


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

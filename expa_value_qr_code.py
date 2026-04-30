"""
qr_tools.py

Функции:
- generate_qr_matrix(text, error_level='M') -> numpy.ndarray of shape (89,89) with 0/1
- matrix_to_image(matrix, scale=10, border=4) -> PIL.Image (visualization)
- flip_bits_in_matrix(matrix, percent, seed=None) -> flipped_matrix, flipped_indices
- read_qr_from_matrix(matrix, scale=10, border=4) -> decoded_text or None

Зависимости:
pip install qrcode[pil] pillow numpy opencv-python

Пример использования внизу файла.
"""
from typing import Tuple, List, Optional
import qrcode
from qrcode.constants import ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H
import numpy as np
from PIL import Image
import cv2
import random

ERROR_MAP = {
    'L': ERROR_CORRECT_L,
    'M': ERROR_CORRECT_M,
    'Q': ERROR_CORRECT_Q,
    'H': ERROR_CORRECT_H,
}


def generate_qr_matrix(text: str, error_level: str = 'M') -> np.ndarray:
    """Создаёт QR-код версии 18 (размер модуля 89x89) и возвращает матрицу 0/1 без внешней белой рамки.

    error_level: одна из {'L','M','Q','H'}
    """
    error_level = error_level.upper()
    if error_level not in ERROR_MAP:
        raise ValueError("error_level должен быть одним из: 'L','M','Q','H'")

    qr = qrcode.QRCode(
        version=18,             # 21 + 4*(18-1) = 89
        error_correction=ERROR_MAP[error_level],
        box_size=1,
        border=0,               # без внешней рамки — вернём чистую матрицу 89x89
    )
    qr.add_data(text)
    qr.make(fit=False)  # фиксированная версия

    matrix = qr.get_matrix()  # list of lists of booleans, True=черный
    arr = np.array(matrix, dtype=np.uint8)
    # приводим к 0/1
    arr = arr * 1
    if arr.shape != (89, 89):
        raise RuntimeError(f"Ожидался размер (89,89), получили {arr.shape}")
    return arr


def matrix_to_image(matrix: np.ndarray, scale: int = 10, border: int = 4) -> Image.Image:
    """Преобразует матрицу 0/1 в PIL.Image (grayscale). scale — количество пикселей на модуль.
    border — ширина внешней белой зоны в модулях (по умолчанию 4 — рекомендуемая).
    """
    if matrix.dtype != np.uint8 and matrix.dtype != np.int64:
        matrix = matrix.astype(np.uint8)
    h, w = matrix.shape
    total_w = (w + 2 * border) * scale
    total_h = (h + 2 * border) * scale

    img = Image.new('L', (total_w, total_h), color=255)  # white background
    px = img.load()

    for r in range(h):
        for c in range(w):
            if matrix[r, c]:  # черный модуль
                # заполнить соответствующий блок scale x scale черным (0)
                start_x = (c + border) * scale
                start_y = (r + border) * scale
                for y in range(start_y, start_y + scale):
                    for x in range(start_x, start_x + scale):
                        px[x, y] = 0
    return img


def flip_bits_in_matrix(matrix: np.ndarray, percent: float, seed: Optional[int] = None) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    """Искажает заданный процент модулей (битов) случайным образом.

    Возвращает (new_matrix, list_of_flipped_indices).
    percent: от 0 до 100
    seed: необязательный сид для воспроизводимости
    """
    if not (0 <= percent <= 100):
        raise ValueError('percent должен быть в диапазоне [0,100]')
    rng = np.random.default_rng(seed)
    arr = matrix.copy().astype(np.uint8)
    h, w = arr.shape
    total = h * w
    k = int(round(total * (percent / 100.0)))
    if k == 0:
        return arr, []

    # Выбираем k уникальных индексов
    flat_indices = rng.choice(total, size=k, replace=False)
    flipped = []
    for idx in flat_indices:
        r = idx // w
        c = idx % w
        arr[r, c] = 1 - arr[r, c]
        flipped.append((r, c))
    return arr, flipped


def read_qr_from_matrix(matrix: np.ndarray, scale: int = 10, border: int = 4) -> Optional[str]:
    """Пытается распознать QR-код из матрицы. Возвращает распознанную строку или None.

    Используется OpenCV QRCodeDetector.
    """
    pil = matrix_to_image(matrix, scale=scale, border=border)
    # Конвертируем в формат OpenCV (BGR)
    img = np.array(pil)
    # img сейчас grayscale (H,W); QRCodeDetector может работать с ним

    detector = cv2.QRCodeDetector()
    # detectAndDecode возвращает tuple: data, points, straight_qrcode
    data, points, _ = detector.detectAndDecode(img)
    if data == "":
        return None
    return data


if __name__ == '__main__':
    # Пример: генерируем, искажаем и читаем
    text = 3*"Привет, QR! Это тестовая строка."
    print('Исходный текст:', text)

    for level in ['H', 'Q','M','L']:

        matrix = generate_qr_matrix(text, error_level=level)
        decoded = read_qr_from_matrix(matrix)

        flipped_matrix, flips = flip_bits_in_matrix(matrix, percent=1.8, seed=42)
        print(f'Искажено модулей: {len(flips)}')

        decoded_after = read_qr_from_matrix(flipped_matrix)
        print('Распознанный текст (после искажений):', decoded_after)

        # Сохранение изображений для визуальной инспекции
        img_orig = matrix_to_image(matrix, scale=4, border=4)
        # img_orig.save('qr_original.png')
        img_flipped = matrix_to_image(flipped_matrix, scale=4, border=4)
        # img_flipped.save('qr_flipped.png')
        # print('Сохранены файлы qr_original.png и qr_flipped.png')

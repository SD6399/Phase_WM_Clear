import matplotlib.pyplot as plt
import numpy as np
from reedsolo import RSCodec
from PIL import Image
from helper_methods import small2big, big2small
from collections import Counter


def create_RS_code(mes, nsym, len_side, count, output_path=None):
    """
    Encode a short message with Reed-Solomon, repeat it `count` times,
    and embed into a square binary image.

    :param mes: bytes — original message (must satisfy len(mes) <= 255 - nsym)
    :param nsym: int — number of RS parity symbols
    :param len_side: int — side length of square binary image
    :param count: int — number of repetitions of the encoded block
    :param output_path: str or None — save image if provided
    :return: tuple (total_bits_used, small_matrix)
    """
    # Validate message length for RS(255, k)
    if len(mes) > 255 - nsym:
        raise ValueError(f"Message too long: {len(mes)} > {255 - nsym} (for nsym={nsym})")

    # Encode once
    rsc = RSCodec(nsym=nsym)
    encoded_block = rsc.encode(mes)  # bytes, length = len(mes) + nsym

    # Repeat `count` times
    repeated_bytes = encoded_block * count  # bytes multiplication

    # Convert to bits
    byte_array = np.frombuffer(repeated_bytes, dtype=np.uint8)
    bits = np.unpackbits(byte_array)

    total_bits_needed = len_side * len_side
    if len(bits) > total_bits_needed:
        raise ValueError(
            f"Encoded data too large: {len(bits)} bits > {total_bits_needed} (image capacity)"
        )

    # Pad with zeros to fill the square
    padded_bits = np.pad(bits, (0, total_bits_needed - len(bits)), constant_values=0)
    small_matrix = padded_bits.reshape((len_side, len_side))

    # Optional: save image
    if output_path:
        img_array = small_matrix.copy()
        img_array[img_array == 1] = 255
        img = Image.fromarray(img_array.astype('uint8'))
        img.convert('RGB').save(output_path)

    total_bits_used = len(bits)
    return total_bits_used, small_matrix, len(encoded_block)


def extract_RS(image_RS, nsym, count, block_len_bytes, original_msg_len=None):
    """
    Extract and decode repeated RS-encoded message from binary image.

    :param image_RS: np.ndarray — binary image (0/1 or 0/255), shape (N, N)
    :param nsym: int — number of RS parity symbols (must match encoding)
    :param count: int — number of repetitions
    :param block_len_bytes: int — length of one encoded RS block in bytes
    :param original_msg_len: int or None — optional, for sanity check
    :return: decoded message (bytes) or empty bytes on failure
    """
    # Normalize image to 0/1
    mat = np.copy(image_RS)
    mat[mat == 255] = 1
    mat = mat.astype(int)

    total_bits = mat.size
    total_bytes = total_bits // 8
    bits = mat.reshape(-1)[: total_bytes * 8]
    byte_data = np.packbits(bits).tobytes()

    expected_total_bytes = count * block_len_bytes
    if len(byte_data) < expected_total_bytes:
        print(f"Warning: not enough data. Got {len(byte_data)} bytes, need {expected_total_bytes}")
        # Try to proceed with what we have (pad with zeros)
        byte_data = byte_data.ljust(expected_total_bytes, b'\x00')

    # Split into `count` blocks
    blocks = []
    for i in range(count):
        start = i * block_len_bytes
        end = start + block_len_bytes
        blocks.append(byte_data[start:end])

    # Voting per byte position
    voted_block = bytearray()
    for pos in range(block_len_bytes):
        candidates = [blocks[i][pos] for i in range(count)]
        voted_byte = Counter(candidates).most_common(1)[0][0]
        voted_block.append(voted_byte)

    # Decode
    rsc = RSCodec(nsym=nsym)
    try:
        decoded, decoded_ecc, errata_pos = rsc.decode(voted_block)
        if original_msg_len is not None:
            decoded = decoded[:original_msg_len]  # truncate if needed
        return decoded
    except Exception as e:
        print(f"RS decoding failed: {e}")
        return b""


def add_random_bit_flips(binary_matrix, flip_ratio=0.10, seed=None):
    """
    Invert a random fraction of bits in a binary matrix (e.g., to simulate noise or distortion).

    :param binary_matrix: np.ndarray — input matrix with values 0/1 or 0/255
    :param flip_ratio: float — fraction of bits to flip (default: 0.10 = 10%)
    :param seed: int or None — for reproducibility
    :return: np.ndarray — matrix with flipped bits (same dtype as input)
    """
    if seed is not None:
        np.random.seed(seed)

    mat = binary_matrix.copy()

    # Normalize to 0/1 if needed
    is_255 = (mat.max() == 255)
    if is_255:
        mat = mat.astype(np.uint8)
        mat[mat == 255] = 1

    # Flatten for easy indexing
    flat = mat.flatten()
    total_bits = flat.size
    num_flips = int(flip_ratio * total_bits)

    if num_flips > 0:
        # Randomly choose indices to flip
        flip_indices = np.random.choice(total_bits, size=num_flips, replace=False)
        flat[flip_indices] = 1 - flat[flip_indices]  # invert: 0<->1

    # Reshape back
    flipped = flat.reshape(mat.shape)

    # Restore 0/255 format if input was in that format
    if is_255:
        flipped_255 = flipped.astype(np.uint8) * 255
        return flipped_255
    else:
        return flipped.astype(binary_matrix.dtype)


if __name__ == '__main__':
    len_side_code = 89  # must be large enough
    # count = 1
    # nsym = 31
    # symb = b'sousigge'
    # # mes = b'Correct extraction of redis at garden and tomato at forest. I wanna sleep and finalize this testing'
    # for num in range(13, 18):
    #     mes = symb * num
    #     # print("Original message:", mes)
    #     # print("Message length:", len(mes))
    #
    #     # Encode
    #     total_bits, clean_matrix, block_len = create_RS_code(
    #         mes=mes,
    #         nsym=nsym,
    #         len_side=len_side_code,
    #         count=count,
    #         output_path="data/RS_cod.png"
    #     )
    #
    #     print(f"Encoded block length: {block_len} bytes")
    #     print(f"Total bits used: {total_bits}")
    #
    #     noisy_matrix = add_random_bit_flips(clean_matrix, flip_ratio=0.02, seed=42)
    #
    #     # Decode
    #     recovered = extract_RS(
    #         image_RS=noisy_matrix,
    #         nsym=nsym,
    #         count=count,
    #         block_len_bytes=block_len,
    #         original_msg_len=len(mes)
    #     )
    #
    #     print("Number", num)
    #     print("Recovered message:", recovered)
    #     print("Match:", recovered == mes)

    plt.plot([11, 12, 13, 14, 15], [77 * 8, 61 * 8, 472, 432, 352])
    plt.xlabel("Процент искажений", fontsize=20)
    plt.ylabel("Количество встроенных бит", fontsize=20)
    plt.title("Объём встраивания полезной информации", fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)

    plt.show()

from __future__ import annotations

"""
EEG analogue of the spectral watermark pipeline from `main2.py`.

Video frames are (H, W, time). EEG is represented as a 3D cube (H, W, T) built from EDF:
channels × time → square grid (H, W) × temporal windows (T).

Important: do NOT `import scpetrcal_halftone` as a module — that file executes IO at import time.
The spectral/spatial helpers below mirror `scpetrcal_halftone.py` without side effects.
"""

import math
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


# --- Spectral <-> spatial helpers (from `scpetrcal_halftone.py`, import-safe copy) ---


def spatial_to_spectral(qr_spatial: np.ndarray) -> np.ndarray:
    """Spatial 2D patch → normalized magnitude spectrum (same logic as `check_spatial2spectr`)."""
    qr_spatial = np.asarray(qr_spatial, dtype=np.float64)
    if qr_spatial.ndim != 2:
        raise ValueError("spatial_to_spectral expects a 2D array")

    f = np.fft.fft2(qr_spatial)
    f[0, 0] = 0
    mag = np.abs(f)
    mag = 255 * (mag - np.min(mag)) / (np.max(mag) - np.min(mag) + 1e-12)
    return mag.astype(np.float32)


def spectral_to_spatial(qr: np.ndarray, *, size_wm: int, shift: int = 40) -> np.ndarray:
    """
    Spectral QR (typically 49×49, values {0,255}) → spatial patch (size_wm×size_wm),
    same construction as `spectr_to_spatial` in `scpetrcal_halftone.py`.
    Returns real-valued spatial image scaled to uint8-friendly float32 in [0,255].
    """
    size_qr = 49
    half_qr_size = int((size_qr + 1) / 2)  # 25

    qr_back = np.zeros((size_wm, size_wm), dtype=np.complex128)
    qr_c = qr.astype(np.complex128, copy=True)

    rng = np.random.default_rng()
    mask = qr_c == 255
    qr_c[mask] = np.exp(1j * rng.uniform(0, 2 * np.pi, size=int(np.count_nonzero(mask))))

    qr_back[shift + 1 : shift + size_qr + 1, shift + 1 : shift + half_qr_size] = qr_c[:, : half_qr_size - 1]

    for i in range(size_qr + 1):
        for j in range(half_qr_size - 1):
            qr_back[-shift - (i + 1), -shift - (j + 1)] = np.conj(qr_back[shift + i + 1, shift + j + 1])

    if shift != 0:
        qr_back[shift + 1 : shift + size_qr + 1, -shift - half_qr_size : -shift] = qr_c[:, half_qr_size - 1 :]
    else:
        qr_back[shift + 1 : shift + size_qr + 1, shift - half_qr_size :] = qr_c[:, half_qr_size - 1 :]

    for i in range(size_qr + 1):
        for j in range(1, half_qr_size + 1):
            if shift != 0:
                qr_back[-shift - (i + 1), shift + j] = np.conj(qr_back[shift + i + 1, -shift - j])
            else:
                qr_back[-shift - (i + 1), shift + j] = np.conj(qr_back[shift + i + 1, -j])

    spat = np.fft.ifft2(qr_back).real
    spat = 255 * (spat - np.min(spat)) / (np.max(spat) - np.min(spat) + 1e-12)
    return spat.astype(np.float32)


@dataclass(frozen=True)
class EegCube:
    cube: np.ndarray  # float32, shape (H, W, T)
    channel_grid_shape: tuple[int, int]
    sfreq: Optional[float]
    meta: dict


def iter_edf_files(root: str) -> list[str]:
    edfs: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".edf"):
                edfs.append(os.path.join(dirpath, fn))
    edfs.sort()
    return edfs


def _channel_grid(n_channels: int) -> tuple[int, int]:
    h = int(math.ceil(math.sqrt(n_channels)))
    return h, h


def load_edf_as_cube(
    edf_path: str,
    *,
    window_size: int = 32,
    step_size: Optional[int] = None,
    max_frames: Optional[int] = 1024,
    picks: Optional[Iterable[str]] = None,
    scale: str = "robust",
) -> EegCube:
    """Read `.edf` via MNE and build (H, W, T) cube from channels×time."""
    try:
        import mne  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Для чтения .edf установите `mne` (например `pip install mne`).") from e

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    if picks is not None:
        raw.pick(list(picks))

    data = np.asarray(raw.get_data(), dtype=np.float32)
    if data.size == 0:
        raise ValueError(f"Пустые данные в EDF: {edf_path}")

    sfreq = float(raw.info["sfreq"]) if raw.info and "sfreq" in raw.info else None

    if scale == "robust":
        med = np.median(data, axis=1, keepdims=True)
        mad = np.median(np.abs(data - med), axis=1, keepdims=True)
        data = (data - med) / (mad + 1e-6)
    elif scale == "zscore":
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        data = (data - mean) / (std + 1e-6)
    elif scale == "none":
        pass
    else:
        raise ValueError(f"Unknown scale='{scale}'. Use 'robust', 'zscore', or 'none'.")

    n_channels, n_times = data.shape
    if step_size is None:
        step_size = window_size
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size и step_size должны быть > 0")

    h, w = _channel_grid(n_channels)
    grid_ch = h * w
    pad = grid_ch - n_channels
    if pad > 0:
        data = np.pad(data, ((0, pad), (0, 0)), mode="constant", constant_values=0.0)

    starts = np.arange(0, max(0, n_times - window_size + 1), step_size, dtype=int)
    if max_frames is not None:
        starts = starts[:max_frames]

    frames: list[np.ndarray] = []
    for st in starts:
        win = data[:, st : st + window_size]
        frames.append(np.mean(win, axis=1, dtype=np.float32))

    if not frames:
        raise ValueError("Не удалось сформировать ни одного временного окна.")

    frames_arr = np.stack(frames, axis=1)
    cube = frames_arr.reshape(h, w, frames_arr.shape[1]).astype(np.float32, copy=False)

    return EegCube(
        cube=cube,
        channel_grid_shape=(h, w),
        sfreq=sfreq,
        meta={
            "edf_path": edf_path,
            "n_channels_original": n_channels,
            "window_size": int(window_size),
            "step_size": int(step_size),
            "n_frames": int(cube.shape[2]),
            "scale": scale,
        },
    )


def _tile_apply_wm(frame: np.ndarray, wm: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    """
    Same tiling idea as `embed()` in `main2.py`: repeatedly embed one WM tile across the frame.
    `wm` must have the same shape as the binary stencil used to generate it.
    """
    a = np.asarray(frame, dtype=np.float32)
    wm = np.asarray(wm, dtype=np.float32)

    if a.ndim != 2:
        raise ValueError("frame must be 2D (H,W)")
    if wm.ndim != 2:
        raise ValueError("wm must be 2D")

    tile_h, tile_w = wm.shape
    out = np.copy(a)

    # Mirrors the nested loops in `main2.embed`, including boundary partial tiles.
    for row_ind in range(0, a.shape[0], tile_h):
        for col_ind in range(0, a.shape[1], tile_w):
            if ((a.shape[0] - row_ind) >= tile_h) and ((a.shape[1] - col_ind) >= tile_w):
                sub_a = a[row_ind : row_ind + tile_h, col_ind : col_ind + tile_w, ...]
                out[row_ind : row_ind + tile_h, col_ind : col_ind + tile_w] = np.where(
                    np.float32(sub_a + wm) > vmax,
                    vmax,
                    np.where(sub_a + wm < vmin, vmin, np.float32(sub_a + wm)),
                )
            elif ((a.shape[0] - row_ind) < tile_h) and ((a.shape[1] - col_ind) >= tile_w):
                sub_a = a[row_ind:a.shape[0], col_ind : col_ind + tile_w]
                sub_wm = wm[: a.shape[0] - row_ind, :]
                out[row_ind:a.shape[0], col_ind : col_ind + tile_w] = np.where(
                    np.float32(sub_a + sub_wm) > vmax,
                    vmax,
                    np.where(sub_a + sub_wm < vmin, vmin, np.float32(sub_a + sub_wm)),
                )
            elif ((a.shape[0] - row_ind) >= tile_h) and ((a.shape[1] - col_ind) < tile_w):
                sub_a = a[row_ind : row_ind + tile_h, col_ind : a.shape[1]]
                sub_wm = wm[:, : a.shape[1] - col_ind]
                out[row_ind : row_ind + tile_h, col_ind : a.shape[1]] = np.where(
                    np.float32(sub_a + sub_wm) > vmax,
                    vmax,
                    np.where(sub_a + sub_wm < vmin, vmin, np.float32(sub_a + sub_wm)),
                )
            else:
                sub_a = a[row_ind:a.shape[0], col_ind : a.shape[1]]
                sub_wm = wm[: a.shape[0] - row_ind, : a.shape[1] - col_ind]
                out[row_ind:a.shape[0], col_ind : a.shape[1]] = np.where(
                    np.float32(sub_a + sub_wm) > vmax,
                    vmax,
                    np.where(sub_a + sub_wm < vmin, vmin, np.float32(sub_a + sub_wm)),
                )

    return out


def embed(
    cube: np.ndarray,
    *,
    binary_pattern: np.ndarray,
    amplitude: float,
    tt: float,
    var: float = 0.0,
    fi_scale: Optional[float] = None,
) -> np.ndarray:
    """
    EEG analogue of `embed()` from `main2.py`.

    `binary_pattern` is the same role as binarized QR stencil `st_qr` (2D).
    Each time slice gets:
      wm = amplitude * sin(cnt * tt + (pi/(2*255)) * binary_pattern)
    and wm is tiled across the frame.
    """
    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError("cube must be (H,W,T)")
    st_qr = np.asarray(binary_pattern, dtype=np.float32)
    if st_qr.ndim != 2:
        raise ValueError("binary_pattern must be 2D")

    fi = fi_scale if fi_scale is not None else (math.pi / 2 / 255.0)
    out = np.empty_like(cube, dtype=np.float32)
    vmin = float(np.min(cube))
    vmax = float(np.max(cube))

    for cnt in range(cube.shape[2]):
        temp = fi * st_qr
        wm = amplitude * np.sin(cnt * tt + temp)
        framed = _tile_apply_wm(cube[:, :, cnt], wm.astype(np.float32), vmin=vmin, vmax=vmax)

        if var and var > 0:
            rng = np.random.default_rng((cnt + 1) * 1000003)
            noise = rng.normal(0.0, float(var) ** 0.5, size=framed.shape).astype(np.float32)
            framed = framed + noise

        out[:, :, cnt] = framed
    return out


def _histogram_phase_normalize_like_main2(wm_uint8: np.ndarray) -> np.ndarray:
    """
    Phase histogram normalization block from `main2.py` (after computing wm in [0..255]),
    adapted to use vectorized numpy histogram for speed (same math structure).
    """
    from skimage.exposure import histogram  # local import: keeps module import lighter

    a1 = np.asarray(wm_uint8, dtype=np.float32)
    fi = (a1 * np.pi * 2.0) / 255.0

    coord1 = np.where(fi < np.pi, (fi / np.pi * 2 - 1) * (-1), ((fi - np.pi) / np.pi * 2 - 1))
    coord2 = np.where(
        fi < np.pi / 2,
        (fi / np.pi / 2),
        np.where(
            fi > 3 * np.pi / 2,
            ((fi - 1.5 * np.pi) / np.pi * 2) - 1,
            ((fi - 0.5 * np.pi) * 2 / np.pi - 1) * (-1),
        ),
    )

    hist, bin_centers = histogram(coord1, normalize=False)
    hist2, bin_centers2 = histogram(coord2, normalize=False)

    # replicate main2 indexing assumptions
    mx_sp = np.arange(bin_centers2[0], bin_centers2[-1], bin_centers2[1] - bin_centers2[0])
    ver = hist2 / (np.sum(hist) + 1e-12)
    mo = np.sum(bin_centers2 * ver)
    dis = np.abs(mo - mx_sp)
    pr1 = np.min(dis)

    mx_sp2 = np.arange(bin_centers2[0], bin_centers2[-1], bin_centers2[1] - bin_centers2[0])
    ver2 = hist2 / (np.sum(hist2) + 1e-12)
    mo2 = np.sum(bin_centers2 * ver2)
    dis2 = np.abs(mo2 - mx_sp2)
    x = np.min(dis2)

    idx = int(np.argmin(np.abs(dis2 - x)))
    pr2 = float(bin_centers2[idx])

    moment = np.where(
        pr1 < 0,
        np.arctan((pr2 / pr1)) + np.pi,
        np.where(pr2 >= 0, np.arctan((pr2 / pr1)), np.arctan((pr2 / pr1)) + 2 * np.pi),
    )

    # Vectorized equivalent of the branching in `main2.py` (must work for full H×W maps).
    fi_branch = fi
    fi_adj = np.where(fi_branch < np.pi / 4, fi_branch + 2 * np.pi, fi_branch)

    cond_mid = (moment >= np.pi / 4) & (moment <= np.pi * 2 - np.pi / 4)
    cond_high = moment > np.pi * 2 - np.pi / 4

    fi_tmp = np.where(cond_mid, fi_branch - moment + 0.25 * np.pi, np.nan)
    fi_tmp = np.where(cond_high & ~cond_mid, fi_adj - moment + 0.25 * np.pi, fi_tmp)
    fi_tmp = np.where(~cond_mid & ~cond_high, fi_branch - 2 * np.pi - moment + 0.25 * np.pi, fi_tmp)

    fi_tmp = np.where(fi_tmp < -np.pi / 4, fi_tmp + 2 * np.pi, fi_tmp)
    fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)
    fi_tmp = np.clip(fi_tmp, 0, np.pi)
    l_kadr = fi_tmp * 255.0 / np.pi

    l_kadr = 255.0 * (l_kadr - np.min(l_kadr)) / (np.max(l_kadr) - np.min(l_kadr) + 1e-12)
    return l_kadr.astype(np.float32)


def extract(
    cube: np.ndarray,
    *,
    alf: float,
    beta: float,
    tt: float,
    rand_fr: int = 0,
    normalize_phase_histogram: bool = True,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict]:
    """
    EEG analogue of the core `extract()` recursion from `main2.py`.

    Returns:
      phase_maps: float32 array (H,W,T) in ~[0..255] after optional normalization (like `l_kadr`)
      aux: small diagnostics dict
    """
    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError("cube must be (H,W,T)")
    h, w, t_total = cube.shape

    rand_fr = int(rand_fr)
    if rand_fr < 0 or rand_fr >= t_total:
        raise ValueError("rand_fr must be within [0, T)")

    # --- first smoothing pass (as in main2) ---
    smooth = np.empty_like(cube, dtype=np.float32)
    f1 = None
    for cnt in range(t_total):
        arr = cube[:, :, cnt]
        if cnt == rand_fr:
            f1 = arr
        else:
            assert f1 is not None
            f1 = np.float32(f1) * alf + np.float32(arr) * (1.0 - alf)
        np.clip(f1, np.min(cube), np.max(cube), out=f1)
        smooth[:, :, cnt] = f1

    diff = smooth - cube

    # --- recursive spectral phase extraction (structure aligned with main2) ---
    g = np.zeros((h, w), dtype=np.float32)
    d = np.ones((h, w), dtype=np.float32)
    f = np.zeros((h, w), dtype=np.float32)

    g2 = np.zeros((h, w), dtype=np.complex64)
    d2 = np.ones((h, w), dtype=np.complex64) + 1j * np.ones((h, w), dtype=np.complex64)
    f2 = np.zeros((h, w), dtype=np.complex64)

    phase_maps = np.zeros_like(cube, dtype=np.float32)

    for cnt in range(rand_fr, t_total):
        a1 = diff[:, :, cnt]

        g = np.copy(d)
        d = np.copy(f)

        if cnt == rand_fr:
            f = np.copy(a1)
            d = np.ones((h, w), dtype=np.float32)
        elif cnt == rand_fr + 1:
            f = 2.0 * beta * math.cos(tt) * np.float32(d) + np.float32(a1)
        else:
            f = 2.0 * beta * math.cos(tt) * np.float32(d) - (beta**2) * np.float32(g) + np.float32(a1)

        yc = np.float32(f) - beta * math.cos(tt) * np.float32(d)
        ys = beta * math.sin(tt) * np.float32(d)

        tmp_signal = np.empty((h, w), dtype=np.complex64)
        tmp_signal.real = yc.astype(np.float32)
        tmp_signal.imag = ys.astype(np.float32)

        g2 = np.copy(d2)
        d2 = np.copy(f2)

        if cnt == rand_fr:
            f2 = tmp_signal
            d2 = np.ones((h, w), dtype=np.complex64)
            d2.imag = np.ones((h, w), dtype=np.float32)
        elif cnt == rand_fr + 1:
            f2.real = 2 * beta * math.cos(tt) * np.float32(d2.real) + np.float32(tmp_signal.real)
            f2.imag = 2 * beta * math.cos(tt) * np.float32(d2.imag) + np.float32(tmp_signal.imag)
        else:
            f2.real = 2 * beta * math.cos(tt) * np.float32(d2.real) - (beta**2) * np.float32(g2.real) + np.float32(
                tmp_signal.real
            )
            f2.imag = 2 * beta * math.cos(tt) * np.float32(d2.imag) - (beta**2) * np.float32(g2.imag) + np.float32(
                tmp_signal.imag
            )

        c = np.cos(tt * cnt) * np.float32(f2.real) + np.sin(tt * cnt) * np.float32(f2.imag)
        s = np.cos(tt * cnt) * np.float32(f2.imag) - np.sin(tt * cnt) * np.float32(f2.real)

        fi = np.arctan2(s, c + eps)
        fi = np.nan_to_num(fi)
        fi = np.where(fi < -np.pi / 4, fi + 2 * np.pi, fi)
        fi = np.where(fi > 9 * np.pi / 4, fi - 2 * np.pi, fi)

        wm = 255.0 * fi / 2.0 / math.pi
        wm = np.clip(wm, 0, 255)

        if normalize_phase_histogram:
            l_kadr = _histogram_phase_normalize_like_main2(wm.astype(np.uint8))
        else:
            l_kadr = wm.astype(np.float32)

        phase_maps[:, :, cnt] = l_kadr

    aux = {"rand_fr": rand_fr, "smooth": smooth, "diff": diff}
    return phase_maps, aux


def spectral_aggregate_over_phase_map(
    phase_map: np.ndarray,
    *,
    patch: int,
    stride: int,
    divide_by: float = 8.0,
) -> np.ndarray:
    """Same sliding-window averaging idea as in `main2.py` around `check_spatial2spectr(...)`."""
    phase_map = np.asarray(phase_map, dtype=np.float32)
    if phase_map.ndim != 2:
        raise ValueError("phase_map must be 2D")

    acc = np.zeros((patch, patch), dtype=np.float64)
    count = 0
    for row_ind in range(0, phase_map.shape[0] - patch + 1, stride):
        for col_ind in range(0, phase_map.shape[1] - patch + 1, stride):
            spat = phase_map[row_ind : row_ind + patch, col_ind : col_ind + patch]
            spec = spatial_to_spectral(spat)
            acc += spec.astype(np.float64)
            count += 1

    if count == 0:
        raise ValueError("phase_map too small for given patch/stride")

    acc /= divide_by
    acc = 255.0 * (acc - np.min(acc)) / (np.max(acc) - np.min(acc) + 1e-12)
    return acc.astype(np.float32)


def spectral_method_cube(*args, **kwargs) -> np.ndarray:
    """Backward-compatible alias: returns normalized phase maps only."""
    phase, _aux = extract(*args, **kwargs)
    return phase


# %% Example usage (EEG EDF -> embed -> extract -> spatial/spectral)
if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise SystemExit(f"matplotlib недоступен: {e}")

    EEG_ROOT = r"D:\dk\eeg\files"
    edfs = iter_edf_files(EEG_ROOT)
    if not edfs:
        raise SystemExit(f"Не найдено ни одного .edf в {EEG_ROOT}")

    base = load_edf_as_cube(edfs[0], window_size=32, step_size=32, max_frames=128, scale="robust")

    # demo pattern: small binary stencil (replace with real QR-like stencil if needed)
    rng = np.random.default_rng(0)
    pat = (rng.random((8, 8)) > 0.5).astype(np.float32) * 255.0

    watermarked = embed(base.cube, binary_pattern=pat, amplitude=0.05, tt=2.9, var=0.0)
    phase_maps, _aux = extract(watermarked, alf=0.005, beta=0.999, tt=2.9, rand_fr=0)

    mid = phase_maps.shape[2] // 2
    agg = spectral_aggregate_over_phase_map(phase_maps[:, :, mid], patch=min(32, base.cube.shape[0]), stride=8)

    print("EDF:", base.meta["edf_path"])
    print("cube:", base.cube.shape, "phase:", phase_maps.shape, "agg:", agg.shape)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(base.cube[:, :, mid], cmap="gray")
    axes[0].set_title("EEG frame (spatial)")
    axes[1].imshow(phase_maps[:, :, mid], cmap="gray")
    axes[1].set_title("extracted phase map")
    axes[2].imshow(agg, cmap="gray")
    axes[2].set_title("spectral aggregate")
    for ax in axes:
        ax.axis("off")
    plt.show()

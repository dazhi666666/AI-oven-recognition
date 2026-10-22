"""人工构造的左右图（已知视差），供 stereo 测试使用。

约定：视差 d = 左图横坐标 - 右图横坐标，即 right(x - d) = left(x)。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from oven_stereo.stereo import SGBMConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def _texture(h: int, w: int, seed: int) -> np.ndarray:
    """确定性的纹理：高频噪声 + 低频起伏，浮点 0–255。"""
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32) * 255.0, (3, 3), 0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    smooth = 120.0 + 80.0 * np.sin(xx / 13.0) * np.cos(yy / 17.0)
    return np.clip(0.6 * noise + 0.4 * smooth, 0, 255)


@pytest.fixture
def make_shift_pair():
    """整幅平移 shift 像素：right(x) = left(x + shift)，真实视差处处为 shift。"""

    def make(shift: int = 20, h: int = 160, w: int = 256, seed: int = 11):
        left = _texture(h, w, seed).astype(np.uint8)
        right = np.empty_like(left)
        right[:, : w - shift] = left[:, shift:]
        right[:, w - shift :] = left[:, -1:]
        return left, right

    return make


@pytest.fixture(scope="session")
def slanted_pair():
    """斜平面：视差随列线性变化 d(x) = d0 + k * x（x 为左图列）。

    右图第 x_r 列取左图纹理坐标 x_l，满足 x_l = x_r + d(x_l)，即 x_l = (x_r + d0) / (1 - k)。
    返回 left, right, 真实视差, 真实视差有效掩膜。
    """
    h, w, d0, k = 240, 320, 16.0, 0.10
    tex = _texture(h, w, seed=3)
    xx = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    x_src = (xx + d0) / (1.0 - k)
    x0 = np.clip(np.floor(x_src).astype(np.int64), 0, w - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    frac = x_src - np.floor(x_src)
    rows = np.arange(h)[:, None]
    right = (1 - frac) * tex[rows, x0] + frac * tex[rows, x1]
    d_gt = d0 + k * xx
    valid_gt = (xx - d_gt) >= 0  # 对应点要落在右图内
    return tex.astype(np.uint8), np.clip(right, 0, 255).astype(np.uint8), d_gt, valid_gt


@pytest.fixture
def c01_config_path() -> Path:
    return REPO_ROOT / "configs" / "stereo" / "sgbm_c01.yaml"


@pytest.fixture
def c01_config(c01_config_path) -> SGBMConfig:
    return SGBMConfig.from_yaml(c01_config_path)

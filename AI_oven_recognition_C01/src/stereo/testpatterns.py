"""**人工构造**测试图样（非真实场景、非数据集、非 C-04 合成资产）。

用途只有两个：

1. :func:`render_slanted_plane_pair` —— 已知解析视差的斜平面图对，用于
   *实测核验 OpenCV 输出的定点缩放*（任务书步骤 3）与端到端链路自检。
2. 给单元测试提供确定性输入。

必须遵守的表述纪律（任务书"依赖和不等待事项"、"验收"）：

* 由本模块产生的任何结果都标注 ``provenance="artificial_test_pattern"``，
  **不得**用来宣称真实链路完成或任何精度达标。
* 斜平面而非常数视差平面：实测常数视差平面在 SGBM 下是病态输入
  （无语义梯度，匹配退化），不能作为解码核验依据。

渲染约定（左校正相机系，左图为参考网格）::

    left(x, y)  = tex(x, y)
    right(x_r, y) = tex(x_l, y),   x_l = (x_r + d0) / (1 - k)
    d_gt(x_l)   = d0 + k * x_l            （左图列上的真值视差）

求解依据：右图某像素对应左图纹理坐标 ``x_l`` 满足 ``x_l = x_r + d(x_l)``，
代入线性视差 ``d(x) = d0 + k*x`` 得 ``x_l = (x_r + d0) / (1 - k)``。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np

__all__ = [
    "ARTIFICIAL_PROVENANCE",
    "render_slanted_plane_pair",
    "render_constant_shift_pair",
    "unit_check_pattern",
]

ARTIFICIAL_PROVENANCE = "artificial_test_pattern"

# 自检图样固定几何（单位：像素；只用于解码核验，不代表任何真实场景尺度）
UNIT_CHECK_D0_PX = 16.0
UNIT_CHECK_SLOPE = 0.10


def _texture(h: int, w: int, seed: int = 3) -> np.ndarray:
    """带高频细节 + 低频结构的确定性纹理（浮点 0..255）。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(
        (rng.random((h, w)).astype(np.float32) * 255.0), (3, 3), 0
    )
    smooth = 120.0 + 80.0 * np.sin(xx / 13.0) * np.cos(yy / 17.0)
    return np.clip(0.6 * noise + 0.4 * smooth, 0.0, 255.0)


def _bilinear_x(img: np.ndarray, x_coords: np.ndarray) -> np.ndarray:
    """沿 x 方向双线性采样（y 取整），越界列用最近边界。"""
    h, w = img.shape[:2]
    x0 = np.floor(x_coords)
    frac = (x_coords - x0).astype(np.float32)
    x0i = np.clip(x0.astype(np.int32), 0, w - 1)
    x1i = np.clip(x0i + 1, 0, w - 1)
    yi = np.clip(np.arange(h, dtype=np.int32)[:, None], 0, h - 1)
    a = img[yi, x0i]
    b = img[yi, x1i]
    return (1.0 - frac) * a + frac * b


def render_slanted_plane_pair(
    height: int = 240,
    width: int = 320,
    d0: float = UNIT_CHECK_D0_PX,
    slope: float = UNIT_CHECK_SLOPE,
    seed: int = 3,
) -> Dict[str, Any]:
    """渲染斜平面人工图对，返回左右图与**左图列**上的解析真值视差。

    Returns
    -------
    dict
        ``left``/``right`` (uint8 HxW)、``d_gt_px`` (float32 HxW，左图列真值)、
        ``valid_gt`` (bool HxW，右图采样落在图内且真值在可表示范围内)、``meta``。
    """
    if not (0 < height and 0 < width):
        raise ValueError("height/width 必须为正")
    if not (0.0 <= slope < 1.0):
        raise ValueError("slope 需在 [0,1) 内，且 1-slope 不能为 0")
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    tex = _texture(height, width, seed=seed)

    # 左图为参考网格
    left_f = tex
    # 右图：对每个右图列 x_r，采样纹理坐标 x_l = (x_r + d0) / (1 - slope)
    x_l_for_right = (xx + d0) / (1.0 - slope)
    right_f = _bilinear_x(tex, x_l_for_right)

    d_gt = d0 + slope * xx
    # 真值在左图列有效域：右图采样需落在纹理内，即 x_l_for_right <= width-1
    valid_gt = x_l_for_right <= (width - 1)

    left = np.clip(left_f, 0, 255).astype(np.uint8)
    right = np.clip(right_f, 0, 255).astype(np.uint8)

    meta = {
        "provenance": ARTIFICIAL_PROVENANCE,
        "kind": "slanted_plane",
        "height": int(height),
        "width": int(width),
        "d0_px": float(d0),
        "slope_px_per_px": float(slope),
        "d_gt_range_px": [float(d_gt.min()), float(d_gt.max())],
        "note": (
            "人工构造：由解析线性视差扭曲纹理得到；不含真实相机噪声、"
            "曝光差异或标定误差，不能用于任何精度达标声明"
        ),
    }
    return {
        "left": left,
        "right": right,
        "d_gt_px": d_gt.astype(np.float32),
        "valid_gt": valid_gt.astype(bool),
        "meta": meta,
    }


def render_constant_shift_pair(
    height: int = 160, width: int = 256, shift_px: int = 20, seed: int = 11
) -> Dict[str, Any]:
    """常数平移图对（**仅**用于错误用例/非退化检查，不作为精度核验）。

    注意：常数视差平面在 SGBM 下属于病态输入，本函数不用于解码缩放核验。
    """
    if shift_px < 0 or shift_px >= width:
        raise ValueError("shift_px 需在 [0, width) 内")
    tex = _texture(height, width, seed=seed).astype(np.uint8)
    right = np.zeros_like(tex)
    right[:, shift_px:] = tex[:, : width - shift_px]
    right[:, :shift_px] = tex[:, :1]
    d_gt = np.full((height, width), float(shift_px), dtype=np.float32)
    valid_gt = np.zeros((height, width), dtype=bool)
    valid_gt[:, shift_px:] = True
    return {
        "left": tex,
        "right": right,
        "d_gt_px": d_gt,
        "valid_gt": valid_gt,
        "meta": {
            "provenance": ARTIFICIAL_PROVENANCE,
            "kind": "constant_shift",
            "shift_px": int(shift_px),
            "note": "常数视差为病态输入，仅用于错误用例与非退化检查",
        },
    }


def unit_check_pattern() -> Dict[str, Any]:
    """解码缩放自检图样（固定参数的斜平面对）。"""
    return render_slanted_plane_pair(
        height=240, width=320, d0=UNIT_CHECK_D0_PX, slope=UNIT_CHECK_SLOPE, seed=3
    )

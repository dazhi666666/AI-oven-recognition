"""读取 Middlebury 的单通道浮点视差，不把彩色预览当数值。"""

from pathlib import Path

import numpy as np


def read_disparity_pfm(path: str | Path) -> np.ndarray:
    """按 Middlebury/RAFT 约定：scale 的符号决定字节序，视差直接读原始浮点数。

    该数据的 scale 可能为 -0.003922，不能把此值再乘到像素视差上。
    这不是适用于任意用途 PFM 的通用缩放约定。
    """
    with Path(path).open("rb") as stream:
        if stream.readline().strip() != b"Pf":
            raise ValueError("视差必须是单通道 Pf 格式")
        try:
            dimensions = stream.readline().split()
            if len(dimensions) != 2:
                raise ValueError
            width, height = map(int, dimensions)
            scale = float(stream.readline().strip())
        except ValueError as exc:
            raise ValueError("PFM 尺寸或缩放头无效") from exc
        if min(width, height) <= 0:
            raise ValueError("PFM 尺寸必须为正")
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("PFM 缩放头必须是有限非零数")
        payload = stream.read()
    if len(payload) != width * height * 4:
        raise ValueError("PFM 数据长度与宽高不匹配")
    dtype = "<f4" if scale < 0 else ">f4"
    image = np.frombuffer(payload, dtype=dtype).reshape(height, width)
    return np.flipud(image).astype(np.float32, copy=True)

"""探针 3：确认 SGBM 左侧无效边界是否恰为 numDisparities 列（多尺寸/多配置）。"""
import numpy as np

from stereo.config import C01_DEFAULT, C01_OVEN_RANGE, SGBMConfig
from stereo.sgbm import build_matcher
from stereo.testpatterns import render_slanted_plane_pair

configs = [
    C01_DEFAULT,
    SGBMConfig(name="ndisp32", num_disparities=32, block_size=5, p1=200, p2=800),
    SGBMConfig(name="ndisp128", num_disparities=128, block_size=5, p1=200, p2=800),
    C01_OVEN_RANGE,
]
for width in (320, 640):
    pat = render_slanted_plane_pair(width=width)
    left, right = pat["left"], pat["right"]
    for cfg in configs:
        raw = build_matcher(cfg).compute(
            np.ascontiguousarray(left), np.ascontiguousarray(right)
        )
        # 有效像素（非 -16）的最小列号：取整行一致的部分
        valid = raw != np.int16(-16)
        first_valid = int(np.max([np.argmax(v) for v in valid[::20]]))
        # 核对：前 numDisparities 列是否全无效，其后是否出现过有效
        prefix_all_invalid = bool((raw[:, : cfg.num_disparities] == -16).all())
        print(
            f"width={width:4d} ndisp={cfg.num_disparities:4d} "
            f"first_valid_col~{first_valid:4d} ==ndisp? {first_valid == cfg.num_disparities} "
            f"prefix_all_invalid={prefix_all_invalid}"
        )

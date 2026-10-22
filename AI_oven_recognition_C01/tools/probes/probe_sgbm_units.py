"""探针：核验本机 OpenCV 的 SGBM 输出 dtype 与定点缩放、无效编码。

非交付物，仅用于确定 src/stereo 的解码策略。
"""
import numpy as np
import cv2

print("cv2 version:", cv2.__version__)
print("has StereoSGBM class:", hasattr(cv2, "StereoSGBM"))
print("has StereoSGBM_create:", hasattr(cv2, "StereoSGBM_create"))

rng = np.random.default_rng(0)
h, w = 240, 320
tex = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
tex = cv2.GaussianBlur(tex, (5, 5), 0)
D = 32  # 已知常数视差
left = tex
right = np.zeros_like(tex)
right[:, D:] = tex[:, : w - D]
right[:, :D] = tex[:, :1]

builder = cv2.StereoSGBM if hasattr(cv2, "StereoSGBM") else cv2.StereoSGBM_create
kwargs = dict(
    minDisparity=0,
    numDisparities=64,
    blockSize=5,
    P1=8 * 5 * 5,
    P2=32 * 5 * 5,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM,
)
for name, kw in [("mode=SGBM", kwargs),
                 ("mode=HH", dict(kwargs, mode=cv2.STEREO_SGBM_MODE_HH)),
                 ("mode=SGBM_3WAY", dict(kwargs, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY))]:
    try:
        m = builder(**kw)
        d = m.compute(left, right)
    except Exception as e:
        print(name, "ERR", type(e).__name__, e)
        continue
    print(f"--- {name}: dtype={d.dtype} shape={d.shape} min={d.min()} max={d.max()}")
    md = d.astype(np.float32) / 16.0
    core = md[D + 20 : h - 20, D + 20 : w - 20]
    print("    /16 后有效区中位数:", float(np.median(core)),
          " 最小:", float(core.min()), " 最大:", float(core.max()))
    # 无效编码检查：SGBM 无效标记通常是 minDisparity-1 的定点形式
    inv16 = float(kwargs["minDisparity"] - 1) * 16
    print("    等于 (minDisp-1)*16 的像素数:", int((d == inv16).sum()),
          " 等于 (minDisp-1) 的像素数:", int((d == kwargs["minDisparity"] - 1).sum()))
    if d.dtype == np.int16:
        print("    int16 唯一值个数:", int(np.unique(d).size))
        print("    前5个最小唯一值:", np.unique(d)[:5].tolist())

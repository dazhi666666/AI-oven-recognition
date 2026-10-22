"""探针 2：ndisp=256 时中部列被判无效的原始值到底是什么。"""
import numpy as np

from stereo.config import C01_OVEN_RANGE
from stereo.sgbm import build_matcher
from stereo.testpatterns import render_slanted_plane_pair

pat = render_slanted_plane_pair()
left, right = pat["left"], pat["right"]
raw = build_matcher(C01_OVEN_RANGE).compute(
    np.ascontiguousarray(left), np.ascontiguousarray(right)
)
row = raw[120, 60:260].astype(np.int64)
print("raw dtype", raw.dtype, "global min/max", raw.min(), raw.max())
print("row120[60:260] uniques:", np.unique(row)[:12], "... total", np.unique(row).size)
print("count == -16 :", int((row == -16).sum()), "/", row.size)
print("count < 0    :", int((row < 0).sum()))
print("count in [0,16):", int(((row >= 0) & (row < 16)).sum()))
neg = row[row < 0]
print("negative uniques:", np.unique(neg)[:10], "count", neg.size)
# 目标真值区域
gt = pat["d_gt_px"][120, 60:260]
print("gt[60:260] range", float(gt.min()), float(gt.max()))
# 另一张图：常数平移（纹理更弱）对照
for label, r in (("row 0", raw[0]), ("row 239", raw[239]), ("row 120", raw[120])):
    print(label, "invalid count =", int((r == -16).sum()), "/", r.size)

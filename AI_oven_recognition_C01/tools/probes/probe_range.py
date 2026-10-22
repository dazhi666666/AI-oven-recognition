"""探针：同一人工图对在 ndisp=64 / 256 下的有效像素分布与解码一致性。"""
import numpy as np

from stereo.config import C01_DEFAULT, C01_OVEN_RANGE
from stereo.sgbm import build_matcher, decode_disparity
from stereo.testpatterns import render_slanted_plane_pair

pat = render_slanted_plane_pair()
left, right, gt, gt_valid = pat["left"], pat["right"], pat["d_gt_px"], pat["valid_gt"]
print("gt range:", float(np.nanmin(gt)), float(np.nanmax(gt)), "valid_gt=%", round(gt_valid.mean(), 3))

for cfg in (C01_DEFAULT, C01_OVEN_RANGE):
    m = build_matcher(cfg)
    raw = m.compute(np.ascontiguousarray(left), np.ascontiguousarray(right))
    d, valid, info = decode_disparity(raw, cfg)
    both = valid & gt_valid
    err = np.abs(d[both].astype(np.float64) - gt[both].astype(np.float64))
    print(
        f"ndisp={cfg.num_disparities:4d} raw[{raw.min()},{raw.max()}] "
        f"valid={valid.mean():.3f} valid&gt={both.mean():.3f} "
        f"err_med={np.median(err):.3f} err_p90={np.percentile(err, 90):.3f}"
    )
    row = valid[120]
    print("   row120 valid[:60]=", row[:60].sum(), "/60  60:260=", row[60:260].sum(),
          "/200  260:=", row[260:].sum(), "/60")
    print("   row120 decoded unique =", np.unique(np.round(d[120][np.isfinite(d[120])], 2)).size,
          " first invalid col =", int(np.argmin(row)) if not row.all() else -1)

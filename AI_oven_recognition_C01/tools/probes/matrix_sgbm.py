"""隔离矩阵：定位 cv2 5.0.0 下 SGBM 崩溃触发条件。"""
import itertools
import subprocess
import sys

SNIPPET = r"""
import numpy as np, cv2
h,w=240,320
p = __import__('stereo.testpatterns', fromlist=['x']).unit_check_pattern()
kw = dict(minDisparity=0, numDisparities=64, blockSize=5, P1=200, P2=800,
          disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
          speckleRange=2{extra})
ctor = cv2.{ctor}
m = ctor(**kw)
d = m.compute(np.ascontiguousarray(p['left']), np.ascontiguousarray(p['right']))
print('OK', d.dtype, d.shape)
"""

CASES = []
for ctor in ("StereoSGBM", "StereoSGBM_create"):
    for extra_name, extra in (
        ("no_mode_no_pfc", ""),
        ("preFilterCap", ", preFilterCap=63"),
        ("mode_SGBM", ", mode=cv2.STEREO_SGBM_MODE_SGBM"),
        ("mode+preFilterCap", ", mode=cv2.STEREO_SGBM_MODE_SGBM, preFilterCap=63"),
    ):
        CASES.append((ctor, extra_name, extra))

for ctor, name, extra in CASES:
    code = SNIPPET.format(ctor=ctor, extra=extra)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
    )
    status = "OK" if proc.returncode == 0 else f"EXIT={proc.returncode}"
    print(f"ctor={ctor:18s} {name:20s} -> {status} {proc.stdout.strip()[:40]}")

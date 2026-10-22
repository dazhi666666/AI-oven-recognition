"""命令行：对一对左右图计算视差（和距离），结果存到指定目录。

python -m oven_stereo.stereo --left L.png --right R.png \
    --config configs/stereo/sgbm_c01.yaml [--calib calib.json] --out results/C-01/<本次运行>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oven_stereo.contracts import RectifiedStereo
from oven_stereo.data.images import load_gray_pair

from .config import SGBMConfig
from .pipeline import estimate_depth


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SGBM 视差与距离")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--config", required=True, help="SGBM 参数 YAML")
    parser.add_argument("--calib", help="标定 JSON（字段见 RectifiedStereo）；不给则只输出视差")
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    config = SGBMConfig.from_yaml(args.config)
    calib = RectifiedStereo.from_json(args.calib) if args.calib else None
    left, right = load_gray_pair(args.left, args.right)
    result = estimate_depth(left, right, config, calib, sample_id=args.sample_id)

    out = result.save(args.out)
    (out / "config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if calib is not None:
        (out / "calibration.json").write_text(
            json.dumps(calib.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(result.summary(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""C-01 复跑入口：一次生成人工构造输入的端到端证据包。

运行（在本文件夹内）::

    cd AI_oven_recognition
    PYTHONPATH=src .venv/bin/python -m stereo.run_c01 --out outputs/stereo-c01

产出（全部标注 ``artificial_test_pattern``，**不得**作为真实场景达标声明）：

* ``inputs/``                     人工构造左右图与场景元数据（d_gt 单独存放，属参考数据）
* ``run_with_scale/``             带标定的预测运行：视差 + 深度 + 有效掩膜 + 元数据
* ``run_disparity_only/``         无标定预测运行：只输出视差（验证不编造深度）
* ``error_cases.json``            最小错误用例的实际报错与错误码
* ``determinism.json``            同输入同配置的重复性证据（数值逐位比较）
* ``reference_conversion_check.csv``  参考换算检查（手算尺度核对，非预测误差）
* ``summary.md``                  本轮交接摘要
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from .calibration import StereoCalibration, StereoCalibrationError
from .config import C01_DEFAULT, C01_OVEN_RANGE, SGBMConfig, StereoConfigError
from .pipeline import (
    PROVENANCE_ARTIFICIAL,
    StereoInputError,
    estimate_depth,
)
from .reference import handcheck_depth_m
from .sgbm import StereoUnitError
from .testpatterns import ARTIFICIAL_PROVENANCE

# 人工构造场景的"标定"：数值合理但**非真实相机**，仅用于打通换算链路
ARTIFICIAL_CALIB = dict(
    calibration_id="ARTIFICIAL-slanted-plane-v1",
    fx_px=700.0,
    baseline_m=0.1,
    cx_left=160.5,
    cx_right=158.5,  # doff = 2.0 px，用来暴露"漏掉主点差"的错误
)

CONFIGS = {"c01_default": C01_DEFAULT, "c01_oven_range": C01_OVEN_RANGE}


def _write_error_case(
    cases: List[Dict[str, Any]], name: str, expected_code: str, func
) -> None:
    try:
        func()
    except (StereoInputError, StereoConfigError, StereoCalibrationError, StereoUnitError,
            ValueError) as exc:
        cases.append(
            {
                "case": name,
                "expected_code": expected_code,
                "raised": type(exc).__name__,
                "code": getattr(exc, "code", type(exc).__name__),
                "message": str(exc),
                "passed": getattr(exc, "code", type(exc).__name__) == expected_code,
            }
        )
    except Exception as exc:  # 非预期异常也要记录，不掩盖
        cases.append(
            {
                "case": name,
                "expected_code": expected_code,
                "raised": type(exc).__name__,
                "code": "UNEXPECTED",
                "message": str(exc),
                "passed": False,
            }
        )
    else:
        cases.append(
            {
                "case": name,
                "expected_code": expected_code,
                "raised": None,
                "code": None,
                "message": "没有抛出任何异常（应当拒绝）",
                "passed": False,
            }
        )


def run(out_dir: Path, config: SGBMConfig, seed: int = 3) -> Dict[str, Any]:
    from .testpatterns import render_slanted_plane_pair

    out_dir.mkdir(parents=True, exist_ok=True)
    inp = out_dir / "inputs"
    inp.mkdir(parents=True, exist_ok=True)

    # ---------------- 1) 人工构造输入
    pattern = render_slanted_plane_pair(seed=seed)
    left_path = inp / "left_ARTIFICIAL.png"
    right_path = inp / "right_ARTIFICIAL.png"
    cv2.imwrite(str(left_path), pattern["left"])
    cv2.imwrite(str(right_path), pattern["right"])
    # 真值单独存放，明确标注为参考数据
    ref_dir = out_dir / "reference_only"
    ref_dir.mkdir(parents=True, exist_ok=True)
    np.save(ref_dir / "d_gt_px_ARTIFICIAL.npy", pattern["d_gt_px"])
    np.save(ref_dir / "valid_gt_ARTIFICIAL.npy", pattern["valid_gt"])
    (ref_dir / "README.txt").write_text(
        "本目录是人工构造的解析真值（参考数据），仅供独立换算/核验使用。\n"
        "预测链路 (stereo.pipeline.estimate_depth) 不读取本目录任何文件。\n",
        encoding="utf-8",
    )
    scene_meta = dict(pattern["meta"])
    scene_meta["calibration_ARTIFICIAL"] = ARTIFICIAL_CALIB
    scene_meta["config"] = config.to_dict()
    (inp / "scene_meta.json").write_text(
        json.dumps(scene_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    z_min = 70.0 / (ARTIFICIAL_CALIB["cx_left"] - ARTIFICIAL_CALIB["cx_right"] + 48.0 - 2.0)
    z_max = 70.0 / max(pattern["d_gt_px"].min(), 1.0)

    # ---------------- 2) 带标定运行（视差 + 深度）
    calib = StereoCalibration.from_intrinsics(
        fx_px=ARTIFICIAL_CALIB["fx_px"],
        baseline_m=ARTIFICIAL_CALIB["baseline_m"],
        cx_left=ARTIFICIAL_CALIB["cx_left"],
        cx_right=ARTIFICIAL_CALIB["cx_right"],
        calibration_id=ARTIFICIAL_CALIB["calibration_id"],
        image_size=(pattern["left"].shape[1], pattern["left"].shape[0]),
        is_rectified=True,
        extra={"note": "人工构造标定：非真实相机参数，仅用于打通换算链路"},
    )
    with_scale = estimate_depth(
        left_path,
        right_path,
        config=config,
        calibration=calib,
        provenance=PROVENANCE_ARTIFICIAL,
        depth_range_m=(z_min, z_max),
    )
    files_scale = with_scale.save(out_dir / "run_with_scale")

    # ---------------- 3) 无标定运行（disparity_only）
    disp_only = estimate_depth(
        left_path, right_path, config=config, calibration=None,
        provenance=PROVENANCE_ARTIFICIAL,
    )
    files_disp = disp_only.save(out_dir / "run_disparity_only")

    # ---------------- 4) 重复性证据（同输入同配置逐位比较）
    again = estimate_depth(
        left_path, right_path, config=config, calibration=calib,
        provenance=PROVENANCE_ARTIFICIAL, depth_range_m=(z_min, z_max),
    )
    same_disparity = np.array_equal(
        np.nan_to_num(with_scale.disparity_px, nan=-999.0),
        np.nan_to_num(again.disparity_px, nan=-999.0),
    )
    same_mask = np.array_equal(with_scale.disparity_valid_mask, again.disparity_valid_mask)
    same_raw = np.array_equal(with_scale.raw_disparity, again.raw_disparity)
    same_depth = np.array_equal(
        np.nan_to_num(np.asarray(with_scale.depth_m), nan=-999.0),
        np.nan_to_num(np.asarray(again.depth_m), nan=-999.0),
    )
    determinism = {
        "runs": 2,
        "identical_disparity_px": bool(same_disparity),
        "identical_valid_mask": bool(same_mask),
        "identical_raw_output": bool(same_raw),
        "identical_depth_m": bool(same_depth),
        "timings_ms_run1": with_scale.timings_ms,
        "timings_ms_run2": again.timings_ms,
        "note": "同一输入与配置下 SGBM 输出应完全确定；计时仅作量级参考，不作为稳定性证明",
    }
    (out_dir / "determinism.json").write_text(
        json.dumps(determinism, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------------- 5) 最小错误用例
    bad_left = out_dir / "_case_bad_left.png"
    cv2.imwrite(str(bad_left), pattern["left"][:, : pattern["left"].shape[1] - 17])
    cases: List[Dict[str, Any]] = []
    _write_error_case(
        cases, "左右图尺寸不一致", "E_SIZE_MISMATCH",
        lambda: estimate_depth(bad_left, right_path, config=config),
    )
    _write_error_case(
        cases, "图像不可读(路径不存在)", "E_READ_FAIL",
        lambda: estimate_depth(out_dir / "no_such.png", right_path, config=config),
    )
    _write_error_case(
        cases, "非法搜索参数 num_disparities=30（非16倍数）", "E_INVALID_SEARCH_PARAMS",
        lambda: estimate_depth(
            left_path, right_path,
            config=SGBMConfig(name="bad", num_disparities=30),
        ),
    )
    _write_error_case(
        cases, "非法搜索参数 blockSize 偶数", "E_INVALID_SEARCH_PARAMS",
        lambda: estimate_depth(
            left_path, right_path, config=SGBMConfig(name="bad", block_size=4)
        ),
    )
    _write_error_case(
        cases, "标定未标记已校正", "E_NOT_RECTIFIED",
        lambda: estimate_depth(
            left_path, right_path, config=config,
            calibration=StereoCalibration.from_intrinsics(
                fx_px=700.0, baseline_m=0.1, calibration_id="unrectified",
                image_size=(pattern["left"].shape[1], pattern["left"].shape[0]),
                is_rectified=False,
            ),
        ),
    )
    _write_error_case(
        cases, "标定分辨率与图像不一致", "E_CALIB_SIZE_MISMATCH",
        lambda: estimate_depth(
            left_path, right_path, config=config,
            calibration=StereoCalibration.from_intrinsics(
                fx_px=700.0, baseline_m=0.1, calibration_id="wrong-size",
                image_size=(640, 480),
            ),
        ),
    )
    _write_error_case(
        cases, "非法标定 fx<=0", "E_INVALID_CALIBRATION",
        lambda: estimate_depth(
            left_path, right_path, config=config,
            calibration=StereoCalibration.from_intrinsics(
                fx_px=-700.0, baseline_m=0.1, calibration_id="bad-fx",
                image_size=(pattern["left"].shape[1], pattern["left"].shape[0]),
            ),
        ),
    )
    _write_error_case(
        cases, "非法标定 baseline<=0", "E_INVALID_CALIBRATION",
        lambda: estimate_depth(
            left_path, right_path, config=config,
            calibration=StereoCalibration.from_intrinsics(
                fx_px=700.0, baseline_m=0.0, calibration_id="bad-baseline",
                image_size=(pattern["left"].shape[1], pattern["left"].shape[0]),
            ),
        ),
    )
    (out_dir / "error_cases.json").write_text(
        json.dumps(
            {"cases": cases, "all_passed": all(c["passed"] for c in cases)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------- 6) 参考换算检查（手算，非预测误差）
    import csv

    rows = []
    for d_px in (20.0, 25.0, 40.0, 0.0, 1.0, 2.0):
        z_hand = handcheck_depth_m(
            ARTIFICIAL_CALIB["fx_px"],
            ARTIFICIAL_CALIB["baseline_m"],
            ARTIFICIAL_CALIB["cx_left"] - ARTIFICIAL_CALIB["cx_right"],
            d_px,
        )
        rows.append(
            {
                "d_px": d_px,
                "d_minus_doff_px": d_px
                - (ARTIFICIAL_CALIB["cx_left"] - ARTIFICIAL_CALIB["cx_right"]),
                "z_hand_m": "" if z_hand != z_hand else round(z_hand, 6),
                "formula": "Z = fx*B/(d-doff), fx=700, B=0.1, doff=2.0",
                "role": "reference_conversion_check_only",
            }
        )
    with (out_dir / "reference_conversion_check.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    stats = with_scale.stats()
    summary = [
        "# C-01 端到端证据摘要（人工构造输入）",
        "",
        f"- 输入来源：`{ARTIFICIAL_PROVENANCE}`（**非真实场景**，不得作为达标声明）",
        f"- 真实场景链路是否完整：`{with_scale.is_real_scene_chain()}`",
        f"- 固定配置：{config.describe()}",
        f"- 缩放实测核验：dtype=`{with_scale.unit_report.raw_dtype}`，"
        f"dtype 先验={with_scale.unit_report.dtype_prior_scale}，"
        f"实测核验={with_scale.unit_report.verified_scale}",
        f"- 视差有效比例：{stats['disparity_valid_ratio']:.4f} "
        f"（有效 {stats['disparity_valid_pixels']} / 共 {stats['disparity_total_pixels']}）",
    ]
    if "depth_valid_ratio" in stats:
        summary += [
            f"- 深度有效比例：{stats['depth_valid_ratio']:.4f}，"
            f"深度范围 {stats.get('depth_m_min'):.3f}~{stats.get('depth_m_max'):.3f} m",
            f"- 搜索范围覆盖检查：{with_scale.search_range_check['note']}",
        ]
    summary += [
        f"- 重复性：视差逐位一致={determinism['identical_disparity_px']}，"
        f"掩膜一致={determinism['identical_valid_mask']}，"
        f"深度一致={determinism['identical_depth_m']}",
        f"- 错误用例全部按预期拒绝：{all(c['passed'] for c in cases)}",
        "",
        "## 文件",
        "",
        "| 路径 | 说明 |",
        "|---|---|",
        "| `inputs/` | 人工构造左右图与场景元数据（含构造参数） |",
        "| `reference_only/` | 解析真值（参考数据），预测链路不读取 |",
        "| `run_with_scale/` | 带标定运行：`disparity_px.npy`、`depth_m.npy`、掩膜、`run_metadata.json` |",
        "| `run_disparity_only/` | 无标定运行：只输出视差，`depth_m.npy` 不存在 |",
        "| `error_cases.json` | 最小错误用例的实际错误码与信息 |",
        "| `determinism.json` | 重复运行一致性证据 |",
        "| `reference_conversion_check.csv` | 手算换算检查（参考换算，非预测误差） |",
        "",
        "## 关于误差",
        "",
        "本包**不**给出精度达标结论：误差统计应由 D 的独立工具在留出数据上完成。",
        "本包只保证：单位正确、接口隔离、数值可复跑、无效区域显式可见。",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    # 清理临时文件
    bad_left.unlink(missing_ok=True)

    return {
        "out_dir": str(out_dir),
        "files_with_scale": files_scale,
        "files_disparity_only": files_disp,
        "stats": stats,
        "determinism": determinism,
        "error_cases_all_passed": all(c["passed"] for c in cases),
    }


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="C-01 复跑：生成人工构造证据包")
    parser.add_argument("--out", default="outputs/stereo-c01", help="输出目录")
    parser.add_argument("--config", default="c01_default", choices=sorted(CONFIGS))
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args(argv)

    result = run(Path(args.out), CONFIGS[args.config], seed=args.seed)
    print(json.dumps(
        {k: v for k, v in result.items() if k not in ("files_with_scale", "files_disparity_only")},
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

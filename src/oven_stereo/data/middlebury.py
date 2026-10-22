"""B-01：输入与答案分别读取，准备一个原分辨率 Middlebury 场景包。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from oven_stereo.contracts import ReferenceData, StereoSample

from .calibration import read_middlebury_calibration
from .images import load_gray_pair
from .manifest import acquire_assets, asset_path, read_manifest, verify_asset
from .pfm import read_disparity_pfm


def load_middlebury_sample(manifest_path: str | Path, root: str | Path = ".") -> StereoSample:
    """只访问左右图和标定；删除整个答案目录后本函数仍能运行。"""
    meta = read_manifest(manifest_path)
    required = {
        "sample_id", "left_path", "right_path", "image_size_px", "source", "source_version",
        "domain", "split", "conditions", "rectified", "calibration_id", "calibration_path",
        "metric_depth_available",
    }
    if missing := required - meta.keys():
        raise ValueError(f"输入清单缺少字段：{sorted(missing)}")
    if meta["source"] != "Middlebury" or meta["source_version"] != "2014/perfect":
        raise ValueError("本读取器仅处理 Middlebury 2014/perfect 已校正场景")
    if meta["rectified"] is not True or meta["metric_depth_available"] is not True:
        raise ValueError("场景必须已校正且有完整米制标定")
    if meta["domain"] != "public" or meta["split"] not in {"development", "test"}:
        raise ValueError("公开场景的 domain 或 split 不合法")
    if not isinstance(meta["conditions"], dict) or not meta["sample_id"]:
        raise ValueError("conditions 必须是字典且 sample_id 不能为空")
    left_path = asset_path(root, meta["left_path"])
    right_path = asset_path(root, meta["right_path"])
    if left_path == right_path:
        raise ValueError("左右图不能指向同一文件")
    left, right = load_gray_pair(left_path, right_path)
    size = (left.shape[1], left.shape[0])
    if list(size) != meta["image_size_px"]:
        raise ValueError("图像尺寸与输入清单不一致")
    calib = read_middlebury_calibration(asset_path(root, meta["calibration_path"]), meta["calibration_id"])
    calib.stereo.validate(size)
    return StereoSample(
        **{key: meta[key] for key in (
            "sample_id", "source", "source_version", "domain", "split", "conditions",
            "rectified", "calibration_id", "metric_depth_available",
        )},
        left_path=left_path, right_path=right_path, image_size_px=size,
        calibration=calib.stereo,
        calibration_metadata={"cam0": calib.cam0.tolist(), "cam1": calib.cam1.tolist(), "ndisp": calib.ndisp},
    )


def load_middlebury_reference(
    manifest_path: str | Path, sample: StereoSample, root: str | Path = "."
) -> ReferenceData:
    """参考深度独立换算，不调用 C 的算法或换算函数。"""
    meta = read_manifest(manifest_path)
    required = {"sample_id", "calibration_id", "disparity_path", "disparity_units", "depth_units", "source_url"}
    if missing := required - meta.keys():
        raise ValueError(f"答案清单缺少字段：{sorted(missing)}")
    if meta["sample_id"] != sample.sample_id or meta["calibration_id"] != sample.calibration_id:
        raise ValueError("答案与样本或标定版本不匹配")
    if meta["disparity_units"] != "px" or meta["depth_units"] != "m":
        raise ValueError("答案单位必须是像素和米")
    disparity = read_disparity_pfm(asset_path(root, meta["disparity_path"]))
    if disparity.shape != sample.image_size_px[::-1]:
        raise ValueError("答案视差尺寸与输入不一致")
    valid = np.isfinite(disparity)
    disparity[~valid] = np.nan
    denominator = disparity.astype(np.float64) + sample.calibration.doffs_px
    depth_valid = valid & (denominator > 0)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    # 输入已在 read_middlebury_calibration 中把毫米转换为米。
    depth[depth_valid] = (
        sample.calibration.f_px * sample.calibration.baseline_m / denominator[depth_valid]
    )
    depth_valid &= np.isfinite(depth) & (depth > 0)
    depth[~depth_valid] = np.nan
    return ReferenceData(sample.sample_id, disparity, valid, depth, depth_valid, meta["source_url"])


def prepare_package(
    manifest_path: str | Path, reference_manifest: str | Path | None, out: str | Path,
    root: str | Path = ".", download: bool = False,
) -> dict:
    manifests = [read_manifest(manifest_path)]
    if reference_manifest is not None:
        manifests.append(read_manifest(reference_manifest))
    for manifest in manifests:
        if download:
            acquire_assets(manifest, root)
        else:
            for asset in manifest["assets"]:
                verify_asset(asset_path(root, asset["path"]), asset)
    sample = load_middlebury_sample(manifest_path, root)
    reference = None if reference_manifest is None else load_middlebury_reference(reference_manifest, sample, root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "inputs"
    inputs.mkdir()
    # 输入清单无标准答案路径；图像复用原始文件，不重新编码。
    input_meta = {key: value for key, value in manifests[0].items() if key != "assets"}
    input_meta["calibration"] = {**sample.calibration.to_dict(), **sample.calibration_metadata}
    (inputs / "sample.json").write_text(json.dumps(input_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (inputs / "calibration.json").write_text(json.dumps(sample.calibration.to_dict(), indent=2), encoding="utf-8")
    summary = {
        "sample_id": sample.sample_id, "image_size_px": list(sample.image_size_px),
        "baseline_m": sample.calibration.baseline_m, "f_px": sample.calibration.f_px,
        "doffs_px": sample.calibration.doffs_px, "resolution_profile": "native",
        "reference_prepared": reference is not None,
    }
    if reference is not None:
        answers = out / "reference"
        answers.mkdir()
        for name in ("disparity_px", "disparity_valid_mask", "depth_m", "depth_valid_mask"):
            np.save(answers / f"{name}.npy", getattr(reference, name), allow_pickle=False)
        (answers / "reference.json").write_text(json.dumps(manifests[1], ensure_ascii=False, indent=2), encoding="utf-8")
        summary["reference_valid_ratio"] = float(reference.disparity_valid_mask.mean())
    (out / "preparation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="准备 B-01 原分辨率场景包；答案通过单独参数显式提供")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference-manifest")
    parser.add_argument("--root", default=".", help="清单中的路径相对于此仓库根目录")
    parser.add_argument("--out", required=True, help="新的本地输出目录，不覆盖旧结果")
    parser.add_argument("--download", action="store_true", help="仅下载清单中缺少的文件，并核验 SHA256")
    args = parser.parse_args(argv)
    summary = prepare_package(args.manifest, args.reference_manifest, args.out, args.root, args.download)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

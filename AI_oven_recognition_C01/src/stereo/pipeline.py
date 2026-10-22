"""C-01 预测链路：左右图 + 标定 + 固定配置 → 数值视差/深度 + 有效区域。

严格边界（任务书 "最容易越界" 第 2、3 条，C-01 步骤 1、4、6、7）：

* **预测入口不接收真值。** :func:`estimate_depth` 的签名里没有任何 GT/参考参数，
  也不读取任何参考文件；本模块**不导入** :mod:`stereo.reference`。
* 输入只做校验与只读使用：尺寸一致、8 位灰度、可读；不修改 B 的标定。
* 输出保留**数值**（``.npy``）与**显式有效掩膜**，不只给彩色图；无效像素固定 ``NaN``。
* 缺失标定 → 返回 ``disparity_only``，不编造深度。
* 每次运行记录输入 ID、哈希、参数、版本、方法与分段计时。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .version import METHOD as METHOD_DESCRIPTION
from .version import __version__ as STEREO_PKG_VERSION
from .calibration import StereoCalibration, StereoCalibrationError
from .config import (
    C01_DEFAULT,
    SGBMConfig,
    StereoConfigError,
    check_search_range_covers,
    left_border_invalid_columns,
)
from .sgbm import (
    DisparityUnitReport,
    StereoUnitError,
    compute_raw_disparity,
    decode_disparity,
    verify_disparity_units,
)

__all__ = [
    "PROVENANCE_ARTIFICIAL",
    "PROVENANCE_B_PROVIDED",
    "StereoInputError",
    "StereoRunResult",
    "estimate_depth",
    "read_gray_pair",
]

PROVENANCE_ARTIFICIAL = "artificial_test_pattern"
PROVENANCE_B_PROVIDED = "b_provided_rectified"

_ALLOWED_PROVENANCE = (PROVENANCE_ARTIFICIAL, PROVENANCE_B_PROVIDED)

# 有效比例过低时的提示阈值（仅提示，不改变输出）
LOW_VALID_RATIO_WARN = 0.10


class StereoInputError(RuntimeError):
    """输入不满足链路前提（按场景给出稳定错误码）。"""

    def __init__(self, message: str, code: str = "E_INVALID_INPUT") -> None:
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------- 输入
def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _to_gray8(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {}
    if image.ndim == 2:
        gray = image
        info["converted_to_gray"] = False
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        info["converted_to_gray"] = True
        info["color_conversion"] = "cv2.COLOR_BGR2GRAY (888 权重，已记录以便复跑)"
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        info["converted_to_gray"] = True
        info["color_conversion"] = "cv2.COLOR_BGRA2GRAY"
    else:
        raise StereoInputError(
            f"无法解释的图像形状 {image.shape}，仅支持单通道或 3/4 通道 8 位图",
            code="E_UNSUPPORTED_CHANNELS",
        )
    if gray.dtype != np.uint8:
        raise StereoInputError(
            f"本轮只支持 8 位灰度输入，收到 {gray.dtype}；"
            "16 位图的位深缩放会改变视差尺度，需先与 B 确认约定再扩展",
            code="E_UNSUPPORTED_DEPTH",
        )
    info["gray_dtype"] = str(gray.dtype)
    return np.ascontiguousarray(gray), info


def _read_one(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not path.exists():
        raise StereoInputError(f"图像不存在：{path}", code="E_READ_FAIL")
    if path.is_dir():
        raise StereoInputError(f"输入是目录而非图像文件：{path}", code="E_READ_FAIL")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise StereoInputError(
            f"图像不可读或格式不支持：{path}（cv2.imread 返回 None）",
            code="E_READ_FAIL",
        )
    gray, info = _to_gray8(image)
    info.update(
        {
            "path": str(path),
            "sha256": _sha256_file(path),
            "original_shape": list(image.shape),
            "original_dtype": str(image.dtype),
            "size_wh": [int(gray.shape[1]), int(gray.shape[0])],
        }
    )
    return gray, info


def read_gray_pair(
    left_path: os.PathLike, right_path: os.PathLike
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """读取并校验左右图（顺序按调用约定：left 在前，right 在后）。"""
    left_path = Path(left_path)
    right_path = Path(right_path)
    left, left_info = _read_one(left_path)
    right, right_info = _read_one(right_path)
    if left.shape != right.shape:
        raise StereoInputError(
            f"左右图尺寸不一致：left {left.shape} vs right {right.shape}；"
            "未校正/裁切不一致时不得直接匹配",
            code="E_SIZE_MISMATCH",
        )
    meta = {
        "left": left_info,
        "right": right_info,
        "size_wh": left_info["size_wh"],
        "shape_hw": [int(left.shape[0]), int(left.shape[1])],
        "side_convention": "参数1=左图(left)，参数2=右图(right)；不做自动翻转猜测",
    }
    return left, right, meta


# --------------------------------------------------------------------- 结果
@dataclass
class StereoRunResult:
    """一次预测运行的完整结果（预测产物；不含任何参考/真值）。"""

    mode: str  # "disparity_only" | "disparity_and_depth"
    input_id: str
    provenance: str
    config: SGBMConfig
    unit_report: DisparityUnitReport
    disparity_px: np.ndarray
    disparity_valid_mask: np.ndarray
    raw_disparity: np.ndarray
    input_meta: Dict[str, Any]
    timings_ms: Dict[str, float]
    decode_info: Dict[str, Any]
    depth_m: Optional[np.ndarray] = None
    depth_valid_mask: Optional[np.ndarray] = None
    calibration: Optional[StereoCalibration] = None
    search_range_check: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    package_version: str = STEREO_PKG_VERSION

    # --------------------------------------------------------------- 统计
    def stats(self) -> Dict[str, Any]:
        total = int(self.disparity_valid_mask.size)
        valid = int(self.disparity_valid_mask.sum())
        out: Dict[str, Any] = {
            "disparity_total_pixels": total,
            "disparity_valid_pixels": valid,
            "disparity_invalid_pixels": total - valid,
            "disparity_valid_ratio": (valid / total) if total else 0.0,
            # 左边界代价：SGBM 左起 numDisparities 列固定无效（实测规律，见 config）
            "left_border_invalid_columns": left_border_invalid_columns(self.config),
            "theoretical_max_valid_ratio": (
                max(0.0, 1.0 - left_border_invalid_columns(self.config) / width)
                if (width := self.disparity_valid_mask.shape[1]) else 0.0
            ),
        }
        d = self.disparity_px[self.disparity_valid_mask]
        if d.size:
            out["disparity_px_min"] = float(np.min(d))
            out["disparity_px_max"] = float(np.max(d))
            out["disparity_px_median"] = float(np.median(d))
        if self.depth_m is not None and self.depth_valid_mask is not None:
            dt = int(self.depth_valid_mask.size)
            dv = int(self.depth_valid_mask.sum())
            out.update(
                {
                    "depth_total_pixels": dt,
                    "depth_valid_pixels": dv,
                    "depth_invalid_pixels": dt - dv,
                    "depth_valid_ratio": (dv / dt) if dt else 0.0,
                }
            )
            z = self.depth_m[self.depth_valid_mask]
            if z.size:
                out["depth_m_min"] = float(np.min(z))
                out["depth_m_max"] = float(np.max(z))
                out["depth_m_median"] = float(np.median(z))
        return out

    def is_real_scene_chain(self) -> bool:
        """只有 B 提供的已核验已校正真实图对才返回 True。"""
        return self.provenance == PROVENANCE_B_PROVIDED

    # --------------------------------------------------------------- 元数据
    def metadata(self) -> Dict[str, Any]:
        return {
            "module": "stereo (role C)",
            "stage": "C-01",
            "package_version": self.package_version,
            "mode": self.mode,
            "input_id": self.input_id,
            "input_provenance": self.provenance,
            "real_scene_chain_complete": self.is_real_scene_chain(),
            "claim_scope": (
                "真实场景链路" if self.is_real_scene_chain() else
                "人工构造输入：仅验证链路与单位换算，不得作为真实场景或精度达标声明"
            ),
            "matcher": "cv2.StereoSGBM (OpenCV 现成实现，未自写 SGM)",
            "method": METHOD_DESCRIPTION,
            "config": self.config.to_dict(),
            "config_describe": self.config.describe(),
            "disparity_units": self.unit_report.to_dict(),
            "decode": self.decode_info,
            "search_range_check": self.search_range_check,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "depth_model": (
                self.calibration.to_dict()["depth_model"] if self.calibration else None
            ),
            "input": self.input_meta,
            "timings_ms": self.timings_ms,
            "stats": self.stats(),
            "warnings": list(self.warnings),
            "environments": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "platform": platform.platform(),
            },
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reproduce": (
                "PYTHONPATH=src python -m stereo.run_c01 --out outputs/stereo-c01"
            ),
        }

    # ----------------------------------------------------------------- 落盘
    def save(self, out_dir: os.PathLike) -> Dict[str, str]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: Dict[str, str] = {}

        def dump_npy(name: str, arr: np.ndarray) -> None:
            path = out / name
            np.save(path, arr)
            written[name] = str(path)

        def dump_png_mask(name: str, mask: np.ndarray) -> None:
            path = out / name
            cv2.imwrite(str(path), (np.asarray(mask, dtype=np.uint8) * 255))
            written[name] = str(path)

        # 数值输出：原始输出 + 解码视差（NaN 编码无效）+ 显式掩膜
        dump_npy("disparity_raw.npy", self.raw_disparity)
        dump_npy("disparity_px.npy", self.disparity_px)
        dump_npy("disparity_valid_mask.npy", self.disparity_valid_mask.astype(np.uint8))
        dump_png_mask("disparity_valid_mask.png", self.disparity_valid_mask)
        cv2.imwrite(str(out / "disparity_vis.png"), self.visualize_disparity())

        if self.depth_m is not None and self.depth_valid_mask is not None:
            dump_npy("depth_m.npy", self.depth_m)
            dump_npy("depth_valid_mask.npy", self.depth_valid_mask.astype(np.uint8))
            dump_png_mask("depth_valid_mask.png", self.depth_valid_mask)
            cv2.imwrite(str(out / "depth_vis.png"), self.visualize_depth())

        meta_path = out / "run_metadata.json"
        meta_path.write_text(
            json.dumps(self.metadata(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written["run_metadata.json"] = str(meta_path)

        calib_path = out / "calibration.json"
        calib_path.write_text(
            json.dumps(
                self.calibration.to_dict() if self.calibration else {"status": "absent"},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        written["calibration.json"] = str(calib_path)

        cfg_path = out / "config.json"
        cfg_path.write_text(
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written["config.json"] = str(cfg_path)

        note_path = out / "PROVENANCE.txt"
        note_path.write_text(
            "\n".join(
                [
                    "本目录由 stereo 模块（角色 C）C-01 链路生成。",
                    f"输入来源标记：{self.provenance}",
                    f"真实场景链路是否完整：{self.is_real_scene_chain()}",
                    "disparity_px.npy / depth_m.npy 中 NaN = 无效像素；",
                    "有效位置以 *_valid_mask 为准，未做任何静默补全。",
                    "预测链路未读取任何真值/参考数据（参考换算另见 stereo.reference）。",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        written["PROVENANCE.txt"] = str(note_path)
        return written

    # --------------------------------------------------------------- 可视化
    def visualize_disparity(self) -> np.ndarray:
        lo = float(self.config.min_disparity)
        hi = float(self.config.min_disparity + self.config.num_disparities - 1)
        d = np.nan_to_num(self.disparity_px, nan=lo)
        norm = np.clip((d - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis[~self.disparity_valid_mask] = (0, 0, 0)
        return vis

    def visualize_depth(self) -> np.ndarray:
        if self.depth_m is None or self.depth_valid_mask is None:
            raise StereoInputError("本次运行无深度输出，不能可视化深度", code="E_NO_DEPTH")
        z = self.depth_m
        valid = self.depth_valid_mask
        zmin = float(np.min(z[valid])) if valid.any() else 0.0
        zmax = float(np.max(z[valid])) if valid.any() else 1.0
        norm = np.clip((np.nan_to_num(z, nan=zmax) - zmin) / max(zmax - zmin, 1e-9), 0, 1)
        vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis[~valid] = (0, 0, 0)
        return vis


# --------------------------------------------------------------------- 入口
def estimate_depth(
    left_path: os.PathLike,
    right_path: os.PathLike,
    config: SGBMConfig = C01_DEFAULT,
    calibration: Optional[StereoCalibration] = None,
    input_id: Optional[str] = None,
    provenance: str = PROVENANCE_ARTIFICIAL,
    depth_range_m: Optional[Sequence[float]] = None,
) -> StereoRunResult:
    """预测入口：左右图(+可选已核验标定) → 数值视差/深度 + 有效掩膜。

    本函数**只接收**图像、配置与标定；不接收、不查找、不读取任何真值或参考视差。

    Parameters
    ----------
    depth_range_m:
        可选的感兴趣轴向深度区间 ``(z_min, z_max)``（米）。仅与标定一起使用时，
        用于**报告**当前搜索范围是否覆盖该区间；不改变任何参数。
    provenance:
        ``"artificial_test_pattern"``（默认，本轮无 B 数据）或 ``"b_provided_rectified"``。
    """
    if provenance not in _ALLOWED_PROVENANCE:
        raise StereoInputError(
            f"provenance 只能是 {_ALLOWED_PROVENANCE}，收到 {provenance!r}",
            code="E_BAD_PROVENANCE",
        )
    config.validate()
    warnings: List[str] = []

    t0 = time.perf_counter()
    # 1) 定点缩放实测核验（以本机实际 OpenCV 输出为准）
    t = time.perf_counter()
    unit_report = verify_disparity_units(config)
    t_unit = (time.perf_counter() - t) * 1e3

    # 2) 输入校验
    t = time.perf_counter()
    left, right, input_meta = read_gray_pair(left_path, right_path)
    t_io = (time.perf_counter() - t) * 1e3

    # 3) 匹配（原样输出）
    t = time.perf_counter()
    raw = compute_raw_disparity(left, right, config)
    t_compute = (time.perf_counter() - t) * 1e3

    # 4) 解码（按实测核验后的缩放）
    t = time.perf_counter()
    disparity_px, valid_mask, decode_info = decode_disparity(
        raw, config, scale=unit_report.verified_scale
    )
    decode_info["scale_verified_by_api_probe"] = True
    t_decode = (time.perf_counter() - t) * 1e3

    input_id = input_id or (
        f"{input_meta['left']['sha256'][:8]}-{input_meta['right']['sha256'][:8]}"
    )

    result = StereoRunResult(
        mode="disparity_only",
        input_id=input_id,
        provenance=provenance,
        config=config,
        unit_report=unit_report,
        disparity_px=disparity_px,
        disparity_valid_mask=valid_mask,
        raw_disparity=raw,
        input_meta=input_meta,
        timings_ms={},
        decode_info=decode_info,
        warnings=warnings,
    )

    # 5) 可选深度换算（标定缺失则明确返回 disparity_only，不编造深度）
    t = time.perf_counter()
    if calibration is not None:
        calibration.validate(expected_image_size=(left.shape[1], left.shape[0]))
        depth_m, depth_valid = calibration.depth_from_disparity(
            disparity_px, valid_mask=valid_mask
        )
        result.depth_m = depth_m
        result.depth_valid_mask = depth_valid
        result.calibration = calibration
        result.mode = "disparity_and_depth"
        if depth_range_m is not None:
            if len(tuple(depth_range_m)) != 2:
                raise StereoInputError(
                    "depth_range_m 需要两个值 (z_min, z_max)", code="E_BAD_DEPTH_RANGE"
                )
            z_min, z_max = float(depth_range_m[0]), float(depth_range_m[1])
            result.search_range_check = check_search_range_covers(
                config,
                calibration.fx_px,
                calibration.baseline_m,
                z_max,
                z_min,
                calibration.doff_px,
            )
            if not result.search_range_check["covered"]:
                warnings.append(
                    "视差搜索范围未完整覆盖目标深度区间，相关像素会成为无效值；"
                    "本模块只报告，不自动改配置"
                )
    else:
        warnings.append(
            "未提供标定：按约定返回 disparity_only，未生成任何深度数值"
        )
    t_depth = (time.perf_counter() - t) * 1e3

    if decode_info["valid_ratio"] < LOW_VALID_RATIO_WARN:
        warnings.append(
            f"视差有效比例偏低（{decode_info['valid_ratio']:.3f}），"
            "请结合失败位置定位原因，不要用填满作为成功条件"
        )
    if provenance != PROVENANCE_B_PROVIDED:
        warnings.append(
            "本轮为人工构造输入：链路与单位换算已核验，但**不得**声称真实链路完成或精度达标"
        )

    result.timings_ms = {
        "unit_verification": t_unit,
        "read_pair": t_io,
        "sgbm_compute": t_compute,
        "decode": t_decode,
        "depth_conversion": t_depth,
        "total": (time.perf_counter() - t0) * 1e3,
    }
    return result

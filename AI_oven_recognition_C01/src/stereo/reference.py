"""**参考换算与误差核验**（与预测链路严格分离）。

任务书 C-01 步骤 6：

* 参考视差只允许出现在**独立的换算/核验**环节，用于手算核对与误差统计；
* 参考包**不得**传入预测函数（:func:`stereo.pipeline.estimate_depth` 的签名里没有它）；
* 不得生成冒充预测的输出。

纪律（由 ``tests/stereo/test_isolation.py`` 自动检查）：

* 本模块**不读取图像**、不做匹配、不产生任何"预测"；
* 输入永远是**已经给出的数值视差数组**，绝不是图像路径；
* :mod:`stereo.pipeline` 不导入本模块。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "MODULE_ROLE",
    "handcheck_depth_m",
    "handcheck_disparity_px",
    "reference_error_metrics",
    "depth_from_reference_disparity",
]

MODULE_ROLE = (
    "reference-check only: 参考换算/误差核验；不得被预测链路导入，不得用于生成预测输出"
)


# ------------------------------------------------------------- 手算核对用
def handcheck_depth_m(fx_px: float, baseline_m: float, doff_px: float, d_px: float) -> float:
    """纯标量手算换算（独立实现，供人工核对与单元测试交叉验证）。

    ``Z = fx * B / (d - doff)``；``d - doff <= 0`` 返回 ``nan``。
    """
    if not (fx_px > 0 and baseline_m > 0):
        raise ValueError("fx_px 与 baseline_m 必须为正")
    denom = d_px - doff_px
    if denom <= 0:
        return float("nan")
    return fx_px * baseline_m / denom


def handcheck_disparity_px(
    fx_px: float, baseline_m: float, doff_px: float, z_m: float
) -> float:
    """逆换算 ``d = doff + fx * B / Z``（手算核对用）。"""
    if not (fx_px > 0 and baseline_m > 0):
        raise ValueError("fx_px 与 baseline_m 必须为正")
    if not (math.isfinite(z_m) and z_m > 0):
        return float("nan")
    return doff_px + fx_px * baseline_m / z_m


def depth_from_reference_disparity(
    reference_disparity_px: np.ndarray, fx_px: float, baseline_m: float, doff_px: float
) -> Tuple[np.ndarray, np.ndarray]:
    """把**参考**视差换算成深度，用于独立核验换算链路（不产生预测）。

    这是"参考换算检查"：它接收参考视差数值，输出不是预测结果，
    因此在任何交付说明中必须标明来源为参考，不能当作算法输出。
    """
    d = np.asarray(reference_disparity_px, dtype=np.float64)
    denom = d - float(doff_px)
    valid = np.isfinite(denom) & (denom > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = fx_px * baseline_m / denom
    z = np.where(valid, z, np.nan).astype(np.float32)
    return z, valid


# ------------------------------------------------------------- 误差核验
def reference_error_metrics(
    prediction_px: np.ndarray,
    reference_px: np.ndarray,
    eval_mask: Optional[np.ndarray] = None,
    bad_thresholds: Sequence[float] = (1.0, 2.0),
) -> Dict[str, Any]:
    """在**显式**评估掩膜上比较预测与参考（缺省只用两者共同有效像素）。

    覆盖率与误差**同时**报告（任务书"最容易越界"第 3 条）：收紧掩膜会同时改变两者，
    阈值由调用方给定并写入结果，禁止事后偷改。
    """
    pred = np.asarray(prediction_px, dtype=np.float64)
    ref = np.asarray(reference_px, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError(f"预测与参考形状不一致：{pred.shape} vs {ref.shape}")
    pred_valid = np.isfinite(pred)
    ref_valid = np.isfinite(ref)
    common = pred_valid & ref_valid
    mask = common if eval_mask is None else (common & np.asarray(eval_mask, dtype=bool))
    n_eval = int(mask.sum())
    total = int(mask.size)

    out: Dict[str, Any] = {
        "role": MODULE_ROLE,
        "eval_pixels": n_eval,
        "total_pixels": total,
        "eval_coverage": (n_eval / total) if total else 0.0,
        "prediction_coverage": (int(pred_valid.sum()) / total) if total else 0.0,
        "reference_coverage": (int(ref_valid.sum()) / total) if total else 0.0,
        "bad_thresholds_px": [float(t) for t in bad_thresholds],
        "eval_mask_source": "explicit" if eval_mask is not None else "intersection_of_valid",
    }
    if n_eval == 0:
        out.update({"note": "无共同有效像素，误差指标不可计算（不以填满为成功条件）"})
        return out

    err = pred[mask] - ref[mask]
    abs_err = np.abs(err)
    out.update(
        {
            "mae_px": float(abs_err.mean()),
            "rmse_px": float(np.sqrt(np.mean(err**2))),
            "median_abs_err_px": float(np.median(abs_err)),
            "p90_abs_err_px": float(np.percentile(abs_err, 90)),
            "bias_px": float(err.mean()),
            "max_abs_err_px": float(abs_err.max()),
        }
    )
    for t in bad_thresholds:
        key = f"bad{('%g' % t).replace('.', '_')}"
        out[key] = float((abs_err > t).mean())
    return out

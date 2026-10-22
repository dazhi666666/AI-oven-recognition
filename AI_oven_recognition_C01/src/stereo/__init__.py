"""C-01 双目匹配、米制深度与有效区域（角色 C，第一轮唯一授权任务）。

模块边界（见任务书《C：双目匹配、米制深度与可信区域》）：

    左右图 + 已核验标定 + 固定配置  ->  数值视差/深度 + 有效区域

本包**不**负责食品体积、平台设计、数据集标定修改或评测口径定义。
预测入口 :func:`stereo.pipeline.estimate_depth` 不接收任何真值/参考参数；
参考换算与误差核验在独立的 :mod:`stereo.reference` 中，禁止被预测链路导入。

注意：`stereo.testpatterns` 只是**API 解码自检**与**人工构造测试**用的图样生成器，
不是数据集，也不属于后续 C-04 的 `src/synthetic_stereo`。
"""

from .calibration import StereoCalibration, StereoCalibrationError
from .config import (
    C01_DEFAULT,
    C01_OVEN_RANGE,
    SGBMConfig,
    StereoConfigError,
    check_search_range_covers,
    left_border_invalid_columns,
    required_disparity_range,
)
from .pipeline import (
    PROVENANCE_ARTIFICIAL,
    PROVENANCE_B_PROVIDED,
    StereoInputError,
    StereoRunResult,
    estimate_depth,
)
from .sgbm import (
    DisparityUnitReport,
    StereoUnitError,
    compute_raw_disparity,
    decode_disparity,
    verify_disparity_units,
)
from .version import METHOD, __version__

__all__ = [
    "C01_DEFAULT",
    "C01_OVEN_RANGE",
    "PROVENANCE_ARTIFICIAL",
    "PROVENANCE_B_PROVIDED",
    "DisparityUnitReport",
    "SGBMConfig",
    "StereoCalibration",
    "StereoCalibrationError",
    "StereoConfigError",
    "StereoInputError",
    "StereoRunResult",
    "StereoUnitError",
    "compute_raw_disparity",
    "decode_disparity",
    "estimate_depth",
    "left_border_invalid_columns",
    "required_disparity_range",
    "verify_disparity_units",
    "__version__",
]

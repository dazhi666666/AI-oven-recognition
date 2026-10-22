"""模块之间交换的数据格式：样本、深度结果、几何结果、标定健康状态、资源记录。"""

from .calibration import CalibrationError, RectifiedStereo
from .geometry import GeometryResult
from .results import DepthResult
from .sample import ReferenceData, StereoSample

__all__ = [
    "CalibrationError", "DepthResult", "GeometryResult", "RectifiedStereo",
    "ReferenceData", "StereoSample",
]

"""模块之间交换的数据格式：样本、深度结果、几何结果、标定健康状态、资源记录。"""

from .calibration import CalibrationError, RectifiedStereo
from .results import DepthResult

__all__ = ["CalibrationError", "DepthResult", "RectifiedStereo"]

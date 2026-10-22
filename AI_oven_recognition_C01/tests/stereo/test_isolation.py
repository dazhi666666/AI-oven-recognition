"""预测链路与参考/真值的**结构性隔离**检查。

这些测试对应任务书 C-01 步骤 6 与"最容易越界"第 2 条：
不是靠人工承诺，而是靠源码与签名检查来保证预测路径拿不到真值。
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

from stereo.pipeline import estimate_depth

pytestmark = pytest.mark.unit

SRC = Path(__file__).resolve().parents[2] / "src" / "stereo"


def _source(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8")


def _tree(name: str) -> ast.Module:
    return ast.parse(_source(name), filename=name)


def _imported_modules(tree: ast.Module) -> list[str]:
    """收集 ``import x`` / ``from x import y`` 的目标模块名（含相对导入层级）。"""
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            mods.append("." * node.level + (node.module or ""))
    return mods


def _called_names(tree: ast.Module) -> list[str]:
    """收集被调用的函数/属性名（``f()`` → f，``m.f()`` → f）。"""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                names.append(func.attr)
    return names


def test_pipeline_does_not_import_reference_module():
    tree = _tree("pipeline.py")
    for mod in _imported_modules(tree):
        assert "reference" not in mod, f"预测链路导入了参考模块：{mod}"


def test_pipeline_never_reads_reference_or_gt_files():
    tree = _tree("pipeline.py")
    banned = ("d_gt", "reference_only", "ground_truth", "valid_gt", "_truth")
    # 只看字符串常量与文件名常量，避免把注释/文档里的解释性文字误判为读取行为
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            low = node.value.lower()
            for token in banned:
                assert token not in low, f"预测链路出现参考/真值字样：{token}"
        if isinstance(node, ast.Name):
            assert node.id not in banned, f"预测链路出现参考/真值变量：{node.id}"


def test_pipeline_does_not_load_opencv_reference_kernels():
    """预测链路只走 SGBM 封装；不得在 pipeline 内直接调用 OpenCV 匹配/读图 API。"""
    tree = _tree("pipeline.py")
    for name in _called_names(tree):
        assert name not in {
            "StereoSGBM",
            "StereoSGBM_create",
            "StereoBM_create",
            "findFundamentalMat",
        }, f"pipeline 内不应直接调用：{name}"


def test_estimate_depth_signature_has_no_gt_parameter():
    params = set(inspect.signature(estimate_depth).parameters)
    assert params == {
        "left_path",
        "right_path",
        "config",
        "calibration",
        "input_id",
        "provenance",
        "depth_range_m",
    }
    for name in params:
        assert not re.search(r"gt|truth|reference|label|answer", name, re.I), name


def test_reference_module_is_pure_and_array_only():
    tree = _tree("reference.py")
    mods = _imported_modules(tree)
    for banned in ("cv2", "PIL", "stereo.pipeline", ".pipeline", "stereo.sgbm", ".sgbm"):
        assert banned not in mods, f"参考模块不应导入：{banned}"
    called = _called_names(tree)
    for banned in ("imread", "StereoSGBM", "StereoSGBM_create", "estimate_depth"):
        assert banned not in called, f"参考模块不应调用：{banned}"
    src = _source("reference.py")
    # 只声明手算/指标类接口
    assert "MODULE_ROLE" in src
    assert "不得被预测链路导入" in src


def test_package_init_does_not_export_reference():
    import stereo

    assert "reference" not in getattr(stereo, "__all__", [])
    assert "estimate_depth" in stereo.__all__


def test_reference_metrics_signature_takes_arrays_not_paths():
    from stereo import reference as ref

    params = list(inspect.signature(ref.reference_error_metrics).parameters)
    assert params[:2] == ["prediction_px", "reference_px"]
    src = _source("reference.py")
    assert "os.PathLike" not in src


def test_testpatterns_are_marked_artificial():
    from stereo import testpatterns

    pattern = testpatterns.unit_check_pattern()
    assert pattern["meta"]["provenance"] == testpatterns.ARTIFICIAL_PROVENANCE
    assert "人工构造" in pattern["meta"]["note"]
    assert pattern["meta"]["kind"] == "slanted_plane"


def test_no_synthetic_stereo_package_yet():
    """C-04 尚未授权：src/synthetic_stereo 不得提前出现。"""
    root = Path(__file__).resolve().parents[2] / "src"
    assert not (root / "synthetic_stereo").exists()

"""检查包结构完整、各子模块可以导入。"""

import importlib

import pytest

SUBMODULES = [
    "contracts",
    "data",
    "synthetic",
    "calibration",
    "stereo",
    "geometry",
    "evaluation",
    "pipeline",
]


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_importable(name):
    importlib.import_module(f"oven_stereo.{name}")

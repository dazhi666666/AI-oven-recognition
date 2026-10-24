"""几何：三维点、托盘平面、高度地图、厚度和体积。（主要由 A 推进）"""

from .depth_geometry import (
    backproject_depth,
    fit_tray_plane,
    measure_depth_geometry,
    support_plane_cell_areas,
)
from .height_grid import measure_height_grid

__all__ = [
    "backproject_depth", "fit_tray_plane", "measure_depth_geometry",
    "measure_height_grid", "support_plane_cell_areas",
]

# 实测记录：本机 OpenCV 行为（不要再凭记忆推断）

记录环境：macOS (darwin)，Python 3.13.13，
**OpenCV 5.0.0**，numpy 2.5.4。以下每条都是本机跑出来的，附复现方式。
换 OpenCV 版本后**必须重跑**这些探针，尤其是第 1、2 条。

结论的原始复现脚本已随包归档在 `tools/probes/`：
`matrix_sgbm.py`、`probe_range.py`、`probe_range2.py`、`probe_border.py`、`probe_sgbm_units.py`。

---

## 1. `cv2.StereoSGBM(...)` 类构造器会段错误（致命）

```python
m = cv2.StereoSGBM(minDisparity=0, numDisparities=64, blockSize=5, P1=200, P2=800, ...)
m.compute(left, right)   # → Fatal Python error: Segmentation fault（进程 exit -11）
```

`cv2.StereoSGBM_create(...)` 用**完全相同的参数**正常返回 `int16` 结果。

隔离矩阵（`tools/probes/matrix_sgbm.py`，每格独立子进程，240x320）：

| 构造入口 | 无 mode、无 preFilterCap | 带 preFilterCap | 带 mode | 带 mode + preFilterCap |
|---|---|---|---|---|
| `cv2.StereoSGBM` | EXIT=-11 | EXIT=-11 | EXIT=-11 | EXIT=-11 |
| `cv2.StereoSGBM_create` | OK | OK | OK | OK |

结论：**与参数组合、`mode`、`preFilterCap` 都无关，只取决于构造入口**。构造器本身可用
（不报错），一 `compute` 就崩。

处理：`sgbm._matcher_factory()` 优先返回 `cv2.StereoSGBM_create`，仅在工厂函数缺失时
回退到类构造器。回归测试 `tests/stereo/test_sgbm_wrapper.py::test_matcher_factory_prefers_create_function`
断言当前版本下必须走工厂函数。

> 附带教训：早期探针把 `mode=` 误判为崩溃原因——因为那次探针恰好用了类构造器。
> 单变量隔离（每次一个子进程）才定位到真正原因。

## 2. 输出是定点：`int16`，值为视差 × 16

- 输出 dtype `int16`；`minDisparity=0` 时真实视差 = 原始值 / 16。
- 用人工斜平面图核验：按 `scale=16` 解码的中位绝对误差 **0.1000 px**（49920 像素），
  按 `scale=1` 解码为 **493.4 px**。

## 3. 无效像素编码为 `(minDisparity - 1) * 16`

- `minDisparity=0` → 无效值 `-16`（解码后 `-1.0`）。
- 因此无效判定不能只看"小于 0"就够，还要处理解码后仍落在合法范围外的情况：
  本包用 `raw == sentinel` **或** `decoded < minDisparity` **或** 非有限值，三者取并集。
- 无符号 dtype 上的注意事项：numpy 2 把负数写进 `uint16` 会直接抛 `OverflowError`，
  必须显式取模。`invalid_raw_sentinel(config, 16.0, np.uint16)` 应为 `65520`（`-16 % 65536`）；
  已有测试覆盖。

## 4. OpenCV 不校验 SGBM 参数（校验责任在本层）

`cv2.StereoSGBM_create(numDisparities=0, ...)`、`StereoSGBM_create(blockSize=4, ...)`
**都被接受**，不抛异常，随后产出无意义结果。所以
`SGBMConfig.validate()` 必须自己拦 `numDisparities % 16 != 0`、`blockSize` 非奇数等，
错误码 `E_INVALID_SEARCH_PARAMS`。

## 5. 左起 `numDisparities` 列恒无效（参数代价，不是 bug）

探针 `tools/probes/probe_border.py`，宽度 320/640 × `numDisparities` 32/64/128/256
共 8 组：

- 每组的左侧 `numDisparities` 列**全部**是无效编码；
- 校验"首个有效列号 == numDisparities"：8/8 成立，与图像宽度无关。

`tools/probes/probe_range.py` 的对照也印证：320 宽图上
`numDisparities=64` 有效比例 0.800（= 1 − 64/320），`numDisparities=256` 为 0.200。

写进代码：`stereo.config.left_border_invalid_columns()`；
写进运行统计：`stats()["left_border_invalid_columns"]` 与 `stats()["theoretical_max_valid_ratio"]`。

## 6. 搜索范围不够时，失效出现在**右侧（大视差端）**

斜平面真值视差 `[16, 16+0.1·(W−1)]`：W=320 需到 47.9，W=640 需到 79.9。

| 宽度 | numDisparities | 去掉左边界后的有效比例 | 覆盖检查结论 |
|---|---|---|---|
| 320 | 32 | 0.607 | 不够 |
| 320 | 64 / 128 / 256 | 1.000 | 够 |
| 640 | 32 | 0.388 | 不够 |
| 640 | 64 | 0.759 | 不够 |
| 640 | 128 / 256 | 1.000 | 够 |

即 `check_search_range_covers()` 的预测与实测有效区域一致（8/8）。
这正是 `tests/stereo/test_sgbm_wrapper.py::test_valid_region_matches_coverage_prediction`
锁住的接口契约。

## 7. 常数视差平面是病态输入，不能用作解码核验

同一纹理整行平移（无透视、无视差梯度）时，SGBM 的中位误差可达数十像素——代价聚合在
这种输入上没有可区分的匹配峰。**这不是实现缺陷**，而是输入性质。斜平面（线性视差）则稳定：
本机在正确真值映射下中位绝对误差 0.1 px 量级。

因此：`render_constant_shift_pair` 只用于冒烟测试；单位核验一律用
`render_slanted_plane_pair`。

## 8. 复跑这些探针

```bash
cd AI_oven_recognition
PYTHONPATH=src .venv/bin/python tools/probes/matrix_sgbm.py
PYTHONPATH=src .venv/bin/python tools/probes/probe_border.py
PYTHONPATH=src .venv/bin/python -m pytest tests/stereo -q -m integration
```

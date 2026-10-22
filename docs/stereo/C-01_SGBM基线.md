# C-01 SGBM 基线

2026-10-22。基于 Zhang_Yuchen 提交的 C-01 初版整理、修正而成。

## 1. 做了什么

用 OpenCV 现成的 SGBM 计算视差。有标定时，再把视差换算成距离（米）。

| 位置 | 内容 |
|---|---|
| `src/oven_stereo/stereo/config.py` | SGBM 参数：从 YAML 读取、检查是否合法；由工作距离推算视差搜索范围 |
| `src/oven_stereo/stereo/sgbm.py` | 调用 SGBM，把定点输出转成像素；无效点填 NaN，另给掩膜 |
| `src/oven_stereo/stereo/depth.py` | 视差 → 距离：`Z = f·B / (d + doffs)` |
| `src/oven_stereo/stereo/pipeline.py` | 入口 `estimate_depth(left, right, config, calib=None)` → `DepthResult` |
| `src/oven_stereo/stereo/__main__.py` | 命令行 |
| `src/oven_stereo/contracts/` | `RectifiedStereo`（标定）、`DepthResult`（深度结果） |
| `src/oven_stereo/data/images.py` | 读左右灰度图 |
| `configs/stereo/sgbm_c01.yaml` | 基线参数 |
| `tests/stereo/` | 38 项测试 |

## 2. 怎么用

```bash
python -m oven_stereo.stereo --left L.png --right R.png --config configs/stereo/sgbm_c01.yaml --calib calib.json --out results/C-01/<本次运行>
```

- 不给 `--calib` 时只输出视差，`result.json` 里的状态为 `disparity_only`。
- 标定 JSON 的字段：`calibration_id`、`image_size`（[宽, 高]）、`f_px`、`baseline_m`、`doffs_px`、`rectified`。
- 由 `cv2.stereoRectify` 得到的 P1、P2，可以用 `RectifiedStereo.from_projection_matrices` 转换。

输出文件：

| 文件 | 内容 |
|---|---|
| `disparity_px.npy` | float32，无效点为 NaN |
| `disparity_valid_mask.npy` | bool |
| `depth_m.npy`、`depth_valid_mask.npy` | 有标定时才有 |
| `result.json` | 有效比例、耗时、状态 |
| `config.json`、`calibration.json` | 本次用的参数和标定 |

## 3. 主点差 doffs 的符号

统一采用 Middlebury 约定：

- `doffs = 右相机主点 x − 左相机主点 x`
- `Z = f·B / (d + doffs)`

Middlebury `calib.txt` 里的 `doffs` 可以直接填进来。

初版用的是 `doff = cx_left − cx_right`、`Z = f·B / (d − doff)`。公式本身没错，但符号和 Middlebury 相反。如果把 Middlebury 的值直接填进初版，算出来的距离会悄悄出错，而且不报任何错误，所以这里统一了约定。

## 4. 实测结论（Windows，OpenCV 4.13.0，numpy 2.3.5）

| 结论 | 说明 | 对应测试 |
|---|---|---|
| 输出是 int16 定点，数值 = 视差 × 16 | 无效点的编码是 `(min_disparity − 1) × 16` | `test_decode_*` |
| 最左边 `min_disparity + num_disparities` 列一定无效 | 初版写的是 `num_disparities` 列，只在 `min_disparity = 0` 时成立 | `test_left_border_is_invalid` |
| OpenCV 不检查参数 | `numDisparities=0`、偶数 `blockSize` 都会被接受，然后算出无意义的结果，所以由 `SGBMConfig` 负责检查 | `test_invalid_parameters_rejected` |
| 整幅平移 20 像素能准确算出 20.0 | 见下方"更正" | `test_constant_shift_is_recovered` |
| 斜平面（视差 16–48 像素）：中位误差 0.1 像素 | 去掉左侧无效列后，覆盖率 100%。这只是人工图，**不代表真实场景精度** | `test_slanted_plane_subpixel_accuracy` |

**更正：** 初版的结论是"常数视差平面对 SGBM 是病态输入"，这是错的。原因在于初版生成右图时符号写反了：`right[:, s:] = left[:, :w-s]`，这样得到的视差是 −s，不是 +s。正确的写法是 `right[:, :w-s] = left[:, s:]`。用正确的图，SGBM 能准确算出平移量。

初版提到的另一个现象（macOS + OpenCV 5.0 上，用 `cv2.StereoSGBM(...)` 类构造器会段错误）没有在本机验证。现在代码统一使用标准写法 `cv2.StereoSGBM_create`，不受这个问题影响。

## 5. 相对初版的改动

- **目录**：从独立的 `AI_oven_recognition_C01/` 移入统一结构。依赖改用项目锁定的版本，不再单独维护 Python 3.13 / OpenCV 5.0 的环境。
- **删掉每次运行前的"单位实测核验"**：初版每次都额外跑一遍 SGBM 来确认 ×16。现在直接按 int16 定点解码，并用单元测试固定这个行为。
- **删掉 `reference.py` 里的误差统计**：误差统计由 D 的打分程序负责，不在算法模块里重复实现。
- **删掉 `C01_OVEN_RANGE` 预置参数**：它基于 f≈700 px、B=0.1 m、Z≤2.0 m 的假设，和项目的设定（160×120、B≈70 mm、0.3–1.5 m）不符。搜索范围改用 `disparity_range_for_depth` 按实际参数计算。
- **删掉只检查文字和源码字符串的测试**：例如用 AST 搜索禁用词、检查文档措辞。测试改为检查实际行为。
- **不再提交运行产物**：`outputs/` 下的 npy、png 都不进 Git，运行结果放 `results/`。
- **删掉探针脚本**：探针得到的结论已经写进本文和测试。

## 6. 还没做

- 没有用真实数据跑过。要等 B-01 交付 Middlebury 场景包。
- 视差搜索范围要等 A-01 选定模组后，按真实焦距、基线重算。

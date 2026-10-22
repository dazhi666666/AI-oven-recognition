# C-01 端到端证据摘要（人工构造输入）

- 输入来源：`artificial_test_pattern`（**非真实场景**，不得作为达标声明）
- 真实场景链路是否完整：`False`
- 固定配置：c01_default v1.0.0 [SGBM] 视差搜索 0..63 px, blockSize=5, P1=200, P2=800, disp12MaxDiff=1, uniquenessRatio=10, speckle=(100,2), preFilterCap=63
- 缩放实测核验：dtype=`int16`，dtype 先验=16.0，实测核验=16.0
- 视差有效比例：0.8000 （有效 61437 / 共 76800）
- 深度有效比例：0.8000，深度范围 1.532~3.489 m
- 搜索范围覆盖检查：覆盖完整
- 重复性：视差逐位一致=True，掩膜一致=True，深度一致=True
- 错误用例全部按预期拒绝：True

## 文件

| 路径 | 说明 |
|---|---|
| `inputs/` | 人工构造左右图与场景元数据（含构造参数） |
| `reference_only/` | 解析真值（参考数据），预测链路不读取 |
| `run_with_scale/` | 带标定运行：`disparity_px.npy`、`depth_m.npy`、掩膜、`run_metadata.json` |
| `run_disparity_only/` | 无标定运行：只输出视差，`depth_m.npy` 不存在 |
| `error_cases.json` | 最小错误用例的实际错误码与信息 |
| `determinism.json` | 重复运行一致性证据 |
| `reference_conversion_check.csv` | 手算换算检查（参考换算，非预测误差） |

## 关于误差

本包**不**给出精度达标结论：误差统计应由 D 的独立工具在留出数据上完成。
本包只保证：单位正确、接口隔离、数值可复跑、无效区域显式可见。

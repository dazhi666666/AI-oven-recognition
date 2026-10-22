# 角色 C · C-01 交付索引

本目录是角色 C **第一轮唯一授权任务 C-01**（双目匹配、米制深度与可信区域）的文档。

## 所有文件在哪里

全部收在本文件夹 `AI_oven_recognition/` 内——它是一个**自包含的提交单元**，
可以直接纳入 git 仓库（`git add AI_oven_recognition`）：

```
AI_oven_recognition/
├── README.md                顶层入口：安装、复跑、提交说明
├── requirements.txt         依赖（固定版本）
├── src/stereo/              代码：C-01 主模块（9 个 py）
├── tests/stereo/            测试：102 项（含 pytest.ini 标记注册、conftest.py）
├── docs/stereo/             文档：本目录（3 篇）
├── tools/probes/            探针脚本：本文件实测结论的原始复现脚本
└── outputs/stereo-c01/      运行产物：可复跑证据包
```

不需要提交、也**不在**本包内的东西：`.venv/`（按 `requirements.txt` 重建）、
`__pycache__/`（`.gitignore` 已排除）。

`tools/probes/` 明细（这些是复现证据，跑法见 `实测记录_OpenCV行为.md`）：

| 路径 | 用途 |
|---|---|
| `tools/probes/matrix_sgbm.py` | 隔离矩阵：定位 `cv2.StereoSGBM` 类构造器段错误 |
| `tools/probes/probe_border.py` | 验证「左起 numDisparities 列恒无效」规律 |
| `tools/probes/probe_range.py`、`probe_range2.py` | 无效像素分布与原始值探针 |
| `tools/probes/probe_sgbm_units.py` | 早期定点单位探针（早于封装层） |

| 文档 | 内容 |
|---|---|
| `C01_方法与接口.md` | 接口契约、方法、参数选择依据、复跑命令、证据、边界与未完成项 |
| `实测记录_OpenCV行为.md` | 本机 OpenCV 行为的**实测记录**（含两处会导致静默错误的坑） |

## 30 秒版本

- 做什么：`左右图 + 已核验标定 + 固定配置 → 视差(px) / 深度(m) + 显式有效掩膜`。
- 用什么：OpenCV 现成 `StereoSGBM`（未自写 SGM），封装层只做**单位解码、无效编码处理、参数校验、几何换算**。
- 结果：`tests/stereo` 102 项全绿；`outputs/stereo-c01` 可重复生成，8/8 错误用例按预期拒绝，重复运行逐位一致。
- 整合成提交单元后已在**文件夹内**复跑验证：102 项测试全绿，产物 22/22 数值文件逐位一致（仅时间戳/耗时/输出路径不同）。
- **重要**：本轮**没有**任何真实左右图与真实标定，全部输入是人工构造图样（`artificial_test_pattern`）。因此**不存在任何精度达标声明**；真实链路完整性标志为 `False`。

## 边界（本轮明确不做）

- 不接收、不读取真值/参考数据；探测入口 `estimate_depth` 的签名里没有任何真值参数（由 `tests/stereo/test_isolation.py` 用 AST 检查固定）。
- 不做参数自动搜索，不做深度网络训练，不做完整仿真。
- 不碰食品体积、平台设计、数据集标定修改、评测口径定义（那是其他角色/后续任务的边界）。
- `src/synthetic_stereo/` 未创建（C-04 未授权）。

## C-02 及以后的入口条件

进入下一轮前需要（缺一不可）：

1. B 提供的**真实**已校正左右图 + 校正状态与 `P1/P2`（用于把 `doff` 从人工假设换成真实值）。
2. D 给出的误差统计口径（本包只输出数值与掩膜，不自评）。
3. 任务书对 C-02 的明确授权（不自动推进）。

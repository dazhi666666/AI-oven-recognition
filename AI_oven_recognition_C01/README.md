# AI Oven Recognition · 角色 C（C-01 双目深度）

本文件夹是一个**自包含的提交单元**：代码、测试、文档、运行产物、复现脚本全部在里面，
不依赖文件夹之外的任何文件（除 Python 环境）。

对应任务：角色 C 第一轮唯一授权任务 **C-01 —— 双目匹配、米制深度与可信区域**。

- 输入：已校正的左右灰度图 + 已核验标定参数
- 输出：`disparity_px`（数值视差）+ 独立有效掩膜；标定可用时追加轴向 `depth_m` + 深度掩膜
- 实现：封装 OpenCV 现成 `StereoSGBM`，**未自写 SGM**；封装层负责单位解码、无效像素显式保留、参数校验、几何换算

> ⚠️ 重要声明：本包**没有**真实场景数据。所有输入都是人工构造图样（`artificial_test_pattern`），
> 因此**不存在任何精度达标声明**；真实链路完整性标志恒为 `False`。

## 快速开始

```bash
cd AI_oven_recognition

# 1) 建环境（本机验证：Python 3.13.13）
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# 如果 pypi.org 不通，用镜像：
# .venv/bin/pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 2) 跑测试（102 项）
PYTHONPATH=src .venv/bin/python -m pytest tests/stereo -q

# 3) 跑产线，重新生成证据包
PYTHONPATH=src .venv/bin/python -m stereo.run_c01 --out outputs/stereo-c01
```

`outputs/stereo-c01/` 已被提交过一次；重跑会原地覆盖，结果应逐位一致
（`determinism.json` 里 4 个 `identical_*` 应为 `true`）。

## 目录结构

```
AI_oven_recognition/
├── README.md                本文件
├── requirements.txt         依赖（固定版本）
├── .gitignore
├── src/stereo/              主模块（9 个 py）
├── tests/stereo/            测试 102 项（含 pytest.ini 标记注册、conftest.py）
├── docs/stereo/             文档 3 篇（交付索引 / 方法与接口 / OpenCV 实测记录）
├── tools/probes/            探针脚本：产生 docs 里"实测记录"那些结论的原始脚本
└── outputs/stereo-c01/      运行产物证据包
```

说明：`tests/stereo/conftest.py` 会自动把 `src/` 加入 `sys.path`，所以跑 pytest 时
`PYTHONPATH=src` 可省；但用 `python -m stereo.run_c01` 直接跑模块时必须有它。

## 文档怎么读

| 文档 | 内容 |
|---|---|
| `docs/stereo/README.md` | 交付索引：做了什么、结论、边界、下一步入口条件 |
| `docs/stereo/C01_方法与接口.md` | 接口契约、方法、参数选择依据、复跑命令、证据、未完成项 |
| `docs/stereo/实测记录_OpenCV行为.md` | 本机 OpenCV 行为的**实测记录**（含两处会导致静默错误的坑） |

先看 `docs/stereo/README.md`。最有价值的两个坑：

1. `cv2.StereoSGBM(...)` **类构造器**一旦 `compute` 就段错误，必须用 `cv2.StereoSGBM_create(...)`。
2. SGBM 输出是 `int16` 定点，值 = 视差 × 16；无效像素编码为 `(minDisparity-1)*16`，
   直接当视差用会静默产生错误深度。

## 提交到 git

本文件夹内部结构干净（无缓存、无虚拟环境），可直接纳入你的仓库：

```bash
# 中文文件名建议先关掉路径转义，否则 git status 会显示成 \344\270\255...
git config core.quotepath false

# 若要把本文件夹作为一个独立仓库
cd AI_oven_recognition
git init
git add .
git commit -m "feat(stereo): C-01 双目匹配、米制深度与可信区域封装"
git remote add origin <你的仓库地址>
git push -u origin main
```

若并入已有仓库，把 `AI_oven_recognition/` 整个目录拷到目标仓库后 `git add AI_oven_recognition` 即可。

## 边界（本任务明确不做）

- 不接收、不读取真值；探测入口 `estimate_depth` 的签名里没有任何真值参数
  （由 `tests/stereo/test_isolation.py` 用 AST 检查固定）。
- 不做参数自动搜索、不做深度网络训练、不做完整仿真。
- 不碰食品体积、平台设计、数据集标定修改、评测口径定义。
- `src/synthetic_stereo/` 未创建（C-04 未授权）。

## 进入下一轮（C-02）的条件

1. B 提供**真实**已校正左右图 + 校正状态与 `P1/P2`（现在视差-深度换算是人工假设的 `fx=700.0, B=0.1`）。
2. D 给出误差统计口径（本包只输出数值与掩膜，不自评达标）。
3. 任务书对 C-02 的明确授权。

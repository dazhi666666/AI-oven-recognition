# 数据目录

| 目录 | 放什么 | 进不进 Git |
|---|---|---|
| `manifests/` | 数据清单：每个样本的路径、来源、开发/测试划分、困难条件标签 | 进 |
| `raw/` | 原始数据：下载的公开数据、渲染输出、实拍原图 | 不进 |
| `interim/` | 中间数据：格式转换、缩放后的数据 | 不进 |

`raw/` 里的原始文件只读，不要直接修改。

## 建议的子目录

```
raw/
├── middlebury/<场景名>/     公开数据集（如 Middlebury 2014）
├── synthetic/<批次>/        虚拟烤箱渲染
└── real/<日期_批次>/        台架实拍
```

## 数据获取

| 数据 | 来源 | 说明 |
|---|---|---|
| Middlebury 2014 | https://vision.middlebury.edu/stereo/data/scenes2014/ | 先下载一个场景即可（左右图、`calib.txt`、`disp0.pfm`） |
| 仿真 | `src/oven_stereo/synthetic/` + `configs/data/` | 由配置生成，可重新生成 |
| 实拍 | 拍摄台架 | 每批附卡尺测量记录和标定版本 |

拿到数据后，在 `manifests/` 里登记。

## 已可复现的数据包

- [B-01：Adirondack-perfect 原分辨率公开场景包](../docs/data/B-01_公开场景包.md)：两个独立清单分别提供输入与答案；支持按需下载、SHA256 校验、标定/PFM 读取。原始文件和派生数组留在本地。
- [B 岗工作分析与开工清单](../docs/data/B_工作分析与开工清单.md)：任务、资源、依赖与下一步。

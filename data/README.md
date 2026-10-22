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

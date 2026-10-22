# 板端代码（RK3506）

C 语言实现，用 CMake 构建，只移植电脑端已经定稿的算法。

| 目录 | 放什么 |
|---|---|
| `toolchain/` | 交叉编译配置 |
| `src/` | 采集、校正查表、几何、主程序 |
| `src/match/` | 匹配程序 |
| `bench/` | 计算量、内存、程序大小、帧率的测量工具 |
| `tests/` | 与电脑端"标准输出"（`tests/fixtures/`）对答案 |

设计见 [docs/architecture.md](../docs/architecture.md) 第 3、4 节。

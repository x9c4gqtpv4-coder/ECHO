# 源码交付与复现 · 0.5.3

## 内容与版本

当前包含 Python 调色、A0 基线契约、SKU Profile、人工审核工作流、标准证据包和安全检查源码、Swift Vision 辅助程序、测试、GitHub Actions 配置模板、配置与诊断工具，以及默认关闭的 ATR18 精细蒙版、部位追色和人工真值验证模块。C1 Observer 只读诊断；0.5.3 还增加了可复算的 B1 授权、审核事务锁和无像素权限的影子规划器。主说明是 [0.5.3 安全闭环升级报告](UPGRADE_REPORT_0.5.3_SAFETY_PLANNER.md)。

已归档的 0.4.1 源码包保留在 `deliverables/2026-08-30-v0.4.1-a0-baseline/`。生成 0.5.3 交付包时，使用本文后面的重新导出命令，导出目录必须事先不存在。每个交付包包括：

- `batch-color-standardizer-source-v<version>.zip`
- `完整源代码.txt`
- `完整技术报告.txt`
- `MANIFEST.json`：逐文件 SHA-256、大小、权限、版本和主报告。
- `SHA256SUMS.json`：四项交付文件校验值。

不含照片、截图、.bcp 私有参考标准、模型、虚拟环境、凭据或原生二进制。所有旧版源码包均不覆盖。

## 在当前 Mac 使用

本机采用普通安装，重新安装后 doctor 应显示 0.5.3。开发时直接验证源码：

```sh
source .venv/bin/activate
PYTHONPATH=src VECLIB_MAXIMUM_THREADS=2 python -m unittest discover -s tests -v
batch-color doctor
```

0.5.3 当前本机完整回归为 166 项通过、0 失败、0 跳过，包含真实 macOS Vision EXIF 集成测试。单元测试不代表真实 SKU 识别准确率或成图验收。

源代码改变后，需要更新普通安装：

```sh
python -m pip install --no-deps --no-build-isolation --force-reinstall .
```

此电脑的可编辑安装曾因隐藏 .pth 文件被运行时忽略而失效，因此本次采用普通安装，不依赖修改系统或运行时设置。

## 在干净的 Mac 环境复现

需要 Apple Silicon Python 3.12、Swift 编译器及 Xcode 命令行工具。

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install .
zsh scripts/build_person_mask.sh
batch-color doctor
PYTHONPATH=src VECLIB_MAXIMUM_THREADS=2 python -m unittest discover -s tests -v
```

源码包不自带依赖，新电脑安装时可能需要联网。原生 EXIF 集成测试在未构建 Vision helper 的环境会跳过，其他测试不要求照片或模型。

精细自动解析是额外可选能力。基础安装不包含 PyTorch、Transformers 或任何 ATR18 权重；仅在隔离环境验证本地、已审计模型时安装 `.[fine-semantic]`。人工标签导入、蒙版验证和 `precision-match` 不需要这些大依赖。

## 使用及状态

操作命令和模式说明见根目录 README。所有调色结果必须复核；候选成功默认返回 0，但仍为 `accepted=false`。需要将待复核表达为退出码 3 时使用 `--strict-quality-exit`。未经 `sku-review` 批准不得接入发布流程。

新标准使用 .bcp；旧 JSON 只允许未验证的全局路径。verify-profile 是只读证据复算，不批准画质。match 默认保存源/参考两张蒙版和参考标准包；batch 每批只加载复算一次标准，复用参考资产。标准包含规范化客户照片，不进入源码交付或 Git。

`--mode background` 为默认；`--mode both` 为背景驱动人物调整实验。显式保护蒙版支持单图 match/apply，目前没有批量逐图商品语义解析。

只写无损 PNG/TIFF 母版，预览可用 JPEG。8-bit 处理限制没有被隐藏：高位深、透明度或坏 ICC 等不支持输入会被拒绝，而非悄悄改变。

## 局部对照诊断

```sh
PYTHONPATH=src python scripts/audit_skin_transition.py \
  --source data/input/source.png \
  --corrected data/output/candidate.png \
  --output data/output/new_local_audit \
  --roi forehead:350,80,690,335
```

ROI 必须按转正后的原图坐标人工确认，输出目录必须不存在。该脚本不识别皮肤、不自动判定色带，只给原尺寸对照、放大差分和诊断量。

## 重新导出

在 Git 工作区使用尚不存在的输出目录：

```sh
python scripts/export_source.py --output deliverables/new-handoff
```

工具只收集白名单文本源文件，包含未提交的新源码，并在清单记录工作区是否有修改。ZIP 不包含 Git 历史，重新导出应回到原 Git 工作区。

公开仓库使用清理后的单一初始历史，不携带本机路径、个人邮箱或旧私有提交元数据。交付清单仍记录实际 Git HEAD 和工作区状态，便于独立核验。

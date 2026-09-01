# 0.3.2 升级报告：可复算参考标准

日期：2026-08-27。范围：上一轮审计提出的 Profile 证据、参考蒙版留存和版本基线问题。

本版不增加人物追色算法，不接入新模型，不扩大到自动批准或无人值守恢复。它解决的是“依据什么标准计算、这份证据能否重新验证”，不是“所有图片已经追准且没有伪影”。

## 1. 交付结论

- 新增 .bcp 单文件参考标准包，包含实际规范化参考像素、实际参考背景蒙版、Profile、清单和生成记录。
- 加载标准时重新计算统计、曲面与诊断。仅有合法 JSON、trusted=true 或重新计算的归档哈希，不足以启用可信空间候选。
- 旧 JSON 仍可读取，但只能进行未验证的全局实验；显式请求 surface 也不能绕过证据要求。
- match 默认保存源图、参考图两张实际蒙版及参考标准包；新增只读 verify-profile。
- 保留原有输入精度、数值、防覆盖、缓存和事务回滚措施。所有候选仍为 review / accepted=false。
- 当前 Mac 114 项测试全部通过；两组实际照片、两种模式共四个结果，与 0.3.1 逐像素一致。
- 已更新本机普通安装，未下载模型、未新增依赖，未推送 GitHub。

四张成图完全一致意味着这次没有改变原有调色效果；不能把它解释为原来的画质已经通过独立验收。

## 2. 版本与代码边界

0.3.1 首先固化为本地提交 70ab8e7，并建立 v0.3.1-review-baseline 标签。只提交了与既有 Manifest 核对一致的源码范围，未包含客户照片、诊断产物和模型。

本版程序版本为 0.3.2，Profile schema 为 v5，Bundle schema 为 1，批处理缓存 schema 为 5。最终源码包 Manifest 记录实际 Git HEAD、工作树状态、每个源文件的大小与 SHA-256。

主要改动：

| 文件 | 作用 |
| --- | --- |
| src/batch_color/bundle.py | 标准包写入、边界检查、资产读取与统计复算 |
| src/batch_color/profile.py | v5 配方身份、运行期证据绑定、旧 JSON 降级基础 |
| src/batch_color/transfer.py | 空间路径增加证据门，报告区分各类验证状态 |
| src/batch_color/cli.py | 标准创建、双侧蒙版留存、只读验证与事务输出 |
| src/batch_color/batch.py | 标准包复用、每批一次复算、缓存身份更新 |
| tests/test_profile_evidence.py | 新增 30 项证据、篡改、兼容、资源与事务测试 |
| .gitignore | 忽略包含参考照片的所有 .bcp 标准包 |

原有颜色映射公式、空间曲面拟合算法和人物实验映射未更换。源图与参考图是否存在可迁移的空间光照对应关系，仍需人工判断。

## 3. 标准包如何工作

一个 .bcp 文件是受限 ZIP，固定包含：

    manifest.json
    profile.json
    provenance.json
    reference.png
    reference_mask.png

选择单文件而不是需要分别替换的目录，是为了让标准自身可以通过一次文件替换发布，减少半包状态。profile 命令额外导出的蒙版和报告，仍遵循整组暂存与报告最后提交协议；不承诺多个路径在断电时具有物理原子性。

reference.png 保存完整、已转正、已规范化的 8-bit sRGB 像素，而不是只保存缩略图。标准内部的 PNG 不再携带第二套 ICC/EXIF 解释，避免加载时重复转换；颜色空间、位深和坐标约定由规范化契约记录。它不是原始文件的字节级备份。

reference_mask.png 是实际参与参考背景估计的 L8 蒙版。255 表示背景，0 表示不可按背景调整。采样核心仍按既有算法由此蒙版计算。

加载顺序：

1. 限制归档大小，核对固定成员、成员类型与大小。
2. 验证清单、资产哈希、JSON 结构和原始 PNG 位深/尺寸。
3. 检查 Profile 版本、生成配方、相关实现哈希以及 NumPy/Pillow 身份。
4. 核对实际参考像素与蒙版的像素哈希和几何。
5. 用包内实际像素、蒙版及当前受支持实现重新计算 Profile。
6. 对比统计、曲面、诊断和配方；不一致则拒绝，不偷偷采用导入值。
7. 返回重新计算的对象，赋予当前进程中的证据绑定资格。

浮点复算允许 1e-7 的绝对/相对比较容差，整数、结构、字段与非浮点值严格比较。即使在容差内，运行时也使用重新计算值，不使用导入系数。

不是只检查“manifest 哈希一致”。如果有人修改曲面，同时重算 manifest，复算仍会发现其与参考像素不符。

## 4. trusted 不再等于证据成立

空间候选需要同时满足：

    参考证据绑定有效
    + 参考空间支撑通过
    + 当前源图空间支撑通过
    + 用户请求的路径允许空间候选

原有 trusted 字段只是拟合/支撑诊断，不独立授予导入文件的证据资格。

运行期绑定不写入 JSON。JSON 往返、dataclasses.replace 创建新对象、修改嵌套统计或诊断，都不能沿用旧绑定。直接调用 create_profile 时，必须提供相同几何的 RGB 参考像素与 L8 参考蒙版才能建立绑定；未提供蒙版时明确记录 reference_mask_missing，不再宣称空间支撑可信。

报告分别显示：

- numeric_valid：通过数值与结构检查。
- reference_evidence_verified：依据实际参考资产复算过。
- reference_surface_support_passed：参考空间采样支撑是否通过。
- reference_allows_spatial_candidate：仅参考这一侧是否满足候选条件。
- sampling_reviewed：本版不自动认定人工采样复核，保持 false。
- spatial_correspondence_verified：没有自动验证图对的光照坐标对应，保持 false。
- human_review_required：始终为 true。

运行期绑定是输入证据契约，不是对任意恶意 Python 代码的沙箱。哈希与复算不认证参考图片的作者身份，不保证蒙版语义正确；如果所有像素、蒙版和统计都被一致改成另一套标准，系统只能验证新标准的内部一致性，不能代替用户判断它是否是想要的标准。

## 5. 导入资源与文件安全

标准包不解压到外部路径，不读取 pickle/NPZ，不动态加载任何代码。重复成员、额外成员、缺失成员、符号链接成员、加密或不支持的压缩类型会被拒绝。

当前限制：

- 归档不超过 128 MiB。
- 参考及蒙版不超过 2400 万像素、单边不超过 12000 像素。
- 参考 PNG 必须是原始 8-bit RGB；蒙版必须是原始 8-bit L。
- 拒绝 Alpha、附带二次 ICC/EXIF 解释和多帧资产。
- 单独 Profile JSON 限制为 1 MiB；拒绝重复键和非有限常量。
- 各成员另有解压后大小限制，读取有界，不靠压缩比猜测安全性。

资产哈希和复算均在发布前完成。新增标准与参考蒙版纳入已有整组输出事务；一项暂存或普通发布异常不能留下新候选搭配缺失标准的伪完成状态。发布失败测试包含新增参考资产角色及替换已有完整结果。

断电、SIGKILL 或回滚时磁盘再次失败的恢复，仍是 0.3.1 报告承认的边界；本版没有增加 recover 命令。

## 6. CLI 与兼容性

### 新建标准

    batch-color profile --reference reference.jpg --name warm-gray --output standard.bcp --reference-mask-output reference-mask.png

输出 standard.bcp 和 standard.bcp.report.json。参考蒙版在包内始终存在，额外 PNG 可选。重要参考图建议先检查实际蒙版，或通过 --reference-mask 提供人工复核的蒙版。

### 只读验证

    batch-color verify-profile --profile standard.bcp

它重新计算参考证据，不重新分割图片，不产生新资产、不更改批准状态。证据通过返回 0；旧 JSON 的未验证状态返回 3；损坏或不支持的标准返回受控错误 2。任何返回值都不等于图片画质已通过。

### 单图匹配

match 默认在候选之外保存：

    <output>.report.json
    <output>.mask.png
    <output>.reference-mask.png
    <output>.reference.bcp

可以用 --mask-output、--reference-mask-output、--profile-output 明确指定路径。源蒙版为实际应用保护约束后的背景蒙版，参考蒙版为实际参考取样依据。所有输出统一检查，不能覆盖源图、参考图、输入蒙版或互相别名。

apply 默认保存实际源蒙版，复用既有标准包，不为每张图重复存参考图。

### 批处理

    batch-color batch --input input --profile standard.bcp --output output --no-previews

每次 batch 只加载复算一次参考标准，后续逐图使用同一标准。缓存身份包含标准文件内容、证据状态、算法、依赖、参数与实际输出哈希。缓存命中仍是待复核，不改变 accepted=false。

### 旧格式

旧 v2/v3/v4 JSON 和单独的 v5 JSON 可继续用于受限全局追色，并记录 reference_evidence_unverified_global_only_rebuild_bundle。显式 surface 也退回 global。

新 CLI 创建标准必须使用 .bcp 扩展名，旧的 --profile-output xxx.json 命令需修改。标准必须从参考图重建，不能仅改扩展名。Python to_json 保留为统计调试出口，不保留证据资格。

算法实现或相关依赖身份改变时，本版严格要求重建标准。跨版本自动迁移尚未实现，这是兼容性限制而非静默接受旧证据。

## 7. 实际验证

### 回归

当前 Mac / Python 3.12.13 / NumPy 2.3.5 / Pillow 12.3.0：

- 114 项测试通过，0 跳过、0 失败/错误，包含本机 Vision EXIF 集成测试。
- 原有 84 项继续通过；新增 30 项覆盖证据包与新输出路径，包括损坏 DEFLATE 数据的受控错误。
- 新增测试包括：伪造 JSON 与直接 API、重算 manifest 后的篡改、参考像素/蒙版替换、缺失/重复/越界/链接成员、尺寸和位深限制、错误复读保留旧包、参考资产发布失败回滚。
- 窄条蒙版形成的标准即使证据复算通过，也不能绕过空间支撑不足。
- 源码格式检查 git diff --check 通过。

### 使用上一轮原始反例

未重新挑选宽松案例，直接读取上一轮审计留下的 forged-140.json：

| 项目 | 0.3.1 | 0.3.2 |
| --- | --- | --- |
| 默认 auto 路径 | spatial-surface | global-monotone |
| 均匀 RGB 160 输入的输出范围 | RGB 137–149 | RGB 143–143 |
| 参考证据资格 | 未独立验证 | 明确未验证，禁止空间路径 |
| accepted | false | false |

RGB 143 是当前强度 0.85 下的保守全局结果，不宣称与 RGB 140 目标完全相同。关键变化是没有凭伪造曲面制造横向渐变。

### 两组真实照片

每组分别运行 background 和 both。参考资产复读像素、双侧蒙版、所有输出哈希均检查；原图和全部既有结果哈希不变。

| 用例 | 路径 | 内部背景距离：前 → 后 | 与 0.3.1 候选最大通道差 |
| --- | --- | --- | --- |
| 01 全身 background | spatial-surface | 2.7812 → 0.6658 | 0 |
| 01 全身 both | spatial-surface | 2.7812 → 0.6658 | 0 |
| 02 近景 background | global-monotone | 5.9738 → 5.7199 | 0 |
| 02 近景 both | global-monotone | 5.9738 → 5.7199 | 0 |

两个 background 结果的零蒙版核心分别为 333941 和 2238784 像素，最大通道变化都为 0，比较基准为方向/ICC 规范化后的源图。

上述距离是既有自定义 Oklab 背景统计量，不是 ΔE2000 或画质认证。近景 both 的原有全图色域裁切比例仍为 1.3132%；本版没有通过修改指标来隐藏它，也没有宣称已消除所有局部伪影。

单图完整运行约 2.57–2.90 秒。两张图批处理首跑约 4.84 秒，缓存复跑约 0.22 秒。两个参考包约 0.63 MB 和 0.91 MB。这些仅是本机这组样例结果，不是任意图片或冷启动保证。

### 本机安装

使用既有环境离线、无新依赖地安装 0.3.2。安装版 Python 源文件与工作区逐一比对；安装版可以复算新标准，实际 CLI 的两张图批处理首跑与缓存复跑都保持 2 张 review、0 error、0 accepted。

详细机器证据在本地 data/output/upgrade_032_yKXfdB/：

- verification.json：真实照片、旧反例、批处理、输入与旧结果保护。
- unit-tests.log：完整 114 项测试。
- installation-verification.json：安装位置、代码比对与实际 CLI 验证。
- source-package-verification.json：最终源码包完整性与隔离验证结果。

这些本地照片和日志不包含在对外源码包内；不把无素材环境无法复现的照片指标伪装成源码包自带证据。

## 8. 源码交付

目录：deliverables/2026-08-27-v0.3.2-profile-evidence/

- batch-color-standardizer-source-v0.3.2.zip
- 完整源代码.txt
- 完整技术报告.txt
- MANIFEST.json
- SHA256SUMS.json

包内包括源码、回归测试、配置说明、构建脚本及历次报告。不包含照片、标准 .bcp、虚拟环境、模型、凭据、Git 历史或编译产物。旧源码包保留，不覆盖。

## 9. 尚未完成与下一步

尚未完成：

- 独立皮肤、头发、商品的准确追色。
- 人工确认的双侧语义基准与独立验收集。
- 经校准的色块、分层、色边、纹理及局部裁切质量门。
- 跨构图空间光照对应的自动验证。
- 自动事务恢复、完整高位深路径、生产自动批准。

下一步应实施 0.4A：先建立源图与参考图的区域基准、保护权限和最小质量门，再开发第一条背景+皮肤独立追色路径。不要把本次证据修复命名为“人物追色升级完成”或“画质验收完成”。

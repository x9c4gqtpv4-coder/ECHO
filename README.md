# 批量校色 · 0.5.3 安全闭环与影子规划版

面向时尚电商图片的参考追色实验项目。最终目标仍是背景、人物、皮肤和头发的分区标准化；当前不是已通过生产验收的全自动工具。

## 项目效果

输入每个 SKU 自己的 `指定场景` 与一组 `成品动作`，系统会在 Mac 本地生成整套视觉更统一的待复核图片：

- 背景向该 SKU 的指定场景靠拢，统一冷暖、亮度和中性色偏。
- 服装、皮肤和头发在默认 A0 中作为整个人物共用一条平滑变换，避免分层色块和不自然过渡。
- 可选 B1 能够定位脸、头发、上衣、裤、鞋、包和配饰，并在授权区域内做有界局部追色。
- 书本、商标、配饰等对象可通过保护蒙版保持原色。
- 每次运行同时输出蒙版、对照预览、数值报告、执行身份和审核证据。

项目追求的是“同一 SKU 的整套视觉一致性”，不宣称只凭普通 sRGB 图片就能还原物理绝对商品色。所有成图仍需人工复核。

0.5.3 完整保留已经过23套实图验证的低负载 `A0` 两区像素路径：背景追本 SKU `指定场景`；服装、皮肤和头发作为整个人物使用同一条平滑变换。本版不更改 A0 的黄金像素输出，重点修复 A0 代码闭包身份、B1 蒙版授权伪造、审核竞态和 C1 证据越权，并新增一份没有像素权限的 `execution-plan.json`。

`B1` 和 `C1` 都不会被 `sku-pilot` 自动调用。B1 必须从同目录的标签图、置信度图和阈值策略重算授权蒙版；保护蒙版只能扣减修改权限。C1 只会写入 `status=review`、`accepted=false`、`pixel_output_changed=false` 的哈希绑定 JSON 报告。在完成经验证的 Planckian locus + Duv 方法前，CCT/mired 统一不输出。

`A0` 的颜色数学契约和指纹固定在 `batch_color.baseline`。默认背景强度为 0.78、人物强度为 0.58。A0 兼容性现同时校验参数、核心算法指纹、NumPy/Pillow 版本和 macOS Vision 助手哈希。

## 当前边界

所有调色结果都是待复核候选，`accepted=false`。背景指标改善单独记录为 `baseline_checks_passed`，不代表皮肤、商品色或过渡质量达标。本版修复了已复现的文件覆盖、缓存污染、方向错位和亮度反向问题；不宣称所有图片绝无色块。

- 默认 `--mode background`：只在背景蒙版允许区域调整，蒙版为零的区域保持统一色彩空间后的原像素。
- `--mode both`：显式启用背景驱动的人物调整实验。人物采用固定黑白端点的单调映射，但不是独立肤色/头发追色。
- `--protected-mask`：白色区域完全不改，黑色区域允许调整；这是用户指定保护，不是自动商品识别。软灰区域按权重减弱调整。
- 主图只输出 PNG/TIFF 无损候选，JPEG 仅供预览。当前为 8-bit sRGB，拒绝静默降低高位深输入或丢弃透明通道。
- 人物/背景蒙版不等于商品/皮肤语义蒙版。蒙版错误、未知边界及困难光照仍需人工复核。
- `fine-masks` 和 `precision-match` 是可选 B1 能力，不改变 A0。自动标签必须同时提供逐像素置信度；复核标签必须记录审核人。
- 精细追色只在授权蒙版内部变更，区域外像素逐点保持不变；边缘羽化只向区域内部收缩，禁止颜色泄漏到书、包、皮肤或背景。

详见 [0.5.3 安全闭环升级报告](docs/UPGRADE_REPORT_0.5.3_SAFETY_PLANNER.md)、[0.5.3 逻辑思维导图](docs/LOGIC_MINDMAP_0.5.3.md) 和 [源码交付说明](docs/SOURCE_HANDOFF.md)。

## C1 冷暖与曝光只读分析

不提供中性蒙版时，C1 仍可生成曝光型增益和色调诊断，但自动冷暖只标记为 `hypothesis_only`：

```sh
batch-color c1-analyse \
  --input source.png \
  --reference reference.png \
  --region-name background \
  --source-region-mask masks/source-background.png \
  --reference-region-mask masks/reference-background.png \
  --comparison-evidence same_surface \
  --report reports/c1-background.json
```

仅当两张图中有同一中性实体或已人工确认的中性区域时，显式传入两张蒙版：

```sh
batch-color c1-analyse \
  --input source.png --reference reference.png \
  --source-neutral-mask masks/source-neutral.png \
  --reference-neutral-mask masks/reference-neutral.png \
  --neutral-evidence same_entity \
  --report reports/c1-neutral.json
```

报告中的 `relative_exposure_like_stops` 是显示参考下的相对增益，不是拍摄 EV。只有两张可比区域蒙版与 `same_surface`/`human_confirmed` 证据同时存在，才能成为未来曝光候选。C1 不生成图片，也不会自动修改生产流程。

0.3.2 保留 0.3.1 的安全修复，并新增可复算标准包、参考蒙版资产、运行期证据绑定。数值合法的 JSON 不能单凭 trusted=true 启用空间路径。它仍不等于完成了独立肤色目标或全部伪影检测。

## 本机使用

在项目根目录：

```sh
source .venv/bin/activate
batch-color doctor
```

开发测试直接使用源码：

```sh
PYTHONPATH=src VECLIB_MAXIMUM_THREADS=2 python -m unittest discover -s tests -v
```

安装后 doctor 应显示 0.5.3。之后改完源码，普通安装的命令不会自动更新，需要重新安装：

```sh
python -m pip install --no-deps --no-build-isolation --force-reinstall .
```

## SKU 两区批处理

SKU 目录必须包含且只包含一张 `指定场景.*`，需要处理的图片命名为 `成品动作*`。以新的 `run-name` 运行，不覆盖旧候选：

```sh
VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 batch-color sku-pilot \
  --dataset-root "/path/to/fashion-catalog" \
  --sku sz000000000000000000000 \
  --output-root "/path/to/fashion-catalog/校色输出" \
  --run-name v0.4.2-a0 \
  --pipeline-mode person-background \
  --background-strength 0.78 \
  --person-strength 0.58 \
  --mask-backend vision
```

输出直接保存在 `校色输出/<SKU>/`，包含 `待复核候选`、`蒙版`、`报告`、`预览`、`sku-profile.json`、`run-identity.json`、`execution-plan.json`、`review-status.json`、`summary.json` 和 `整套对照.jpg`。只有执行人工批准后才生成 `已批准成品`。

如需先确认产品图和服装锚点，先建立 SKU Profile：

```sh
batch-color sku-init \
  --dataset-root "/path/to/fashion-catalog" \
  --sku SKU-001 \
  --output-root "/path/to/fashion-catalog/校色输出" \
  --product-anchor "产品图.png" --confirm-product \
  --garment-anchor "成品动作3.png" --confirm-garment
```

这一步会真正读取并绑定产品图哈希，但 A0 中仍只作证据，不会暗中改成产品色追色。审图后显式批准：

```sh
batch-color sku-review \
  --output-root "/path/to/fashion-catalog/校色输出" \
  --sku SKU-001 --decision approve \
  --reviewer "name" --reason "套图、人脸、服装和过渡已复核"
```

对已有 SKU 结果重算时必须显式传入 `--replace-output`。它只替换上述由本工具管理的产物，不删除 SKU 目录中其他旧测试文件夹。

0.5.3 审核必须绑定 `execution-plan.json`。0.5.2 或更旧的候选包不会被静默升格批准；需用 0.5.3 重算后再复核，被替换的旧报告证据会保留在 `历史/`。

此命令适用于当前已经配置好依赖和构建工具的虚拟环境。新电脑按交付说明安装依赖，不要在未知环境跳过依赖安装。

## 可选精细识别与部位追色

精细层采用 ATR18 固定类别契约，不把模型自己的类别顺序直接当真。当前支持两种输入：

1. 本地、已审核许可证的 SegFormer ATR18 safetensors 模型；只本地读取，不联网下载，默认 CPU、最长边 768、2 线程。
2. 外部或人工修正的 0..17 标签图。自动标签必须有逐像素置信度图；人工修正标签必须记录审核人。

模型权重不随代码提供。只有需要自动精细解析的环境才安装重量较大的可选依赖：

```sh
python -m pip install '.[fine-semantic]'

batch-color fine-masks \
  --input source.png \
  --model-dir /path/to/audited-segformer-atr18 \
  --label-status automatic \
  --confidence-policy configs/fine-confidence-policy.example.json \
  --output-dir masks/source
```

如自动标签需要修正，编辑导出的 `labels-atr18.png`，保持 0..17 类别索引，再重新形成复核蒙版包：

```sh
batch-color fine-masks \
  --input source.png \
  --label-map corrected-labels.png \
  --label-status reviewed \
  --reviewed-by "name" \
  --output-dir masks/source-reviewed
```

源图和参考图都需要各自的蒙版与报告。下面示例只追上衣，其他部位可换成 `garment`、`pants`、`skirt`、`dress`、`shoes`、`bag`、`skin`、`hair` 等：

```sh
batch-color precision-match \
  --input source.png --reference reference.png \
  --source-mask masks/source-reviewed/upper_clothes.png \
  --reference-mask masks/reference-reviewed/upper_clothes.png \
  --source-mask-report masks/source-reviewed/fine-mask-report.json \
  --reference-mask-report masks/reference-reviewed/fine-mask-report.json \
  --sku-profile /path/to/校色输出/SKU-001/sku-profile.json \
  --region upper_clothes \
  --object-id "SKU-001:upper-clothes" \
  --sku-role target_sku \
  --reference-policy sku_approved_anchor \
  --reference-id "<reference.png 的 SHA-256>" \
  --output output/upper-clothes-candidate.png
```

`precision-match` 使用区域独立的 OKLab 亮度分位数和稳健色度统计、有界强度/明度/色度变换，从原图一次计算并输出无损候选。报告记录目标距离改善、新增裁切、区域外像素变化和边界残差；结果仍为 `accepted=false`。角色、区域与参考策略不相容时会拒绝执行；参考 ID 必须等于实际参考文件的 SHA-256，且该文件必须匹配 SKU Profile 中已确认的指定场景或服装锚点。

将模型预测与人工复核真值量化对照：

```sh
batch-color validate-fine \
  --predicted-label-map validation/predicted.png \
  --truth-label-map validation/reviewed-truth.png \
  --required-region garment --required-region skin \
  --report validation/report.json
```

它输出逐类 IoU、边界 F1 和跨角色泄漏，只验证给定标签相对给定真值，不批准调色画质。B1 目前仍是独立工具，还没有自动叠加进 23 套 A0 批处理。

## 创建并复核一个标准

```sh
VECLIB_MAXIMUM_THREADS=2 batch-color profile \
  --reference data/references/test_01_target.jpg \
  --name warm-gray \
  --output data/output/warm-gray.bcp \
  --reference-mask-output data/output/warm-gray-reference-mask.png

batch-color verify-profile --profile data/output/warm-gray.bcp
```

`.bcp` 是单文件 ZIP 标准包，包含 profile.json、manifest.json、provenance.json、规范化 reference.png 和实际 reference_mask.png。标准包内始终保留参考蒙版，额外导出 PNG 便于查看。profile 命令同时生成 `<output>.report.json`，多产物仍按暂存、校验、报告最后提交的协议发布。

verify-profile 只读加载并重新计算参考统计、曲面和诊断。它不运行自动分割，也不批准画质。数值、证据、空间支撑和人工复核是不同状态：证据已复算不代表蒙版语义正确，亦不代表两张图的光照位置对应。

导入时检查固定文件名、类型、哈希、大小和 PNG 原始位深；不解压外部路径，不执行代码或读取 pickle。参考资产上限为 2400 万像素、单边 12000 像素，归档上限 128 MiB。算法实现或相关依赖身份不符时要求重新生成标准，不悄悄接受旧曲面。

**标准包包含客户参考图像素与路径元数据，属于私有资产。** 已加入 Git 忽略，不应随源码包公开上传。

### 从 0.3.1 迁移

- 旧 v2/v3/v4 JSON（以及单独导出的 v5 JSON）仍可读，但只能进行明确标为未验证的全局追色；`--path surface` 也不能绕过。
- 新 CLI 的 `profile --output` 和 `match --profile-output` 必须改用 `.bcp`，不能继续输出 `.json`。从原始参考图重新创建，不是改扩展名。
- Python 的 `ColorProfile.to_json()` 仍可用于统计调试；JSON 往返不保留运行期证据资格。标准包使用 `batch_color.bundle.save_bundle/load_profile`。
- 旧候选、旧 Profile、旧源码包不自动覆盖或删除。已有文件仍需新路径或显式 `--overwrite`。

## 单图追色

以下文件名是本机测试素材，源码包不包含照片。

```sh
VECLIB_MAXIMUM_THREADS=2 batch-color match \
  --input data/input/test_02_source.png \
  --reference data/references/test_02_target.jpg \
  --mode background \
  --output data/output/new_candidate.png \
  --preview data/output/new_comparison.jpg \
  --profile-output data/output/new_standard.bcp \
  --mask-output data/output/new_mask.png \
  --reference-mask-output data/output/new_reference_mask.png
```

如果要对照人物整体调整实验，显式改为 `--mode both`。有人工确认的保护区域时，额外传入 `--protected-mask masks/protected.png`。外部蒙版必须与 EXIF 转正后的图像同尺寸，不能自动拉伸。

还支持 `--background-mask`、`--reference-mask` 分别指定源图和参考图的背景蒙版，白色表示背景。没有外部蒙版时，Mac 优先使用原生 Vision；降级启发式时会记录原因，结果仍必须复核。

未指定报告路径时，自动生成 `<output>.report.json`。所有产物路径会在写入前一起检查；主图、参考图、蒙版输入和 Profile 输入都不能被覆盖。输出已存在时，需新路径或显式 `--overwrite`；该参数不能绕过输入保护。

match 默认还会保存 `<output>.mask.png`、`<output>.reference-mask.png` 和 `<output>.reference.bcp`。apply 默认保存实际源图蒙版，但复用既有标准包，不为每张源图重复存参考图。批量使用应先创建一次标准，再运行 batch。

整组产物先暂存、校验，再发布。正常异常会回滚；错误另写到 .batch-color-errors，不保留 processing 作为最终状态。

多路径发布不是文件系统级原子事务。断电或强制终止可能留下暂存、备份和锁，需要人工检查恢复。下游必须验证最终报告及全部产物哈希，不能仅凭文件存在使用结果。

## 批处理和缓存

```sh
VECLIB_MAXIMUM_THREADS=2 batch-color batch \
  --input data/input \
  --profile data/output/new_standard.bcp \
  --output data/output/new_batch \
  --mode background \
  --no-previews
```

输出示例：

```text
new_batch/
├── candidates/photo.jpg.png
├── masks/photo.jpg.png
├── reports/photo.jpg.json
├── errors/<job-id>.json
├── summary.json
└── summary.csv
```

保留原文件扩展名，使 `photo.jpg` 与 `photo.png` 的结果、蒙版和报告互不覆盖。旧 `corrected/` 目录不会自动迁移、删除或充当新缓存。

缓存签名绑定输入内容、Profile 内容、参数、源码、依赖、原生分割程序与输出策略；产物哈希也必须一致。此版本不再单独猜测复用一个同尺寸蒙版。改变配置会重算同一输出位置的候选，不会把旧图标成新标准。

缓存 schema 为 5。每次批处理开始复算一次标准证据，而不是每张图重复复算参考图。修改包内资产会改变标准文件内容；清单造假或统计不一致会在候选计算前被拒绝。

auto 请求遇到 Vision 临时运行失败或超时，其 heuristic 候选不形成长期缓存命中资格，下次重试。明确选择 heuristic，或机器确实未配置 helper，仍可正常缓存；后来安装 helper 会使身份签名变化。

`status=review/candidate` 和 `computation=cached` 是两个维度。缓存命中不消除复核状态。

退出码：

- `0`：执行成功，包括成功生成待复核候选；不代表画质批准。
- `3`：传入 `--strict-quality-exit` 时，候选仍待复核；或 `verify-profile` 检查到未验证旧标准。
- `verify-profile` 检查旧 JSON 返回 `3`，表示未验证、仅供全局实验，不是画质批准。
- `2`：输入、参数、路径或处理错误。

不要把“文件已存在”或“命令退出了”当作合格交付。批处理同一输出目录只允许一个写入者；中断遗留锁文件时，先确认没有任务运行，再处理该目录下的锁，不能盲删。

## Mac 低负载与隐私

A0 不新增大模型框架，继续使用 NumPy、Pillow 和系统 Vision。B1 的 PyTorch/Transformers 是可选依赖，未显式安装和调用时不加载；默认 CPU、2 线程、缩略图分割、串行执行，最终变换仍分块。两条路径都不会把照片上传到云端。

`data/input`、`data/references`、`data/output`、缓存、模型、虚拟环境及交付 ZIP 都被 Git 忽略。公开仓库只保存源码、测试、通用配置与说明，不包含用户照片、SKU 数据、个人路径、凭据或模型权重。

Profile schema 为 v5，标准包 schema 为 1。参考像素、蒙版、分析配方及实现身份绑定后，加载重新计算，不信任单独的 trusted 字段。无参考蒙版的公共 API 不再宣称空间支撑可信；源图或参考的空间门未通过时仍回退 global。运行期绑定防止 JSON 或误修改复用资格，不是对任意 Python 代码的沙箱，也不认证参考图作者身份。所有版本都不是完整的人物色彩标准。

0.5.1 已把 `B1 SKU Color Fidelity` 升级到验证就绪状态，但仓库仍没有已批准权重或用户人工真值集。下一阶段需要用真实 SKU 哨兵集校准阈值、核验模型许可证、绑定锚点审批记录、验证产品图与上身图颜色对应，并把已验收的 B1 残差以“从原图单次渲染”的方式接入 SKU 批处理。完成这些门槛之前，B1 保持显式可选，不能改写或冒充 A0。

## 开源许可

项目源码使用 [MIT License](LICENSE)。可选第三方模型权重、训练数据和系统框架各自保留它们的许可条款，不因本仓库的 MIT License 而改变。

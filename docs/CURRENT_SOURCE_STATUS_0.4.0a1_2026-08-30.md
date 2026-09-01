# 当前源码状态 · 0.4.0a1 · 2026-08-30

## 本次新增

- 按 SKU 扫描 `指定场景`、`成品动作*` 和可选产品图，扩展名大小写兼容。
- 双锚点流程：指定场景负责背景、肤色、发色和整体光感；同 SKU 内服装色彩中心图负责服装一致性。
- Apple Vision 人物蒙版与脸部检测辅助源码。
- 保守的服装、脸部皮肤和头发区域蒙版，以及未经授权区域像素不变检查。
- 每个 SKU 独立输出无损 PNG、蒙版、逐图 JSON 报告、局部检查和整套对照图。
- 输入哈希复核、临时目录发布、人工复核状态和成套一致性指标。

## 当前质量状态

当前版本是实验性待复核版本，不是自动发布版本。第一套白色上衣测试的第三版已通过人工边界检查；第一、第二版作为不通过证据保留。第二套黑色连衣裙独立验证尚未完成，因此不能宣称全部 SKU 已验证。

已知限制：手臂和手部肤色区域仍需增强；黑色、蕾丝、网纱、印花及发丝与服装重叠仍需在锁定验证套中检查。自动数值通过不能替代视觉复核。

## 当前命令

```sh
PYTHONPATH=src VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 \
.venv/bin/python -m batch_color sku-pilot \
  --dataset-root '/path/to/fashion-catalog' \
  --sku '<SKU>' \
  --run-name '<run-name>' \
  --garment-kind top \
  --garment-hint light \
  --mask-backend vision
```

`garment-kind` 支持 `top`、`dress`、`bottom`、`set`；`garment-hint` 支持 `light`、`dark`、`midtone`、`any`。默认输出为 `<dataset-root>/校色输出/<SKU>/<run-name>/`，不会覆盖原图。

## 当前验证

```sh
PYTHONPATH=src VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 \
.venv/bin/python -m unittest discover -s tests -v
```

2026-08-30 本机完整回归为 118 项通过。源码包不包含照片、输出图、私有标准、模型、虚拟环境、凭据或已编译二进制。

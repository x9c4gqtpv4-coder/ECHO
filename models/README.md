# 模型目录

模型权重只保存在本机并按需下载。每个模型接入时需要记录源码许可证、权重许可证、下载地址、哈希值和目标运行后端；权重文件不会提交到 GitHub。

## MediaPipe Selfie Multiclass

- 用途：只用于质量检测和可选精细语义路径，不是当前两区批处理的调色必需条件。
- 类别：background、hair、body-skin、face-skin、clothes、accessories。
- 官方资产：`https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite`
- SHA-256：`c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0`
- 下载：`scripts/download_semantic_model.sh`
- 运行时：MediaPipe Tasks 1.0.1，Apple Silicon CPU。

权重不进入 Git；对外发布或商用前仍需由项目负责人对当时的 Google MediaPipe 模型资产条款进行独立复核。

## 可选 ATR18 精细服装解析

- 用途：0.5.1 `fine-masks` 的可选自动后端，可区分上衣、裙子、裤子、连衣裙、左右鞋、包、腰带、围巾、帽子、墨镜、脸、手臂、腿和头发。
- 格式：本地 Transformers SegFormer 目录，必须包含 safetensors；拒绝 pickle 权重、远程下载、符号链接和类别顺序不一致。
- 运行：默认 CPU、最长边 768、2 线程；不调用时不会加载 PyTorch。
- 状态：项目不提供权重。代码许可证、模型权重许可证、训练数据条款和商用范围必须分别审核，并记录模型目录清单哈希。权重接入后还必须用人工复核 ATR18 标签和 `validate-fine` 测每类 IoU、边界 F1 与跨角色泄漏。

可参考的公开实现包括 SCHP 的 ATR18 人体解析和 SegFormer ATR18 模型，但不能因为代码仓库是 MIT 就推定权重与训练数据也可商用。未经书面确认的权重只允许隔离评估，不进入客户生产流程。

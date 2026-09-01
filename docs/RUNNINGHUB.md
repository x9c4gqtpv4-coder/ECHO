# ECHO 接入 RunningHub：四条可并行路径

四条路径共用同一个 `ECHOReferenceMatch` 节点合同，不复制另一套追色算法。

## A. RunningHub 云端原生节点

把 ECHO 公开节点包收录到 ComfyUI Registry，再申请 RunningHub 在云端环境安装/收录。
用户直接在 RunningHub 画布中使用：

```text
LoadImage(原图) + LoadImage(参考图)
                    ↓
        ECHO Reference Match
                    ↓
                SaveImage
```

优点：体验最简单，与生成流程一次完成。  
前置：RunningHub 云端必须已安装 ECHO 自定义节点；这是平台权限，GitHub 公开本身不等于云端已安装。  
费用：ECHO 本身不收 API 费，但 RunningHub 任务会按账户/会员规则消耗 RH 算力资源。

## B. Workflow API 批量调用

在 RunningHub 保存 A 路径的工作流，获得 `workflowId`，然后用官方高级接口：

```text
POST https://www.runninghub.ai/task/openapi/create
```

核心节点编号固定为：

- `1.image`：原图
- `2.image`：参考图
- `3.strength / adjustment_mode / transform_path / mask_backend`：ECHO 参数
- `4`：保存结果

参考文件：

- [API 格式工作流](../examples/runninghub/ECHO_reference_match_api.json)
- [nodeInfoList 覆盖示例](../examples/runninghub/node_overrides.example.json)
- `batch_color.runninghub.echo_node_overrides()`
- `batch_color.runninghub.advanced_task_payload()`

优点：最适合每个 SKU 一张参考图 + 5～7 张成品图的自动化，可使用轮询或 webhook。  
前置：和 A 一样，RunningHub 执行环境需要有 ECHO 节点。  
安全：API Key 只存在调用端/密钥管理中，禁止写入工作流、GitHub 或报告。

## C. Native ComfyUI Proxy

RunningHub 目前提供与本地 `http://127.0.0.1:8188` 兼容的代理入口：

```text
https://www.runninghub.ai/proxy/<API_KEY>
https://www.runninghub.ai/proxy-plus/<API_KEY>   # 48 GB 机型
```

现有的 ComfyUI API 客户端可将 server URL 切换到这个地址，然后提交
[API 格式工作流](../examples/runninghub/ECHO_reference_match_api.json)。

优点：本地 ComfyUI、Krita 或自建后端可保持同一套调用方式。  
前置：代理指向的 RunningHub 环境仍然必须包含 ECHO 节点。  
代码：`batch_color.runninghub.native_comfy_proxy()` 只生成地址，不记录密钥。

## D. RunningHub 生成 + ECHO 本地后处理（立即可用）

```text
RunningHub 生成任务 → 下载原图 → 本地 ECHO 追色 → 本地输出
```

这条路径不要求 RunningHub 安装自定义节点，现在就可以使用。RunningHub 完成生成后，
使用输出 URL 下载图片，再调用：

```bash
batch-color match \
  --input generated.png \
  --reference reference.png \
  --output ECHO-candidate.png \
  --mode both \
  --path auto
```

优点：不等平台收录；校色用 Mac/CPU，不为 ECHO 额外租 GPU；参考图可只留本地。  
代价：多一次下载和本地处理，不是 RunningHub 画布内单任务闭环。

## 建议同时保留的组合

1. **D 立即上线**：它不受平台节点审核限制。
2. **A 作为公开用户主入口**：完成 ComfyUI Registry 和 RunningHub 节点收录。
3. **B 作为 SKU 批处理主入口**：同一参考图多次复用，用 webhook/轮询归档。
4. **C 作为第三方软件兼容层**：不再额外维护一套专用协议。

当前不把 RunningHub API Key 或 `workflowId` 写入公开仓库。没有用户授权和云端
ECHO 节点安装确认时，不会假装 A/B/C 已在 RunningHub 成功执行。

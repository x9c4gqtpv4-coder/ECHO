# ECHO for ComfyUI

ECHO adds one local-first node to a ComfyUI graph:

`ECHO Reference Match / 回响·参考追色`

It receives a source `IMAGE` and a reference `IMAGE`, then returns the corrected
`IMAGE`, the background `MASK`, and a JSON review report.  It does not call a
paid API and does not need an NVIDIA GPU.

## Install

Clone the public repository into ComfyUI's custom-node directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/x9c4gqtpv4-coder/ECHO.git
```

Install the two dependencies with **the Python executable used by ComfyUI**:

```bash
/path/to/comfyui/python -m pip install -r ECHO/requirements.txt
```

For ComfyUI Desktop, use its environment/terminal dependency installer.  Do not
install into an unrelated system Python.  Restart ComfyUI after installation.

## Minimal workflow

```text
Load Image (source) ---- source     corrected ---- Save Image
                         ECHO
Load Image (reference) - reference  background_mask ---- Preview Mask
```

Drag [ECHO_reference_match_workflow.json](../examples/ECHO_reference_match_workflow.json)
onto the ComfyUI canvas, select the two images, then queue the graph.

## Controls

| Input | Meaning |
|---|---|
| `strength` | Overall blend strength, from identity (`0`) to the bounded ECHO candidate (`1`). |
| `adjustment_mode` | `background + person` follows the reference scene with one smooth subject transform; `background only` leaves non-background pixels byte-identical. |
| `transform_path` | `auto` compares safe global/spatial candidates; `global` is most conservative; `surface` only activates when both spatial fits have verified support. |
| `mask_backend` | `heuristic` is deterministic and cross-platform. `auto` uses the optional native macOS Vision helper when installed, otherwise falls back safely. |
| `source_background_mask` | Optional reviewed background mask: white is background, black is person/object. |
| `reference_background_mask` | Optional reviewed mask for the reference image. |
| `protect_mask` | Optional protection mask: white pixels remain exactly equal to the source. |

Mask geometry must match its image.  ECHO deliberately refuses to resize a
connected mask silently because that could move protection or subject edges.

One reference image is automatically broadcast across a source batch.  Reused
references are cached in memory (maximum eight profiles), so a workflow does
not repeat reference analysis for every image.

## Recommended public default

- `strength`: `0.85`
- `adjustment_mode`: `background + person`
- `transform_path`: `auto`
- `mask_backend`: `heuristic`

For complex scenes, connect reviewed source and reference background masks.
Connect `protect_mask` for books, text, logos, packaging, jewellery, or any
object whose pixels must not change.

Every result is a **review candidate**.  The JSON report never marks a result as
human-approved; production approval remains a separate decision.

## 中文要点

- 不需要云端算力、付费 API 或 NVIDIA 显卡。
- 原图和参考图均留在本地 ComfyUI。
- 默认将背景和人物向参考图的曝光、冷暖和色彩靠拢。
- 书本、商标、文字或商品不能变色时，连接 `protect_mask`。
- 复杂背景建议连接已复核的人物/背景蒙版，不要强行依赖轻量猜测。

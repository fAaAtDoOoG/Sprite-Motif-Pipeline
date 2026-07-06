# Sprite Motif Pipeline

[English README](README.en.md)

一个开源的 txt2img pipeline，用 Qwen-Image-2512 + Pixel Art LoRA 生成 2D 像素风游戏人物母题：先生成高分辨率母题图，默认 `1024x1024`，再用 nearest-neighbor 压到默认 `64x64`。默认目标是单个静态人物，朝右，适合继续整理成游戏 sprite。

## 功能

- 默认 prompt composer：用户输入一句简单描述，系统改写成适合像素角色母题的正负提示词。
- 可选外部 LLM：支持 OpenAI-compatible API 或 Ollama；没有 LLM 时使用内置规则和示例集。
- 直接 prompt 模式：用户可以绕过 prompt composer，把 prompt 直接送入 pipeline。
- 批次生成：一次生成多张候选图，保存高分辨率、低分辨率、ComfyUI API prompt 和 manifest。
- 选择与迭代：选择上一批中的候选图，并输入修改建议，系统改写 prompt 后生成下一批。
- ComfyUI 原生后端：导出并提交 Qwen-Image-2512 + Pixel Art LoRA API workflow。
- 开源合规：项目代码使用 Apache-2.0；不打包、不再分发任何模型权重。

## 安装

```powershell
$env:PYTHONUTF8="1"
uv venv
uv pip install -e ".[dev]"
```

Windows 路径里如果包含中文、空格或其他非 ASCII 字符，建议保留 `$env:PYTHONUTF8="1"`；否则 editable install 可能被系统默认编码绊住。

准备模型文件见 [docs/model_setup.md](docs/model_setup.md)。默认文件名：

- `qwen_image_2512_fp8_e4m3fn.safetensors`
- `qwen_2.5_vl_7b_fp8_scaled.safetensors`
- `qwen_image_vae.safetensors`
- `Qwen-Image-2512-Master-Pixel-Art-LoRA.safetensors`

## 使用

Windows 下可以直接双击根目录的 `start_gui.bat`。它会设置 UTF-8、检查 `uv`，并启动默认网页 GUI；如果 7865 端口上已经有服务在跑，它会直接打开现有页面。
网页顶部有 `Stop Server` 按钮可以手动关闭本地服务；默认勾选 `Auto stop on close`，关闭浏览器页面后一小段时间本地网页服务器会自动退出，刷新页面时通常不会误关。

启动 GUI（默认网页界面）：

```powershell
$env:PYTHONUTF8="1"
uv run spritepipe-gui
```

`spritepipe-gui` 会启动本地网页控制台并自动打开浏览器，默认地址为 `http://127.0.0.1:7865/`。也可以显式运行：

```powershell
uv run spritepipe-web --port 7865
```

网页 GUI 支持描述生成、直接 prompt、批量参数、候选预览、选择候选后反馈迭代，以及 ComfyUI 节点和模型文件校验。
默认 prompt model 会选择本地 Ollama 的 `qwen2.5:7b-instruct`。网页中的 `Validate Prompt Model` 会检查 Ollama 服务和模型，缺少模型时可直接拉取；下载和生成过程都会显示进度条与日志。网页 GUI 在 `Provider = ollama` 时会在预览、生成和迭代前先校验 prompt model，避免误以为用了 LLM 但实际走了 fallback；如果想显式使用内置规则，把 `Provider` 改成 `none`。
`Preview Prompt` 现在也会作为后台任务运行并更新顶部进度条，不会在等待本地 LLM 时看起来像没反应。默认情况下，pipeline 会给 Ollama prompt rewrite 请求发送 `keep_alive=0`，让 prompt model 用完后立即卸载；网页里也有 `Unload Prompt Model` 按钮可以手动释放。若你希望连续改 prompt 时更快、且机器内存足够，可以在启动 GUI 前设置 `$env:SPRITEPIPE_LLM_KEEP_ALIVE="5m"`。
生成完成后，候选预览默认显示左右对照：左侧 high-res，右侧把 low-res 按 pixel-perfect 方式放大到 high-res 同尺寸；预览器支持鼠标滚轮缩放、左键拖动，以及移动/放大/缩小按钮。
GUI 里的 `Models` 默认指向本机可发现的 ComfyUI `models` 文件夹。点击 `Validate ComfyUI` 后，如果缺少默认 safetensors，会询问是否自动下载到该文件夹。

旧版 Tk 桌面窗口仍然保留：

```powershell
uv run spritepipe-tk
```

启动 ComfyUI 后检查节点：
此命令也会检查默认模型文件名是否出现在 ComfyUI 后端下拉列表中。

```powershell
spritepipe validate-comfy --comfy-url http://127.0.0.1:8188
```

用自然语言描述生成一批候选：

```powershell
spritepipe generate --description "红发女骑士，轻甲，性格勇敢" --batch-size 4
```

改分辨率：

```powershell
spritepipe generate --description "森林法师，小个子，绿色斗篷" --high-res 1328x1328 --low-res 96x96 --batch-size 6
```

查看某次生成：

```powershell
spritepipe inspect runs\run_YYYYMMDD_HHMMSS
```

选择第 2 张候选并提出修改建议，生成下一批：

```powershell
spritepipe iterate runs\run_YYYYMMDD_HHMMSS --index 2 --feedback "盔甲更轻，头发更短，轮廓更圆润" --batch-size 4
```

直接输入 prompt：

```powershell
spritepipe generate --prompt "Pixel Art, 2D pixel art game character sprite motif, one original full-body desert rogue, static pose, centered, facing right, tan scarf, curved blade, plain neutral background, no readable text." --batch-size 4
```

只导出 ComfyUI API prompt，不实际生成：

```powershell
spritepipe workflow export --output workflows\qwen_image_2512_pixel_sprite_api.json
spritepipe generate --description "冰系少女，蓝白配色" --dry-run
```

## 外部 LLM Prompt Composer

默认不依赖在线 LLM。若要用其他 LLM 改写 prompt，可配置环境变量。

OpenAI-compatible:

```powershell
$env:SPRITEPIPE_LLM_PROVIDER="openai-compatible"
$env:SPRITEPIPE_LLM_ENDPOINT="https://api.example.com/v1/chat/completions"
$env:SPRITEPIPE_LLM_API_KEY="..."
$env:SPRITEPIPE_LLM_MODEL="your-chat-model"
```

Ollama:

```powershell
$env:SPRITEPIPE_LLM_PROVIDER="ollama"
$env:SPRITEPIPE_LLM_ENDPOINT="http://127.0.0.1:11434"
$env:SPRITEPIPE_LLM_MODEL="qwen2.5:7b-instruct"
```

默认 `SPRITEPIPE_LLM_KEEP_ALIVE="0"`，也就是每次 prompt rewrite 完成后释放 Ollama prompt model，降低内存占用。可以改成 `"5m"`、`"30m"` 或 `"-1"` 来保留热加载；这只影响 prompt LLM，不会卸载 ComfyUI 里的 Qwen-Image 主模型。

内置的 prompt curriculum 在 `src/sprite_motif_pipeline/prompt_training_examples.jsonl`。它既作为运行时 few-shot 示例，也可以作为后续微调 prompt LLM 的 starter 数据。

## 输出结构

每次生成会写入 `runs/run_YYYYMMDD_HHMMSS/`：

- `manifest.json`：描述、prompt、seed、选择和候选信息
- `api_prompts/*.json`：每张候选的 ComfyUI API prompt
- `highres/*.png`：ComfyUI 生成图
- `lowres/*.png`：nearest-neighbor 低分辨率图
- `contact_sheet.png`：低分辨率候选图对照表

## 模型与许可证说明

本仓库只包含 pipeline 代码、prompt 示例、workflow 模板和文档，不包含模型权重。

- Qwen-Image-2512：Qwen 团队发布，Apache-2.0。
- Qwen-Image-2512 Pixel Art LoRA：`prithivMLmods/Qwen-Image-2512-Pixel-Art-LoRA`，Apache-2.0，触发词为 `Pixel Art`。
- 本项目代码：Apache-2.0。

用户需要自行下载模型，并遵守模型、LoRA、ComfyUI、LLM 服务和生成内容所在地的适用条款。

## Contributors

- [@fAaAtDoOoG](https://github.com/fAaAtDoOoG)：项目发起者与需求设计。
- Codex：协作开发、实现与文档整理。

## 开发

```powershell
uv run pytest -q
```

上游 ComfyUI workflow 备份在 `workflows/upstream/image_qwen_Image_2512.upstream.json`。项目运行时使用 `src/sprite_motif_pipeline/workflow.py` 生成更小的 API prompt，便于批量参数化和自动化。

# Sprite Motif Pipeline

[English README](README.en.md)

一个开源的 txt2img pipeline，用 Qwen-Image-2512 + Pixel Art LoRA 生成 2D 像素风游戏人物母题：先生成高分辨率母题图，默认 `1024x1024`，再用 nearest-neighbor 压到默认 `64x64`。默认目标是单个静态人物，朝右，适合继续整理成游戏 sprite。

## 功能

- 默认 prompt composer：用户输入一句简单描述，系统改写成适合像素角色母题的正负提示词。
- 可选外部 LLM：支持 OpenAI-compatible API 或 Ollama；没有 LLM 时使用通用的内置 fallback composer。
- 直接 prompt 模式：用户可以绕过 prompt composer，把 prompt 直接送入 pipeline。
- 批次生成：一次生成多张候选图，保存高分辨率、低分辨率、ComfyUI API prompt 和 manifest。
- 选择与迭代：选择上一批中的候选图，并输入修改建议，系统改写 prompt 后生成下一批。
- ComfyUI 原生后端：导出并提交 Qwen-Image-2512 + Pixel Art LoRA API workflow。
- 可选图片后端：保留内置 Qwen-Image-2512，也可加载任意 ComfyUI API 格式的纯 txt2img 工作流，或连接 OpenAI-compatible Images API。
- 可选模型：网页可直接填写本地模型标识、API 模型名、endpoint，以及仅在当前请求中使用的 API key。
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

网页 GUI 支持描述生成、直接 prompt、批量参数、候选预览、选择候选后反馈迭代，以及 ComfyUI 节点和模型文件校验。默认使用 Qwen-Image-2512 + Pixel Art LoRA 的纯 txt2img 工作流；用户也可以选择自定义 ComfyUI 或 Images API 后端。选择候选只把其既有 prompt 作为 LLM 的迭代上下文，不会把图片传回生成模型。
启动网页 GUI 本身不会启动 ComfyUI 或 Ollama。点击 `Generate` 或 `Iterate Selected` 后，pipeline 会先启动 Ollama，在一次请求中扩写并自检整批正负 prompt，然后关闭本次任务启动的 Ollama；内置 Qwen 或自定义 ComfyUI 后端随后才启动 ComfyUI，生成完毕后关闭本次任务启动的 ComfyUI。Images API 后端会跳过 ComfyUI，直接在 prompt 阶段之后发出 API 请求。`Preview Prompt` 只执行 Ollama 阶段。直接 prompt 模式跳过 Ollama。
默认 prompt model 是本地 Ollama 的 `qwen3:32b`，请求默认使用 `num_gpu=999`、`num_ctx=4096`、`num_predict=1024`、`temperature=0.55`。Prompt LLM 不再只做逐字翻译：它会保留所有明确要求，自行判断 positive/negative，并对用户没有写明的轮廓、比例、形态节奏、特征布局、明暗组织和表面表现进行相容的推测。每个批次候选都会得到独立的一组 positive/negative prompt；默认批次为 4，因此会得到 4 个有实质差异的视觉方向，而不是只更换 seed。推测可以丰富角色，但不能与明确要求冲突或替换用户指定的身份。代码没有角色范例、固定中文关键词或针对某个角色的硬编码。
`Validate Prompt Model` 会临时启动 Ollama 完成检查，然后自动退出本次启动的服务；缺少模型时可直接拉取。`Preview Prompt` 作为后台任务运行并更新顶部进度条。候选扩写与事实自检在同一次模型请求中完成，请求结束后立即卸载 prompt model。
Backend 区域仍保留 `Start ComfyUI` 和 `Start Ollama` 作为手动诊断入口。为了不误杀用户的其他任务，pipeline 只终止自己启动的进程；如果检测到用户已经手动启动或由其他程序启动的服务，就复用该服务、在阶段结束时卸载相关模型，但保留外部服务进程。
生成完成后，候选预览默认显示左右对照：左侧 high-res，右侧把 low-res 按 pixel-perfect 方式放大到 high-res 同尺寸；预览器支持鼠标滚轮缩放、左键拖动，以及移动/放大/缩小按钮。
GUI 里的 `Models` 默认指向本机可发现的 ComfyUI `models` 文件夹。点击 `Validate ComfyUI` 后，如果缺少默认 safetensors，会询问是否自动下载到该文件夹。

## 图片生成后端

网页 `Backend` 区域提供三种后端：

- `Built-in Qwen-Image-2512`：默认的 ComfyUI + Pixel Art LoRA 方案；Image model 填写 ComfyUI 中显示的 diffusion model 文件名。
- `Custom ComfyUI workflow`：使用本地部署的其他 txt2img 模型，并由用户提供 ComfyUI API 格式 JSON。
- `OpenAI-compatible Images API`：连接托管 API 或实现 [`POST /v1/images/generations`](https://developers.openai.com/api/reference/resources/images/methods/generate) 的本地服务；支持读取 `data[0].b64_json` 和 `data[0].url`。

自定义 ComfyUI 工作流必须通过 ComfyUI 的 **Save (API Format)** 导出。把希望由 SpritePipe 注入的字段替换为以下占位符：

`{{positive_prompt}}`、`{{negative_prompt}}`、`{{width}}`、`{{height}}`、`{{seed}}`、`{{steps}}`、`{{cfg}}`、`{{filename_prefix}}`、`{{model}}`、`{{lora_name}}`、`{{lora_strength}}`。

占位符单独作为一个 JSON 字符串值时会保留数字类型，例如 `"seed": "{{seed}}"`；嵌入较长字符串时按文本替换。工作流需要自行包含模型加载、采样、解码和保存节点，并且必须保持纯 txt2img。生成前，pipeline 会校验该 JSON 需要的节点是否存在于当前 ComfyUI。

网页中的图片 API key 和 prompt API key 只随当前请求进入内存，不会写入 manifest、导出的请求 JSON、日志或浏览器存储。CLI 推荐使用环境变量：

```powershell
$env:SPRITEPIPE_IMAGE_BACKEND="openai-images"
$env:SPRITEPIPE_IMAGE_ENDPOINT="https://api.example.com/v1/images/generations"
$env:SPRITEPIPE_IMAGE_API_KEY="..."
$env:SPRITEPIPE_IMAGE_MODEL="your-image-model"
spritepipe generate --description "森林法师，小个子，绿色斗篷"
```

标准 Images API 没有独立 negative prompt 字段，因此 pipeline 会把 negative 概念作为明确的 avoid 指令追加到请求 prompt。不同本地兼容服务支持的尺寸和参数可能不同；这里的 compatible 只表示它遵循上述 HTTP 请求与响应结构。

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

使用自定义本地 ComfyUI 模型：

```powershell
spritepipe generate --description "森林法师" --image-backend custom-comfy --custom-workflow workflows\my_model_api.json --image-model my_model.safetensors
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
$env:SPRITEPIPE_LLM_MODEL="qwen3:32b"
$env:SPRITEPIPE_OLLAMA_NUM_GPU="999"
$env:SPRITEPIPE_OLLAMA_NUM_CTX="4096"
$env:SPRITEPIPE_OLLAMA_NUM_PREDICT="1024"
```

默认 `SPRITEPIPE_LLM_KEEP_ALIVE="0"`。pipeline 自己启动 Ollama 时会在整个 prompt 阶段结束后关闭该服务；复用外部 Ollama 时则会卸载本次使用的 prompt model，但不会终止外部进程。
默认 `SPRITEPIPE_LLM_TIMEOUT="900"`，适配 `qwen3:32b` 的冷启动；网页 Prompt Model 区域也可以直接修改 GPU layers、Context、Max tokens 和 Thinking。

Prompt LLM 不会注入角色范例或固定角色词表。它只接收当前描述、候选数量、已选候选的 prompt 和用户反馈，由 `qwen3:32b` 自行判断正向和负向概念，并为每张候选扩写一组不同但相容的 txt2img 视觉提示词。迭代时，每组都保留已选设计和未受反馈影响的特征，同时用不同方式揣测并实现修改建议。

## 输出结构

每次生成会写入 `runs/run_YYYYMMDD_HHMMSS/`：

- `manifest.json`：描述、用户原始输入历史、prompt、seed、选择和候选信息
- `api_prompts/*.json`：每张候选解析后的 ComfyUI prompt 或 Images API 请求，不包含密钥
- `highres/*`：图片后端生成的高分辨率图
- `lowres/*.png`：nearest-neighbor 低分辨率图
- `contact_sheet.png`：低分辨率候选图对照表

## 模型与许可证说明

本仓库只包含 pipeline 代码、workflow 模板和文档，不包含模型权重。

- Qwen-Image-2512：Qwen 团队发布，Apache-2.0。
- Qwen-Image-2512 Pixel Art LoRA：`prithivMLmods/Qwen-Image-2512-Pixel-Art-LoRA`，Apache-2.0，触发词为 `Pixel Art`。
- 本项目代码：Apache-2.0。

用户需要自行下载模型，并在使用或再分发前以各自模型卡和许可证文本为准，遵守模型、LoRA、ComfyUI、LLM 服务和生成内容所在地的适用条款。

自定义模型和外部 API 均由用户自行选择，本项目不对其背书、打包或重新授权。使用前请单独核验模型许可证、API 条款、商业使用限制、隐私规则和生成内容权利。

## Contributors

- [@fAaAtDoOoG](https://github.com/fAaAtDoOoG)：项目发起者与需求设计。
- Codex：协作开发、实现与文档整理。

## 开发

```powershell
uv run pytest -q
```

上游 ComfyUI workflow 备份在 `workflows/upstream/image_qwen_Image_2512.upstream.json`。项目运行时使用 `src/sprite_motif_pipeline/workflow.py` 生成更小的 API prompt，便于批量参数化和自动化。

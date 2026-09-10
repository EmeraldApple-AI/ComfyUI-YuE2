# ComfyUI-YuE2

Unofficial ComfyUI nodes for **[YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)** — lyrics + style → 48 kHz stereo song with an editable ABC score.

## Disclaimers

- **Not official.** This pack is not made by, endorsed by, or affiliated with Multimodal Art Projection (M·A·P), Hugging Face, or ComfyUI / Comfy-Org.
- **Not [ComfyUI_YuE](https://github.com/smthemex/ComfyUI_YuE).** That project is YuE **v1** (`s1` + `s2`). YuE2 is a different model and a different API. Do not mix the two node packs in one graph.
- **Weights are CC BY-NC 4.0.** `m-a-p/YuE2-3B` and `m-a-p/YuE2-Vae` are non-commercial. Do not use the model, this wrapper, or outputs in a product you sell unless you have a separate license from M·A·P. The MIT license on **this repo** covers only the node source, not the weights.
- **Experimental.** Generation can fail, truncate, ignore style, or sound unlike the prompt. There is **no duration slider**. Length is a hint from lyrics, sections, BPM, and token caps.
- **You are responsible for outputs.** Do not treat generated lyrics or audio as cleared for copyright, likeness, or platform upload.
- **Hardware.** Official path is BF16 on NVIDIA. RTX 50-series needs PyTorch built with CUDA **12.8+** (`sm_120`). Official Windows wheels usually **lack Flash Attention**. Use `torch-eager` on Windows. That is slower and expected.
- **Windows encoding.** Korean / Japanese / Chinese lyrics can crash stock YuE2 `plan.json` writes (`cp1252`). These nodes force UTF-8 writes and fall back to `score.abc` + `lyrics.txt` + `audio.wav`.
- **No warranty.** Provided as-is.

## 1. Clone the nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/EmeraldApple-AI/ComfyUI-YuE2.git
```

Use the **same Python that launches ComfyUI** for every pip command below.

| Setup | Python to use |
|---|---|
| Windows portable | `ComfyUI_windows_portable\python_embeded\python.exe` |
| venv / manual | `path/to/venv/Scripts/python.exe` or `bin/python` |

Below, `PY` means that interpreter.

## 2. Install the correct PyTorch

YuE2 needs **CUDA PyTorch**, not the CPU wheel. A 50-series card (5090 / 5080 / 5070) also needs **CUDA 12.8** kernels. CUDA 12.4 / 12.6 wheels often start Comfy then die with `Torch not compiled with CUDA enabled` or missing `sm_120`.

### Check what you have

```bat
PY -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('cap', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
```

You want something like:

```
torch 2.7.x+cu128   (or newer cu128 / cu129)
cuda True
cap (12, 0)         RTX 5090 = sm_120
```

If `cuda False` or version has no `cu128`/`cu129` on a 50-series card, reinstall:

### Windows portable (RTX 40 / 50)

Run from `ComfyUI_windows_portable`:

```bat
python_embeded\python.exe -m pip uninstall -y torch torchvision torchaudio
python_embeded\python.exe -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Nightlies if the stable cu128 index does not list your version yet:

```bat
python_embeded\python.exe -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

Do **not** grab a CPU wheel from PyPI (`pip install torch` with no index). That is what produces `AssertionError: Torch not compiled with CUDA enabled`.

### Linux / venv

```bash
"$PY" -m pip uninstall -y torch torchvision torchaudio
"$PY" -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

RTX 4090 / 3090 can also use cu124 or cu126 if that is already what Comfy shipped. 50-series should stay on cu128+.

### Flash Attention (read this)

YuE2's `backend=torch` calls `aten._flash_attention_forward`. Official **Windows** PyTorch wheels almost always ship **without** `USE_FLASH_ATTENTION`. The op exists and then crashes:

```
RuntimeError: USE_FLASH_ATTENTION was not enabled for build.
```

- On Windows portable: leave Loader backend at **`torch-eager`**. That is the supported path.
- Do **not** `pip install flash-attn` to "fix" it. That is a different package. Broken wheels have taken down portable Python (`0xc0000139`).
- On Linux, a source-built PyTorch with Flash enabled can use `backend=torch`. If generate throws the error above, switch to `torch-eager`.

## 3. Install YuE2 into that same Python

```bat
git clone https://github.com/multimodal-art-projection/YuE.git
PY -m pip install .\YuE
PY -m pip install huggingface-hub soundfile numpy
```

Or the published wheel from the model repo:

```bat
PY -m pip install huggingface-hub
PY -m huggingface_hub.commands.huggingface_cli download m-a-p/YuE2-3B yue2_infer-0.1.5-py3-none-any.whl --local-dir .
PY -m pip install .\yue2_infer-0.1.5-py3-none-any.whl
```

Wheel filename / version can change — check [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B).

## 4. Windows-only environment

Hugging Face cache uses symlinks. Without Developer Mode that raises `WinError 1314`. Korean lyrics need UTF-8 or `plan.json` dies on `cp1252`.

Put these in `run_nvidia_gpu.bat` before `python.exe` starts Comfy, or set them in the user environment:

```bat
set HF_HUB_DISABLE_SYMLINKS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
```

Latest NVIDIA Game Ready or Studio driver. If Comfy prints a `c10.dll` error, install [VC++ redistributable](https://aka.ms/vc14/vc_redist.x64.exe).

## 5. First run

1. Restart Comfy after the pip installs.
2. Add **YuE2 Loader** (`audio → YuE2`). Model `m-a-p/YuE2-3B`, VAE `m-a-p/YuE2-Vae`. The nodes will automatically download the YUE2 models.
3. Windows: backend **`torch-eager`**. Linux with working Flash: `torch` is optional.
4. Connect **YuE2 Generate** → Preview / Save Audio.
5. First queue downloads several GB from Hugging Face.

Confirm in the console:

```
[YuE2] Pipeline backend='torch-eager'
```

Extra notes: `prompts/yue2_comfyui_settings.txt`  
LLM prompt compiler: `prompts/yue2_prompt_generator.txt`  
Example graph: `example_workflows/yue2_generate.json`

## Nodes

| Node | Role |
|---|---|
| YuE2 Loader | Load / cache `m-a-p/YuE2-3B` + VAE |
| YuE2 Style Prompt | Build the style line |
| YuE2 Lyrics Template | Section skeleton |
| YuE2 Generate | Style + lyrics → AUDIO |
| YuE2 Plan Score | ABC only |
| YuE2 Save / Load Text | ABC or lyrics files |
| YuE2 Unload | Free VRAM |

## License

- Node pack source in this repository: MIT
- YuE2 inference code and weights: CC BY-NC 4.0 © Multimodal Art Projection

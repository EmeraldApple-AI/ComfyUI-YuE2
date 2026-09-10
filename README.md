# ComfyUI-YuE2

Unofficial ComfyUI nodes for **[YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)** — lyrics + style → 48 kHz stereo song with an editable ABC score.

## Disclaimers

- **Not official.** This pack is not made by, endorsed by, or affiliated with Multimodal Art Projection (M·A·P), Hugging Face, or ComfyUI / Comfy-Org.
- **Not [ComfyUI_YuE](https://github.com/smthemex/ComfyUI_YuE).** That project is YuE **v1** (`s1` + `s2`). YuE2 is a different model and a different API. Do not mix the two node packs in one graph.
- **Weights are CC BY-NC 4.0.** `m-a-p/YuE2-3B` and `m-a-p/YuE2-Vae` are non-commercial. Do not use the model, this wrapper, or outputs in a product you sell unless you have a separate license from M·A·P. The MIT license on **this repo** covers only the node source, not the weights.
- **Experimental.** Generation can fail, truncate, ignore style, or sound unlike the prompt. There is **no duration slider**. Length is a hint from lyrics, sections, BPM, and token caps.
- **You are responsible for outputs.** Do not treat generated lyrics or audio as cleared for copyright, likeness, or platform upload. Do not prompt for living artists as the only style string.
- **Hardware.** Official path is BF16 on NVIDIA. RTX 50-series needs a PyTorch build with CUDA 12.8+ (`sm_120`). Official Windows wheels usually **lack Flash Attention**. This pack forces `torch-eager` on Windows. That is slower and expected.
- **Windows encoding.** Saving Korean / Japanese / Chinese lyrics can crash stock YuE2 `plan.json` writes (`cp1252`). The nodes force UTF-8 file writes and fall back to `score.abc` + `lyrics.txt` + `audio.wav` if needed.
- **No warranty.** Provided as-is. Issues and breakage are on you / PRs; this is a community wrapper around a research model.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/EmeraldApple-AI/ComfyUI-YuE2.git
```

Install official YuE2 into the **same Python that launches ComfyUI**:

```bash
git clone https://github.com/multimodal-art-projection/YuE.git
"<ComfyUI python>" -m pip install ./YuE
```

### Windows portable

```bat
cd ComfyUI_windows_portable
python_embeded\python.exe -m pip install huggingface-hub
git clone https://github.com/multimodal-art-projection/YuE.git
python_embeded\python.exe -m pip install .\YuE
set HF_HUB_DISABLE_SYMLINKS=1
set PYTHONUTF8=1
```

On **YuE2 Loader** leave backend at `torch-eager`. Do not install random `flash-attn` wheels to "fix" it; that is a different stack and has crashed portable Python.

- Settings: `prompts/yue2_comfyui_settings.txt`
- LLM compiler: `prompts/yue2_prompt_generator.txt`
- Example graph: `example_workflows/yue2_generate.json`

Restart Comfy after replacing `nodes.py`. Look for `YuE2 Loader` under **audio → YuE2**.

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

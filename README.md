# ComfyUI-YuE2

Unofficial ComfyUI nodes for **[YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)** — lyrics + style prompt → 48 kHz stereo song with an editable ABC score.

This is **not** [ComfyUI_YuE](https://github.com/smthemex/ComfyUI_YuE) (YuE v1). YuE2 weights are **CC BY-NC 4.0**.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/EmeraldApple-AI/ComfyUI-YuE2.git
```

Install official YuE2 into the same Python that runs ComfyUI:

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
```

On **YuE2 Loader** set backend to `torch-eager`. Windows official PyTorch wheels usually lack Flash Attention.

- Settings: `prompts/yue2_comfyui_settings.txt`
- LLM compiler: `prompts/yue2_prompt_generator.txt`

## Nodes

YuE2 Loader, Style Prompt, Lyrics Template, Generate, Plan Score, Save/Load Text, Unload.

## License

Node pack: MIT. YuE2 code + weights: CC BY-NC 4.0 from Multimodal Art Projection.

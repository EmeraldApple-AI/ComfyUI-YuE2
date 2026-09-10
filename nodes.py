"""ComfyUI nodes wrapping the official yue2.YuE2Pipeline."""
from __future__ import annotations
import os, sys, traceback
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    _path_write_text = Path.write_text
    def _write_text_utf8(self, data, encoding=None, errors=None, newline=None):
        if encoding is None:
            encoding = "utf-8"
        return _path_write_text(self, data, encoding=encoding, errors=errors, newline=newline)
    Path.write_text = _write_text_utf8

import numpy as np
import torch
try:
    import folder_paths
except ImportError:
    folder_paths = None

CATEGORY = "audio/YuE2"
DEFAULT_MODEL = "m-a-p/YuE2-3B"
DEFAULT_VAE = "m-a-p/YuE2-Vae"
_PIPELINES = {}

def _yue2_import():
    try:
        from yue2 import YuE2Pipeline
    except ImportError as exc:
        raise ImportError("YuE2 is not installed. pip install the official YuE repo into ComfyUI python.") from exc
    return YuE2Pipeline

def _output_dir():
    if folder_paths is not None:
        return Path(folder_paths.get_output_directory())
    return Path("output")

def _flash_attn_usable():
    if sys.platform == "win32" or not torch.cuda.is_available():
        return False
    try:
        q = torch.zeros(1, 1, 1, 32, device="cuda", dtype=torch.float16)
        torch.nn.functional.scaled_dot_product_attention(q, q, q)
        torch.ops.aten._flash_attention_forward
        return True
    except Exception:
        return False

def _apply_safe_backend(runtime):
    want = getattr(runtime, "backend", None) or "torch-eager"
    if not _flash_attn_usable():
        want = "torch-eager"
    try:
        runtime.backend = want
    except Exception:
        pass
    return getattr(runtime, "backend", want)

def _invoke_runtime(runtime, **kwargs):
    try:
        return runtime(**kwargs)
    except RuntimeError as exc:
        if "FLASH_ATTENTION" in str(exc) or "flash_attention" in str(exc):
            print("[YuE2] Flash Attention failed; retrying with backend=torch-eager")
            runtime.backend = "torch-eager"
            return runtime(**kwargs)
        raise

def _normalize_backend(backend):
    name = (backend or "torch-eager").strip()
    if name != "torch-eager" and (sys.platform == "win32" or not _flash_attn_usable()):
        print("[YuE2] Forcing backend=torch-eager")
        return "torch-eager"
    return name

def _pipeline_key(model, vae, device, memory_budget_gib, backend):
    return f"{model}|{vae}|{device}|{memory_budget_gib:g}|{backend}"

def _get_pipeline(model, vae, device, memory_budget_gib, backend, progress):
    YuE2Pipeline = _yue2_import()
    backend = _normalize_backend(backend)
    key = _pipeline_key(model, vae, device, memory_budget_gib, backend)
    pipe = _PIPELINES.get(key)
    if pipe is not None:
        if getattr(pipe, "backend", None) != "torch-eager" and backend == "torch-eager":
            pipe.backend = "torch-eager"
        return pipe
    if _PIPELINES:
        _close_all()
    kwargs = dict(memory_budget_gib=float(memory_budget_gib), backend=backend, progress=bool(progress))
    if device != "auto":
        kwargs["device"] = device
    pipe = YuE2Pipeline.from_pretrained(model, vae=vae, **kwargs)
    try:
        pipe.backend = backend
    except Exception:
        pass
    print(f"[YuE2] Pipeline backend={getattr(pipe, 'backend', backend)!r}")
    _PIPELINES[key] = pipe
    return pipe

def _close_all():
    for pipe in list(_PIPELINES.values()):
        try:
            pipe.close()
        except Exception:
            pass
    _PIPELINES.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _song_to_audio(audio_np, sample_rate):
    arr = np.asarray(audio_np)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 8:
        channels_first = arr
    else:
        channels_first = arr.T
    waveform = torch.from_numpy(np.ascontiguousarray(channels_first)).float().unsqueeze(0)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}

def _clean_abc(abc):
    if abc is None:
        return None
    text = str(abc).strip()
    return text if text else None

def _cfg_or_none(cfg_scale):
    if cfg_scale is None or float(cfg_scale) <= 0:
        return None
    return float(cfg_scale)

def _status_text(truncated, extra=""):
    parts = []
    if isinstance(truncated, dict):
        bad = [k for k, v in truncated.items() if v]
        parts.append("truncated=" + (",".join(bad) if bad else "none"))
    elif truncated:
        parts.append("truncated=yes")
    else:
        parts.append("truncated=none")
    if extra:
        parts.append(extra)
    return " | ".join(parts)

LANGUAGES = ["English","Mandarin","Cantonese","Japanese","Korean","Spanish","French","Portuguese","German","Italian","Hindi","(custom / none)"]
GENRES = ["pop","piano pop","indie pop","synth pop","dance pop","r&b","soul","funk","nu-disco","disco","house","edm","hip hop","trap","rap","rock","indie rock","alternative rock","punk","metal","jazz","jazz-funk","bossa nova","latin","reggaeton","country","folk","indie folk","singer-songwriter","acoustic","gospel","blues","lo-fi","dream pop","shoegaze","ambient","cinematic","film score","classical crossover","k-pop","j-pop","city pop","c-pop","afrobeats","reggae","ska","(custom / none)"]
MOODS = ["(none)","uplifting","euphoric","romantic","intimate","melancholic","bittersweet","dark","aggressive","playful","nostalgic","dreamy","cinematic","chill","anthemic","mysterious"]
VOCALS = ["(instrumental / no lead vocal)","female vocal","male vocal","androgynous vocal","duet male and female","choir / stacked vocals","rap lead","spoken word"]
VOCAL_COLORS = ["(none)","warm","bright","breathy","powerful","raspy","soft intimate","belting","tenor","baritone","soprano","alto"]
ENERGIES = ["(none)","low energy","mid energy","high energy","builds from quiet to loud"]
LENGTH_HINTS = {"model decides":"","short clip (~45-75s)":"short song, about one minute","single edit (~2-3 min)":"radio-length song, about two to three minutes","full song (~3.5-5 min)":"full-length song, about four minutes","very long (may truncate)":"long-form song with extended sections"}
LYRIC_TEMPLATES = {
    "blank": "",
    "verse + chorus": "[Verse]\nLine one of the verse\nLine two of the verse\n\n[Chorus]\nSing the hook here\nRepeat the title line",
    "V-C-V-C": "[Verse]\nFirst verse line\n\n[Chorus]\nMain hook\n\n[Verse]\nSecond verse line\n\n[Chorus]\nMain hook",
    "V-C-V-C-bridge-C": "[Verse]\nFirst verse\n\n[Chorus]\nHook\n\n[Verse]\nSecond verse\n\n[Chorus]\nHook\n\n[Bridge]\nContrast the chorus here\n\n[Chorus]\nFinal hook",
}

def _omit(value):
    return not value or value.startswith("(")

def build_style_prompt(language, genre, mood, vocal, vocal_color, instruments, tempo_bpm, energy, length_hint, extra, custom_language="", custom_genre=""):
    parts = []
    lang = custom_language.strip() if language.startswith("(custom") else language
    if lang:
        parts.append(lang)
    gen = custom_genre.strip() if genre.startswith("(custom") else genre
    if gen:
        parts.append(gen)
    if not _omit(mood):
        parts.append(mood)
    if not _omit(vocal):
        color = "" if _omit(vocal_color) else vocal_color + " "
        parts.append(color + vocal)
    elif not _omit(vocal_color):
        parts.append(vocal_color + " instrumental")
    inst = (instruments or "").strip()
    if inst:
        parts.append(inst)
    if int(tempo_bpm) > 0:
        parts.append(f"{int(tempo_bpm)} BPM")
    if not _omit(energy):
        parts.append(energy)
    hint = LENGTH_HINTS.get(length_hint, "")
    if hint:
        parts.append(hint)
    extra_txt = (extra or "").strip()
    if extra_txt:
        parts.append(extra_txt)
    return ", ".join(parts)

class YuE2Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("STRING", {"default": DEFAULT_MODEL}),
            "vae": ("STRING", {"default": DEFAULT_VAE}),
            "device": (["auto", "cuda", "cuda:0", "cpu"],),
            "memory_budget_gib": ("FLOAT", {"default": 24.0, "min": 8.0, "max": 80.0, "step": 1.0}),
            "backend": (["torch-eager", "torch"],),
            "progress": ("BOOLEAN", {"default": True}),
        }}
    RETURN_TYPES = ("YUE2_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    def load(self, model, vae, device, memory_budget_gib, backend, progress):
        backend = _normalize_backend(backend)
        pipe = _get_pipeline(model.strip(), vae.strip() or DEFAULT_VAE, device, memory_budget_gib, backend, progress)
        handle = {"model": model.strip(), "vae": vae.strip() or DEFAULT_VAE, "device": device, "backend": getattr(pipe, "backend", backend), "_pipe": pipe}
        return (handle,)

class YuE2Unload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": ("YUE2_PIPE",)}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    def unload(self, pipe):
        _close_all()
        if isinstance(pipe, dict):
            pipe["_pipe"] = None
            pipe["closed"] = True
        return ("unloaded",)

class YuE2StylePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "language": (LANGUAGES, {"default": "English"}),
            "genre": (GENRES, {"default": "piano pop"}),
            "mood": (MOODS, {"default": "intimate"}),
            "vocal": (VOCALS, {"default": "female vocal"}),
            "vocal_color": (VOCAL_COLORS, {"default": "warm"}),
            "instruments": ("STRING", {"multiline": True, "default": "acoustic piano, rounded bass and light drums"}),
            "tempo_bpm": ("INT", {"default": 88, "min": 0, "max": 220}),
            "energy": (ENERGIES, {"default": "mid energy"}),
            "length_hint": (list(LENGTH_HINTS.keys()), {"default": "single edit (~2-3 min)"}),
            "extra": ("STRING", {"multiline": True, "default": "lyrical memorable melody, unhurried phrasing"}),
        }, "optional": {"custom_language": ("STRING", {"default": ""}), "custom_genre": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("style",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    def build(self, language, genre, mood, vocal, vocal_color, instruments, tempo_bpm, energy, length_hint, extra, custom_language="", custom_genre=""):
        return (build_style_prompt(language, genre, mood, vocal, vocal_color, instruments, tempo_bpm, energy, length_hint, extra, custom_language, custom_genre),)

class YuE2LyricsTemplate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"template": (list(LYRIC_TEMPLATES.keys()), {"default": "V-C-V-C-bridge-C"}), "lyrics_override": ("STRING", {"multiline": True, "default": ""})}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    def build(self, template, lyrics_override):
        text = (lyrics_override or "").strip()
        return (lyrics_override if text else LYRIC_TEMPLATES.get(template, ""),)

class YuE2Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pipe": ("YUE2_PIPE",),
            "style": ("STRING", {"multiline": True, "default": "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM, radio-length song, about two to three minutes"}),
            "lyrics": ("STRING", {"multiline": True, "default": LYRIC_TEMPLATES["verse + chorus"]}),
            "cot": (["full", "melody", "off"], {"default": "full"}),
            "seed": ("INT", {"default": 831001, "min": 0, "max": 0x7FFFFFFF, "control_after_generate": True}),
            "cfg_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "override_sampling": ("BOOLEAN", {"default": False}),
            "semantic_max_tokens": ("INT", {"default": 9000, "min": 200, "max": 16000, "step": 100}),
            "semantic_min_tokens": ("INT", {"default": 200, "min": 0, "max": 8000, "step": 50}),
            "abc_max_tokens": ("INT", {"default": 4096, "min": 32, "max": 8192, "step": 32}),
            "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 5.0, "step": 0.05}),
            "top_p": ("FLOAT", {"default": 0.9, "min": 0.05, "max": 1.0, "step": 0.01}),
            "top_k": ("INT", {"default": 50, "min": 1, "max": 500}),
            "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.1, "max": 2.0, "step": 0.01}),
            "save_artifacts": ("BOOLEAN", {"default": True}),
            "filename_prefix": ("STRING", {"default": "yue2/song"}),
        }, "optional": {"abc": ("STRING", {"multiline": True, "default": ""})}}
    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "abc", "style_used", "status")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    def generate(self, pipe, style, lyrics, cot, seed, cfg_scale, override_sampling, semantic_max_tokens, semantic_min_tokens, abc_max_tokens, temperature, top_p, top_k, repetition_penalty, save_artifacts, filename_prefix, abc=""):
        runtime = pipe.get("_pipe") if isinstance(pipe, dict) else None
        if runtime is None:
            raise RuntimeError("Connect YuE2 Loader first (or re-run Loader after Unload).")
        used = _apply_safe_backend(runtime)
        print(f"[YuE2] generate using backend={used!r}")
        abc_text = _clean_abc(abc)
        if abc_text is not None and cot == "off":
            raise ValueError("A supplied ABC score requires cot=full or cot=melody.")
        kwargs = dict(style=style, lyrics=lyrics, cot=cot, seed=int(seed))
        cfg = _cfg_or_none(cfg_scale)
        if cfg is not None:
            kwargs["cfg_scale"] = cfg
        if abc_text is not None:
            kwargs["abc"] = abc_text
        if override_sampling:
            kwargs["semantic_sampling"] = {"temperature": float(temperature), "top_p": float(top_p), "top_k": int(top_k), "repetition_penalty": float(repetition_penalty), "min_tokens": int(semantic_min_tokens), "max_tokens": int(semantic_max_tokens)}
            kwargs["abc_sampling"] = {"temperature": 0.7, "top_p": 0.9, "top_k": 30, "repetition_penalty": 1.005, "penalty_window": 100, "min_tokens": 32, "max_tokens": int(abc_max_tokens)}
        try:
            song = _invoke_runtime(runtime, **kwargs)
        except Exception:
            traceback.print_exc()
            raise
        audio = _song_to_audio(song.audio, song.sample_rate)
        score = song.abc or ""
        extra = "no duration API; length ~ lyrics/sections + BPM + token cap"
        if save_artifacts:
            out = _unique_dir(_output_dir() / filename_prefix)
            try:
                result = song.save_artifacts(out)
            except UnicodeEncodeError:
                out.mkdir(parents=True, exist_ok=True)
                if score:
                    (out / "score.abc").write_text(score, encoding="utf-8")
                (out / "lyrics.txt").write_text(str(lyrics), encoding="utf-8")
                extra = f"saved={out} (utf-8 fallback) | {extra}"
            else:
                seconds = result.get("audio_seconds", "?") if isinstance(result, dict) else "?"
                extra = f"saved={out} seconds={seconds} | {extra}"
        return (audio, score, style, _status_text(song.truncated, extra))

class YuE2Plan:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pipe": ("YUE2_PIPE",),
            "style": ("STRING", {"multiline": True, "default": "English, jazz-funk, warm lead vocal"}),
            "lyrics": ("STRING", {"multiline": True, "default": "[Verse]\nWrite lyrics here\n\n[Chorus]\nWrite the hook here"}),
            "cot": (["full", "melody"], {"default": "full"}),
            "seed": ("INT", {"default": 831001, "min": 0, "max": 0x7FFFFFFF, "control_after_generate": True}),
        }, "optional": {"abc": ("STRING", {"multiline": True, "default": ""})}}
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("abc", "status")
    FUNCTION = "plan"
    CATEGORY = CATEGORY
    def plan(self, pipe, style, lyrics, cot, seed, abc=""):
        runtime = pipe.get("_pipe") if isinstance(pipe, dict) else None
        if runtime is None:
            raise RuntimeError("Connect YuE2 Loader first (or re-run Loader after Unload).")
        _apply_safe_backend(runtime)
        kwargs = dict(style=style, lyrics=lyrics, cot=cot, seed=int(seed))
        abc_text = _clean_abc(abc)
        if abc_text is not None:
            kwargs["abc"] = abc_text
        try:
            plan = runtime.plan(**kwargs)
        except RuntimeError as exc:
            if "FLASH_ATTENTION" in str(exc) or "flash_attention" in str(exc):
                runtime.backend = "torch-eager"
                plan = runtime.plan(**kwargs)
            else:
                raise
        return (plan.abc or "", _status_text({"abc": plan.truncated}))

class YuE2SaveText:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"forceInput": True}), "filename_prefix": ("STRING", {"default": "yue2/score"}), "extension": (["abc", "txt", "json"],)}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    def save(self, text, filename_prefix, extension):
        path = _unique_file(_output_dir() / filename_prefix, extension)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text if text is not None else "", encoding="utf-8")
        return (str(path),)

class YuE2LoadText:
    @classmethod
    def INPUT_TYPES(cls):
        files = _list_text_files()
        return {"required": {"file": (files or ["(no .abc/.txt files found)"],)}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    def load(self, file):
        if not file or str(file).startswith("(no "):
            return ("",)
        return (Path(_resolve_text_file(file)).read_text(encoding="utf-8"),)

def _unique_dir(base):
    base = Path(str(base))
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return base
    n = 1
    while True:
        candidate = base.parent / f"{base.name}_{n:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        n += 1

def _unique_file(base, ext):
    parent = Path(str(base)).parent
    stem = Path(str(base)).name
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f"{stem}.{ext}"
    if not path.exists():
        return path
    n = 1
    while True:
        candidate = parent / f"{stem}_{n:03d}.{ext}"
        if not candidate.exists():
            return candidate
        n += 1

def _list_text_files():
    names = []
    roots = []
    if folder_paths is not None:
        roots.append(Path(folder_paths.get_input_directory()))
        roots.append(Path(folder_paths.get_output_directory()))
    else:
        roots.append(Path("."))
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("*.abc", "*.txt", "*.json"):
            for p in sorted(root.rglob(pattern)):
                if len(names) >= 200:
                    break
                try:
                    names.append(str(p.relative_to(root)))
                except ValueError:
                    names.append(str(p))
    out = []
    seen = set()
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out

def _resolve_text_file(name):
    if folder_paths is not None:
        for getter in (folder_paths.get_input_directory, folder_paths.get_output_directory):
            candidate = Path(getter()) / name
            if candidate.is_file():
                return candidate
    p = Path(name)
    if p.is_file():
        return p
    raise FileNotFoundError(name)

NODE_CLASS_MAPPINGS = {"YuE2Loader": YuE2Loader, "YuE2Unload": YuE2Unload, "YuE2StylePrompt": YuE2StylePrompt, "YuE2LyricsTemplate": YuE2LyricsTemplate, "YuE2Generate": YuE2Generate, "YuE2Plan": YuE2Plan, "YuE2SaveText": YuE2SaveText, "YuE2LoadText": YuE2LoadText}
NODE_DISPLAY_NAME_MAPPINGS = {"YuE2Loader": "YuE2 Loader", "YuE2Unload": "YuE2 Unload", "YuE2StylePrompt": "YuE2 Style Prompt", "YuE2LyricsTemplate": "YuE2 Lyrics Template", "YuE2Generate": "YuE2 Generate", "YuE2Plan": "YuE2 Plan Score", "YuE2SaveText": "YuE2 Save Text", "YuE2LoadText": "YuE2 Load Text"}

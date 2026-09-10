"""ComfyUI nodes wrapping the official yue2.YuE2Pipeline.

YuE2 is loaded lazily so ComfyUI still starts if the package is not installed.
Install into the *same* Python that runs ComfyUI:

    pip install huggingface-hub
    # then either:
    pip install /path/to/YuE          # official repo: python -m pip install .
    # or the published wheel from m-a-p/YuE2-3B
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import folder_paths
except ImportError:  # running outside ComfyUI
    folder_paths = None


CATEGORY = "audio/YuE2"

DEFAULT_MODEL = "m-a-p/YuE2-3B"
DEFAULT_VAE = "m-a-p/YuE2-Vae"
DEFAULT_VAE_LEGACY = "m-a-p/YuE2-Vae-legacy"

# Process-wide cache. YuE2 is large; keep one live pipeline per key.
_PIPELINES: dict[str, Any] = {}


def _yue2_import():
    try:
        from yue2 import YuE2Pipeline, SymbolicPlan
    except ImportError as exc:
        raise ImportError(
            "YuE2 is not installed in this Python. In the ComfyUI venv run:\n"
            "  git clone https://github.com/multimodal-art-projection/YuE.git\n"
            "  pip install ./YuE\n"
            "or install the wheel from https://huggingface.co/m-a-p/YuE2-3B"
        ) from exc
    return YuE2Pipeline, SymbolicPlan


def _output_dir() -> Path:
    if folder_paths is not None:
        return Path(folder_paths.get_output_directory())
    return Path("output")


def _pipeline_key(model: str, vae: str, device: str, memory_budget_gib: float, backend: str) -> str:
    return f"{model}|{vae}|{device}|{memory_budget_gib:g}|{backend}"


def _get_pipeline(
    model: str,
    vae: str,
    device: str,
    memory_budget_gib: float,
    backend: str,
    progress: bool,
):
    YuE2Pipeline, _ = _yue2_import()
    key = _pipeline_key(model, vae, device, memory_budget_gib, backend)
    pipe = _PIPELINES.get(key)
    if pipe is not None:
        return pipe
    # Drop any previous pipeline so VRAM is released before the next load.
    if _PIPELINES:
        _close_all()
    kwargs = dict(
        device=None if device == "auto" else device,
        memory_budget_gib=float(memory_budget_gib),
        backend=backend,
        progress=bool(progress),
    )
    if kwargs["device"] is None:
        kwargs.pop("device")
    pipe = YuE2Pipeline.from_pretrained(model, vae=vae, **kwargs)
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


def _song_to_audio(audio_np: np.ndarray, sample_rate: int) -> dict:
    """YuE2 SongResult.audio is [samples, channels] float32 in [-1, 1].

    ComfyUI AUDIO is {"waveform": [B, C, T] float32, "sample_rate": int}.
    """
    arr = np.asarray(audio_np)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 8:
        # already [channels, samples]
        channels_first = arr
    else:
        channels_first = arr.T
    waveform = torch.from_numpy(np.ascontiguousarray(channels_first)).float().unsqueeze(0)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def _clean_abc(abc: str | None) -> str | None:
    if abc is None:
        return None
    text = str(abc).strip()
    return text if text else None


def _cfg_or_none(cfg_scale: float) -> float | None:
    # 0 means "use YuE2 defaults" (1.0 for full/melody, 1.01 for off).
    if cfg_scale is None or float(cfg_scale) <= 0:
        return None
    return float(cfg_scale)


def _status_text(truncated: dict | bool | None, extra: str = "") -> str:
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


LANGUAGES = [
    "English",
    "Mandarin",
    "Cantonese",
    "Japanese",
    "Korean",
    "Spanish",
    "French",
    "Portuguese",
    "German",
    "Italian",
    "Hindi",
    "(custom / none)",
]

GENRES = [
    "pop",
    "piano pop",
    "indie pop",
    "synth pop",
    "dance pop",
    "r&b",
    "soul",
    "funk",
    "nu-disco",
    "disco",
    "house",
    "edm",
    "hip hop",
    "trap",
    "rap",
    "rock",
    "indie rock",
    "alternative rock",
    "punk",
    "metal",
    "jazz",
    "jazz-funk",
    "bossa nova",
    "latin",
    "reggaeton",
    "country",
    "folk",
    "indie folk",
    "singer-songwriter",
    "acoustic",
    "gospel",
    "blues",
    "lo-fi",
    "dream pop",
    "shoegaze",
    "ambient",
    "cinematic",
    "film score",
    "classical crossover",
    "k-pop",
    "j-pop",
    "city pop",
    "c-pop",
    "afrobeats",
    "reggae",
    "ska",
    "(custom / none)",
]

MOODS = [
    "(none)",
    "uplifting",
    "euphoric",
    "romantic",
    "intimate",
    "melancholic",
    "bittersweet",
    "dark",
    "aggressive",
    "playful",
    "nostalgic",
    "dreamy",
    "cinematic",
    "chill",
    "anthemic",
    "mysterious",
]

VOCALS = [
    "(instrumental / no lead vocal)",
    "female vocal",
    "male vocal",
    "androgynous vocal",
    "duet male and female",
    "choir / stacked vocals",
    "rap lead",
    "spoken word",
]

VOCAL_COLORS = [
    "(none)",
    "warm",
    "bright",
    "breathy",
    "powerful",
    "raspy",
    "soft intimate",
    "belting",
    "tenor",
    "baritone",
    "soprano",
    "alto",
]

ENERGIES = ["(none)", "low energy", "mid energy", "high energy", "builds from quiet to loud"]

LENGTH_HINTS = {
    "model decides": "",
    "short clip (~45-75s)": "short song, about one minute",
    "single edit (~2-3 min)": "radio-length song, about two to three minutes",
    "full song (~3.5-5 min)": "full-length song, about four minutes",
    "very long (may truncate)": "long-form song with extended sections",
}

LYRIC_TEMPLATES = {
    "blank": "",
    "verse + chorus": (
        "[Verse]\n"
        "Line one of the verse\n"
        "Line two of the verse\n"
        "Line three of the verse\n"
        "Line four of the verse\n"
        "\n"
        "[Chorus]\n"
        "Sing the hook here\n"
        "Repeat the title line\n"
        "Hold a little room for light\n"
        "We will sing beyond the night"
    ),
    "V-C-V-C": (
        "[Verse]\n"
        "First verse line\n"
        "Keep lines short\n"
        "\n"
        "[Chorus]\n"
        "Main hook\n"
        "Title line\n"
        "\n"
        "[Verse]\n"
        "Second verse line\n"
        "New images, same rhythm\n"
        "\n"
        "[Chorus]\n"
        "Main hook\n"
        "Title line"
    ),
    "V-C-V-C-bridge-C": (
        "[Verse]\n"
        "First verse\n"
        "\n"
        "[Chorus]\n"
        "Hook\n"
        "\n"
        "[Verse]\n"
        "Second verse\n"
        "\n"
        "[Chorus]\n"
        "Hook\n"
        "\n"
        "[Bridge]\n"
        "Contrast the chorus here\n"
        "\n"
        "[Chorus]\n"
        "Final hook"
    ),
    "intro-V-pre-C-V-C-bridge-C-outro": (
        "[Intro]\n"
        "(instrumental atmosphere)\n"
        "\n"
        "[Verse]\n"
        "Verse one\n"
        "\n"
        "[Prechorus]\n"
        "Lift into the hook\n"
        "\n"
        "[Chorus]\n"
        "Hook\n"
        "\n"
        "[Verse]\n"
        "Verse two\n"
        "\n"
        "[Chorus]\n"
        "Hook\n"
        "\n"
        "[Bridge]\n"
        "Bridge\n"
        "\n"
        "[Chorus]\n"
        "Last chorus\n"
        "\n"
        "[Outro]\n"
        "(fade or last line)"
    ),
    "rap song": (
        "[Verse]\n"
        "Sixteen bars of verse start here\n"
        "\n"
        "[Chorus]\n"
        "Sung hook\n"
        "\n"
        "[Verse]\n"
        "Second sixteen\n"
        "\n"
        "[Chorus]\n"
        "Sung hook"
    ),
}


def _omit(value: str) -> bool:
    return not value or value.startswith("(")


def build_style_prompt(
    language: str,
    genre: str,
    mood: str,
    vocal: str,
    vocal_color: str,
    instruments: str,
    tempo_bpm: int,
    energy: str,
    length_hint: str,
    extra: str,
    custom_language: str = "",
    custom_genre: str = "",
) -> str:
    parts: list[str] = []
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
    """Load (and cache) YuE2-3B + VAE. Connect this to generate / plan nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": DEFAULT_MODEL, "tooltip": "HF repo or local folder"}),
                "vae": (
                    "STRING",
                    {
                        "default": DEFAULT_VAE,
                        "tooltip": "m-a-p/YuE2-Vae (listening) or m-a-p/YuE2-Vae-legacy (paper)",
                    },
                ),
                "device": (["auto", "cuda", "cuda:0", "cpu"],),
                "memory_budget_gib": (
                    "FLOAT",
                    {"default": 24.0, "min": 8.0, "max": 80.0, "step": 1.0},
                ),
                "backend": (["torch", "torch-eager"],),
                "progress": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("YUE2_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, model, vae, device, memory_budget_gib, backend, progress):
        pipe = _get_pipeline(
            model.strip(),
            vae.strip() or DEFAULT_VAE,
            device,
            memory_budget_gib,
            backend,
            progress,
        )
        handle = {
            "key": _pipeline_key(model.strip(), vae.strip() or DEFAULT_VAE, device, memory_budget_gib, backend),
            "model": model.strip(),
            "vae": vae.strip() or DEFAULT_VAE,
            "device": device,
            "memory_budget_gib": float(memory_budget_gib),
            "backend": backend,
            "progress": bool(progress),
        }
        # Keep a live reference so GC does not close the cached object mid-graph.
        handle["_pipe"] = pipe
        return (handle,)


class YuE2Unload:
    """Free the cached YuE2 pipeline and CUDA memory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("YUE2_PIPE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def unload(self, pipe):
        _close_all()
        return ("unloaded",)


class YuE2StylePrompt:
    """Build the YuE2 `style` string from structured fields.

    YuE2 has no genre enum or duration slider. Style is one free-text line:
    language, genre, mood, vocal, instruments, BPM, energy, optional length hint.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "language": (LANGUAGES, {"default": "English"}),
                "genre": (GENRES, {"default": "piano pop"}),
                "mood": (MOODS, {"default": "intimate"}),
                "vocal": (VOCALS, {"default": "female vocal"}),
                "vocal_color": (VOCAL_COLORS, {"default": "warm"}),
                "instruments": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "acoustic piano, rounded bass and light drums",
                        "tooltip": "List the band. YuE2 follows this more reliably than a hidden instrument mixer.",
                    },
                ),
                "tempo_bpm": (
                    "INT",
                    {
                        "default": 88,
                        "min": 0,
                        "max": 220,
                        "tooltip": "0 = do not mention tempo. BPM is a text hint, not a metronome lock.",
                    },
                ),
                "energy": (ENERGIES, {"default": "mid energy"}),
                "length_hint": (
                    list(LENGTH_HINTS.keys()),
                    {
                        "default": "single edit (~2-3 min)",
                        "tooltip": "Written into the style line only. YuE2 has no duration API — lyrics + sections + BPM decide length.",
                    },
                ),
                "extra": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "lyrical memorable melody, unhurried phrasing",
                    },
                ),
            },
            "optional": {
                "custom_language": ("STRING", {"default": ""}),
                "custom_genre": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("style",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(
        self,
        language,
        genre,
        mood,
        vocal,
        vocal_color,
        instruments,
        tempo_bpm,
        energy,
        length_hint,
        extra,
        custom_language="",
        custom_genre="",
    ):
        style = build_style_prompt(
            language,
            genre,
            mood,
            vocal,
            vocal_color,
            instruments,
            tempo_bpm,
            energy,
            length_hint,
            extra,
            custom_language,
            custom_genre,
        )
        return (style,)


class YuE2LyricsTemplate:
    """Drop in a section skeleton. Length tracks how many sections and how many lines you write.

    Rough YuE2 / YuE practice: one labeled section is often around 20-40 seconds.
    More sections and more lines → longer song. There is no bars/seconds slider.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": (list(LYRIC_TEMPLATES.keys()), {"default": "V-C-V-C-bridge-C"}),
                "lyrics_override": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "If not empty, this text is used instead of the template.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, template, lyrics_override):
        text = (lyrics_override or "").strip()
        if text:
            return (lyrics_override,)
        return (LYRIC_TEMPLATES.get(template, ""),)


class YuE2Generate:
    """One-shot lyrics + style → 48 kHz stereo song.

    Style controls genre / vocal / instruments / BPM (text).
    Length is NOT a seconds slider — use more/fewer lyric sections, BPM, and
    semantic_max_tokens as a hard cap. If status says truncated=semantic,
    raise the cap or shorten lyrics.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("YUE2_PIPE",),
                "style": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "English, warm piano pop, expressive female voice, "
                            "acoustic piano, rounded bass and light drums, "
                            "lyrical memorable melody, unhurried phrasing, 88 BPM, "
                            "radio-length song, about two to three minutes"
                        ),
                        "tooltip": (
                            "Official YuE2 style line: language, genre, instruments, "
                            "vocal character, tempo. Connect YuE2 Style Prompt or type it."
                        ),
                    },
                ),
                "lyrics": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": LYRIC_TEMPLATES["verse + chorus"],
                        "tooltip": (
                            "Use [Verse] [Chorus] [Bridge] [Intro] [Outro] [Prechorus]. "
                            "Blank line between sections. More sections ≈ longer song."
                        ),
                    },
                ),
                "cot": (
                    ["full", "melody", "off"],
                    {
                        "default": "full",
                        "tooltip": "full = melody+chords, melody = covers, off = no score.",
                    },
                ),
                "seed": ("INT", {"default": 831001, "min": 0, "max": 0x7FFFFFFFFFFFFFFF, "control_after_generate": True}),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.01,
                        "tooltip": "0 = YuE2 default (1.0 full/melody, 1.01 off). 1.2 = stronger text guidance.",
                    },
                ),
                "override_sampling": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Off = official sampling defaults. On = use the sliders below.",
                    },
                ),
                "semantic_max_tokens": (
                    "INT",
                    {
                        "default": 9000,
                        "min": 200,
                        "max": 16000,
                        "step": 100,
                        "tooltip": "Hard cap on the audio token stream (default 9000). Higher can mean longer; too low truncates.",
                    },
                ),
                "semantic_min_tokens": (
                    "INT",
                    {
                        "default": 200,
                        "min": 0,
                        "max": 8000,
                        "step": 50,
                        "tooltip": "Minimum semantic tokens before the model may stop.",
                    },
                ),
                "abc_max_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 32,
                        "max": 8192,
                        "step": 32,
                        "tooltip": "Cap on the planned ABC score. Rarely needs changing.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.05,
                        "tooltip": "Semantic sampling temperature (default 1.0). Higher = wilder.",
                    },
                ),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.05, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 100, "min": 1, "max": 500}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 2.0, "step": 0.01}),
                "save_artifacts": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "yue2/song"}),
            },
            "optional": {
                "abc": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Optional ABC score. Requires cot=full or melody.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "abc", "style_used", "status")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(
        self,
        pipe,
        style,
        lyrics,
        cot,
        seed,
        cfg_scale,
        override_sampling,
        semantic_max_tokens,
        semantic_min_tokens,
        abc_max_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        save_artifacts,
        filename_prefix,
        abc="",
    ):
        runtime = pipe.get("_pipe") if isinstance(pipe, dict) else None
        if runtime is None:
            raise RuntimeError("Connect YuE2 Loader first.")

        abc_text = _clean_abc(abc)
        if abc_text is not None and cot == "off":
            raise ValueError("A supplied ABC score requires cot=full or cot=melody.")

        kwargs = dict(
            style=style,
            lyrics=lyrics,
            cot=cot,
            seed=int(seed),
        )
        cfg = _cfg_or_none(cfg_scale)
        if cfg is not None:
            kwargs["cfg_scale"] = cfg
        if abc_text is not None:
            kwargs["abc"] = abc_text

        if override_sampling:
            kwargs["semantic_sampling"] = {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "repetition_penalty": float(repetition_penalty),
                "min_tokens": int(semantic_min_tokens),
                "max_tokens": int(semantic_max_tokens),
            }
            kwargs["abc_sampling"] = {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 30,
                "repetition_penalty": 1.005,
                "penalty_window": 100,
                "min_tokens": 32,
                "max_tokens": int(abc_max_tokens),
            }

        try:
            runtime.backend = "torch-eager"
            print("[YuE2] forced backend", runtime.backend)
            song = runtime(**kwargs)
        except Exception:
            traceback.print_exc()
            raise

        audio = _song_to_audio(song.audio, song.sample_rate)
        score = song.abc or ""
        extra = "no duration API; length ≈ lyrics/sections + BPM + token cap"
        if save_artifacts:
            out = _unique_dir(_output_dir() / filename_prefix)
            result = song.save_artifacts(out)
            extra = f"saved={out} seconds={result.get('audio_seconds', '?')} | {extra}"
        status = _status_text(song.truncated, extra)
        return (audio, score, style, status)


class YuE2Plan:
    """Write an editable ABC score without rendering audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("YUE2_PIPE",),
                "style": ("STRING", {"multiline": True, "default": "English, jazz-funk, warm lead vocal"}),
                "lyrics": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[Verse]\nWrite lyrics here\n\n[Chorus]\nWrite the hook here",
                    },
                ),
                "cot": (["full", "melody"], {"default": "full"}),
                "seed": ("INT", {"default": 831001, "min": 0, "max": 0x7FFFFFFFFFFFFFFF, "control_after_generate": True}),
            },
            "optional": {
                "abc": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "If set, this score is used as-is (no new planning).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("abc", "status")
    FUNCTION = "plan"
    CATEGORY = CATEGORY

    def plan(self, pipe, style, lyrics, cot, seed, abc=""):
        runtime = pipe.get("_pipe") if isinstance(pipe, dict) else None
        if runtime is None:
            raise RuntimeError("Connect YuE2 Loader first.")
        kwargs = dict(style=style, lyrics=lyrics, cot=cot, seed=int(seed))
        abc_text = _clean_abc(abc)
        if abc_text is not None:
            kwargs["abc"] = abc_text
        plan = runtime.plan(**kwargs)
        status = _status_text({"abc": plan.truncated})
        return (plan.abc or "", status)


class YuE2SaveText:
    """Write ABC / lyrics / style to ComfyUI's output folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "yue2/score"}),
                "extension": (["abc", "txt", "json"],),
            }
        }

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
    """Load an ABC or lyrics file from ComfyUI input/output folders."""

    @classmethod
    def INPUT_TYPES(cls):
        files = _list_text_files()
        return {
            "required": {
                "file": (files or ["(no .abc/.txt files found)"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, file):
        path = _resolve_text_file(file)
        return (Path(path).read_text(encoding="utf-8"),)


def _unique_dir(base: Path) -> Path:
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


def _unique_file(base: Path, ext: str) -> Path:
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


def _list_text_files() -> list[str]:
    names: list[str] = []
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
                try:
                    names.append(str(p.relative_to(root)))
                except ValueError:
                    names.append(str(p))
    # de-dupe, keep order
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _resolve_text_file(name: str) -> Path:
    if folder_paths is not None:
        for getter in (folder_paths.get_input_directory, folder_paths.get_output_directory):
            candidate = Path(getter()) / name
            if candidate.is_file():
                return candidate
    p = Path(name)
    if p.is_file():
        return p
    raise FileNotFoundError(name)


NODE_CLASS_MAPPINGS = {
    "YuE2Loader": YuE2Loader,
    "YuE2Unload": YuE2Unload,
    "YuE2StylePrompt": YuE2StylePrompt,
    "YuE2LyricsTemplate": YuE2LyricsTemplate,
    "YuE2Generate": YuE2Generate,
    "YuE2Plan": YuE2Plan,
    "YuE2SaveText": YuE2SaveText,
    "YuE2LoadText": YuE2LoadText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuE2Loader": "YuE2 Loader",
    "YuE2Unload": "YuE2 Unload",
    "YuE2StylePrompt": "YuE2 Style Prompt",
    "YuE2LyricsTemplate": "YuE2 Lyrics Template",
    "YuE2Generate": "YuE2 Generate",
    "YuE2Plan": "YuE2 Plan Score",
    "YuE2SaveText": "YuE2 Save Text",
    "YuE2LoadText": "YuE2 Load Text",
}

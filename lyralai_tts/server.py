from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import logging
import pathlib
import re
import struct
import threading
import time
from typing import Any

import numpy as np
import soundfile as sf
import soxr
import torch
from aiohttp import web

from lyralai_tts.streaming import (
    Cancelled,
    StreamConfig,
    StreamingSynth,
    VoiceParams,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qwen-shim")

TARGET_SR = 44100
NODE_STRIPS_HEADER_BYTES = 44
UNKNOWN_STREAM_LENGTH = 0xFFFFFFFF
FADE_OUT_MS = 12
WARMUP_RUNS = 4
WARMUP_PHRASES = (
    "Тест один два три четыре пять.",
    "Привет, как дела? Это второй прогрев системы.",
    "Третий тестовый запуск для полного прогрева всех компонентов модели.",
)

LANG_BY_CODE = {
    "ru": "russian",
    "en": "english",
    "ja": "japanese",
    "ko": "korean",
    "zh": "chinese",
}

SCRIPT_PATTERNS = (
    (r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]", "korean"),
    (r"[\u3040-\u309F\u30A0-\u30FF]", "japanese"),
    (r"[\u4E00-\u9FAF]", "chinese"),
    (r"[а-яА-ЯёЁ]", "russian"),
    (r"[a-zA-Z]{3,}", "english"),
)


LIMITS = {
    "temperature": (0.05, 1.5),
    "top_p": (0.05, 1.0),
    "top_k": (0, 200),
    "repetition_penalty": (1.0, 2.0),
}


def clamped(body: dict[str, Any], name: str) -> float | None:
    raw = body.get(name)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    low, high = LIMITS[name]
    return min(max(value, low), high)


def voice_params(body: dict[str, Any]) -> VoiceParams:
    top_k = clamped(body, "top_k")
    seed = body.get("seed")
    return VoiceParams(
        temperature=clamped(body, "temperature"),
        top_p=clamped(body, "top_p"),
        top_k=None if top_k is None else int(top_k),
        repetition_penalty=clamped(body, "repetition_penalty"),
        seed=int(seed) if isinstance(seed, (int, float)) else None,
    )


def detect_language(text: str) -> str:
    for pattern, language in SCRIPT_PATTERNS:
        if re.search(pattern, text):
            return language
    return "russian"


def streaming_wav_header(sample_rate: int = TARGET_SR) -> bytes:
    return (
        b"RIFF"
        + struct.pack("<I", UNKNOWN_STREAM_LENGTH)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", UNKNOWN_STREAM_LENGTH)
    )


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def fade_out(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    length = min(len(audio), int(sample_rate * FADE_OUT_MS / 1000))
    if length <= 0:
        return audio
    faded = audio.copy()
    faded[-length:] *= np.linspace(1.0, 0.0, length, dtype=np.float32)
    return faded


class VoiceEngine:
    def __init__(self, model_path: str, device: str, config: StreamConfig) -> None:
        from qwen_tts import Qwen3TTSModel

        log.info(f"Загружаю модель из {model_path}")
        self.model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.synth = StreamingSynth(self.model, config)
        self.prompts: dict[str, Any] = {}
        self.gpu_lock = threading.RLock()
        self.active_cancel: threading.Event | None = None
        self.device = device
        self.on_gpu = True
        self.warmed_up = False

    def prompt_for(self, reference_b64: str, reference_text: str) -> Any:
        key = hashlib.sha1(
            (reference_b64[:2048] + "|" + reference_text).encode()
        ).hexdigest()
        cached = self.prompts.get(key)
        if cached is not None:
            return cached

        audio, sample_rate = sf.read(
            io.BytesIO(base64.b64decode(reference_b64)), dtype="float32"
        )
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        with self.gpu_lock:
            prompt = self.model.create_voice_clone_prompt(
                ref_audio=(audio, sample_rate),
                ref_text=reference_text,
                x_vector_only_mode=False,
            )
        self.prompts[key] = prompt
        log.info(f"Промпт клонирования собран, в кэше {len(self.prompts)}")
        return prompt

    def release_vram(self) -> None:
        if not self.on_gpu:
            return
        self.interrupt_previous()
        self.model.model.to("cpu")
        torch.cuda.empty_cache()
        self.on_gpu = False
        self.warmed_up = False
        log.info("Веса выгружены на CPU, карта свободна")

    def claim_vram(self) -> None:
        if self.on_gpu:
            return
        self.model.model.to(self.device)
        self.synth.token_bias = self.synth._service_token_bias()
        self.on_gpu = True
        log.info("Веса вернулись на карту")

    def warm_up_from_file(self, reference_path: str, reference_text: str) -> None:
        path = pathlib.Path(reference_path)
        if not path.exists():
            log.warning(f"Прогрев пропущен: {path} не найден")
            return
        text = reference_text.strip()
        if not text:
            lab = path.with_suffix(".lab")
            text = lab.read_text(encoding="utf-8").strip() if lab.exists() else ""
        started = time.perf_counter()
        prompt = self.prompt_for(
            base64.b64encode(path.read_bytes()).decode(), text
        )
        for index in range(WARMUP_RUNS):
            for _chunk in self.synth.stream(WARMUP_PHRASES[index % len(WARMUP_PHRASES)], prompt):
                pass
        torch.cuda.synchronize()
        self.warmed_up = True
        log.info(f"Прогрет за {time.perf_counter() - started:.1f}с")

    def interrupt_previous(self) -> None:
        if self.active_cancel is not None:
            self.active_cancel.set()

    def take_turn(self) -> threading.Event:
        self.interrupt_previous()
        cancel = threading.Event()
        self.active_cancel = cancel
        return cancel

    def release_turn(self, cancel: threading.Event) -> None:
        if self.active_cancel is cancel:
            self.active_cancel = None


class ResampledStream:
    def __init__(self, source_sr: int, target_sr: int = TARGET_SR) -> None:
        self.passthrough = source_sr == target_sr
        self.resampler = (
            None
            if self.passthrough
            else soxr.ResampleStream(source_sr, target_sr, 1, dtype="float32")
        )

    def feed(self, audio: np.ndarray, last: bool = False) -> np.ndarray:
        if self.passthrough:
            return audio
        return self.resampler.resample_chunk(audio, last=last)


async def handle_tts(request: web.Request) -> web.StreamResponse:
    engine: VoiceEngine = request.app["engine"]

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "bad json"}, status=400)

    if not engine.on_gpu:
        return web.json_response({"error": "unloaded"}, status=503)

    text = (body.get("text") or "").strip()
    if not text:
        return web.json_response({"error": "empty text"}, status=400)

    references = body.get("references") or []
    if not references or not references[0].get("audio"):
        return web.json_response({"error": "no reference"}, status=400)

    reference_b64 = references[0]["audio"]
    reference_text = (references[0].get("text") or "").strip()
    language = LANG_BY_CODE.get(body.get("language", ""), detect_language(text))
    params = voice_params(body)

    cancel = engine.take_turn()
    prompt = await asyncio.to_thread(engine.prompt_for, reference_b64, reference_text)

    response = web.StreamResponse(
        status=200,
        headers={"Content-Type": "audio/wav", "Cache-Control": "no-store"},
    )
    await response.prepare(request)
    await response.write(streaming_wav_header())

    queue: asyncio.Queue[tuple[np.ndarray, int] | None] = asyncio.Queue(maxsize=8)
    loop = asyncio.get_running_loop()

    def produce() -> None:
        try:
            with engine.gpu_lock:
                torch.compiler.cudagraph_mark_step_begin()
                for chunk, sample_rate in engine.synth.stream(
                    text, prompt, language=language, cancel=cancel, params=params
                ):
                    asyncio.run_coroutine_threadsafe(
                        queue.put((chunk, sample_rate)), loop
                    ).result()
        except Cancelled:
            log.info(f"Прервано: {text[:40]}")
        except Exception as error:
            log.error(f"Синтез упал: {error}")
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    worker = asyncio.create_task(asyncio.to_thread(produce))

    resampler: ResampledStream | None = None
    started_at = loop.time()
    first_chunk_at: float | None = None
    total_samples = 0

    try:
        while True:
            item = await queue.get()
            if item is None:
                break

            chunk, source_sr = item
            if resampler is None:
                resampler = ResampledStream(source_sr)
            if first_chunk_at is None:
                first_chunk_at = loop.time() - started_at

            resampled = resampler.feed(chunk)
            if len(resampled):
                total_samples += len(resampled)
                await response.write(to_pcm16(resampled))

        if resampler is not None:
            tail = resampler.feed(np.zeros(0, dtype=np.float32), last=True)
            if len(tail):
                total_samples += len(tail)
                await response.write(to_pcm16(fade_out(tail, TARGET_SR)))
    finally:
        engine.release_turn(cancel)
        await worker

    await response.write_eof()
    log.info(
        f"TTFB {(first_chunk_at or 0) * 1000:.0f} мс | "
        f"{total_samples / TARGET_SR:.2f}с звука | {language} | "
        f"t={params.temperature} rp={params.repetition_penalty} | {text[:50]}"
    )
    return response


async def handle_interrupt(request: web.Request) -> web.Response:
    engine: VoiceEngine = request.app["engine"]
    engine.interrupt_previous()
    return web.json_response({"ok": True})


async def handle_unload(request: web.Request) -> web.Response:
    engine: VoiceEngine = request.app["engine"]
    await asyncio.to_thread(engine.release_vram)
    return web.json_response({"on_gpu": engine.on_gpu})


async def handle_load(request: web.Request) -> web.Response:
    engine: VoiceEngine = request.app["engine"]
    await asyncio.to_thread(engine.claim_vram)
    return web.json_response({"on_gpu": engine.on_gpu})


async def handle_health(request: web.Request) -> web.Response:
    engine: VoiceEngine = request.app["engine"]
    return web.json_response(
        {
            "ok": True,
            "on_gpu": engine.on_gpu,
            "prompts_cached": len(engine.prompts),
            "speaking": engine.active_cancel is not None,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--model", default="models/qwen3-tts-1.7b-base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--emit", type=int, default=4)
    parser.add_argument("--window", type=int, default=80)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup-ref", default="")
    parser.add_argument("--warmup-text", default="")
    args = parser.parse_args()

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["engine"] = VoiceEngine(
        args.model,
        args.device,
        StreamConfig(
            emit_every_frames=args.emit,
            decode_window_frames=args.window,
            compile_modules=args.compile,
        ),
    )

    if args.warmup_ref:
        app["engine"].warm_up_from_file(args.warmup_ref, args.warmup_text)
    app.router.add_post("/v1/tts", handle_tts)
    app.router.add_post("/v1/interrupt", handle_interrupt)
    app.router.add_post("/v1/unload", handle_unload)
    app.router.add_post("/v1/load", handle_load)
    app.router.add_get("/health", handle_health)

    log.info(f"Слушаю http://{args.host}:{args.port}/v1/tts")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()

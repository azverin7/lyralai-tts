from __future__ import annotations

import argparse
import logging
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from lyralai_tts.streaming import StreamConfig, StreamingSynth

SEAM_GAP_SECONDS = 0.12

WARMUP_PHRASES = (
    "Тест один два три четыре пять.",
    "Привет, как дела? Это второй прогрев системы.",
    "Третий тестовый запуск для полного прогрева всех компонентов модели.",
)

PHRASES = (
    "Ну что, дружище, видишь?",
    "Слушай, ты правда думаешь, что мне нужно это ведро с болтами и сервоприводами?",
    "Опять этот вакуум, где единственным событием становится звук вентиляторов.",
)


def vram_used_gib() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**30


def load_mono(path: str) -> tuple[np.ndarray, int]:
    wav, sample_rate = sf.read(path, dtype="float32")
    return (wav.mean(axis=1) if wav.ndim > 1 else wav), sample_rate


def write_phrase_audio(
    output_dir: Path,
    phrase_index: int,
    emit_every_frames: int,
    chunks: list[np.ndarray],
    sample_rate: int,
) -> None:
    stem = f"emit{emit_every_frames}_{phrase_index:02d}"
    sf.write(output_dir / f"{stem}.wav", np.concatenate(chunks), sample_rate)

    gap = np.zeros(int(sample_rate * SEAM_GAP_SECONDS), dtype=np.float32)
    with_gaps: list[np.ndarray] = []
    for chunk in chunks:
        with_gaps.append(chunk)
        with_gaps.append(gap)
    sf.write(output_dir / f"{stem}_seams.wav", np.concatenate(with_gaps[:-1]), sample_rate)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen3-tts-1.7b-base")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--emit", type=int, default=4)
    parser.add_argument("--window", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--static-cache", action="store_true")
    parser.add_argument("--mode", default="")
    parser.add_argument("--compile-talker", action="store_true")
    parser.add_argument("--no-codebook", action="store_true")
    parser.add_argument("--no-decoder", action="store_true")
    parser.add_argument("--no-fast-codebook", action="store_true")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    output_dir = Path(args.out_dir) if args.out_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"VRAM до загрузки:     {vram_used_gib():.2f} GiB")

    started = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    print(f"Загрузка модели:      {time.perf_counter() - started:.1f}с")
    print(f"VRAM после загрузки:  {vram_used_gib():.2f} GiB")

    reference, reference_sr = load_mono(args.ref)
    started = time.perf_counter()
    clone_prompt = model.create_voice_clone_prompt(
        ref_audio=(reference, reference_sr),
        ref_text=args.ref_text,
        x_vector_only_mode=False,
    )
    print(f"Промпт клонирования:  {time.perf_counter() - started:.1f}с")

    synth = StreamingSynth(
        model,
        StreamConfig(
            emit_every_frames=args.emit,
            decode_window_frames=args.window,
            compile_modules=args.compile,
            autotune=args.autotune,
            static_kv_cache=args.static_cache,
            compile_mode=args.mode,
            compile_talker=args.compile_talker,
            compile_codebook=not args.no_codebook,
            compile_decoder=not args.no_decoder,
            fast_codebook=not args.no_fast_codebook,
        ),
    )

    warmup_runs = max(args.warmup, 4 if args.compile else args.warmup)
    print(f"\n--- прогрев x{warmup_runs} ---")
    warmup_started = time.perf_counter()
    for index in range(warmup_runs):
        phrase = WARMUP_PHRASES[index % len(WARMUP_PHRASES)]
        for _chunk in synth.stream(phrase, clone_prompt):
            pass
    torch.cuda.synchronize()
    print(f"прогрев занял {time.perf_counter() - warmup_started:.1f}с")

    print("\n--- замер ---")
    time_to_first_chunk: list[float] = []
    realtime_factors: list[float] = []

    for phrase_index, phrase in enumerate(PHRASES, start=1):
        torch.cuda.synchronize()
        started = time.perf_counter()
        first_chunk_at: float | None = None
        chunks: list[np.ndarray] = []
        sample_rate = 0

        for chunk, chunk_sr in synth.stream(phrase, clone_prompt):
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter() - started
            chunks.append(chunk)
            sample_rate = chunk_sr

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        duration = len(audio) / max(sample_rate, 1)
        realtime_factor = elapsed / max(duration, 1e-6)

        time_to_first_chunk.append(first_chunk_at or 0.0)
        realtime_factors.append(realtime_factor)
        print(
            f"TTFB {(first_chunk_at or 0) * 1000:6.0f} мс | "
            f"всего {elapsed:5.2f}с → {duration:5.2f}с звука | "
            f"RTF {realtime_factor:.3f} | чанков {len(chunks):3d} | {phrase[:38]}"
        )

        if output_dir is not None:
            write_phrase_audio(output_dir, phrase_index, args.emit, chunks, sample_rate)

    print(f"\nмедианный TTFB: {statistics.median(time_to_first_chunk) * 1000:.0f} мс")
    print(f"медианный RTF:  {statistics.median(realtime_factors):.3f}")
    print(f"VRAM пик:       {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()

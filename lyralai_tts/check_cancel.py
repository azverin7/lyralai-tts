from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from lyralai_tts.streaming import Cancelled, StreamConfig, StreamingSynth

LONG_PHRASE = (
    "Слушай, ты правда думаешь, что твоя очередная попытка выдать желаемое "
    "за действительное спасёт этот вечер, или ты просто переоцениваешь своё "
    "влияние на мой дофаминовый фон, пока я тут перебираю логи?"
)


def load_mono(path: str) -> tuple[np.ndarray, int]:
    wav, sample_rate = sf.read(path, dtype="float32")
    return (wav.mean(axis=1) if wav.ndim > 1 else wav), sample_rate


def collect(
    synth: StreamingSynth, phrase: str, cancel_after_seconds: float | None
) -> tuple[list[np.ndarray], int, float, bool]:
    cancel = threading.Event()
    if cancel_after_seconds is not None:
        threading.Timer(cancel_after_seconds, cancel.set).start()

    chunks: list[np.ndarray] = []
    sample_rate = 0
    was_cancelled = False
    started = time.perf_counter()

    try:
        for chunk, chunk_sr in synth.stream(phrase, synth.prompt, cancel=cancel):
            chunks.append(chunk)
            sample_rate = chunk_sr
    except Cancelled:
        was_cancelled = True

    return chunks, sample_rate, time.perf_counter() - started, was_cancelled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen3-tts-1.7b-base")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--cancel-at", type=float, default=1.0)
    parser.add_argument("--out", default="/tmp/tts_check/cancelled.wav")
    args = parser.parse_args()

    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    reference, reference_sr = load_mono(args.ref)
    synth = StreamingSynth(model, StreamConfig(emit_every_frames=4))
    synth.prompt = model.create_voice_clone_prompt(
        ref_audio=(reference, reference_sr),
        ref_text=args.ref_text,
        x_vector_only_mode=False,
    )

    for _ in range(2):
        for _chunk in synth.stream("Прогрев.", synth.prompt):
            pass
    torch.cuda.synchronize()

    print("--- полная генерация ---")
    chunks, sample_rate, elapsed, _ = collect(synth, LONG_PHRASE, None)
    full_duration = sum(len(c) for c in chunks) / max(sample_rate, 1)
    print(f"{elapsed:.2f}с → {full_duration:.2f}с звука, чанков {len(chunks)}")

    torch.cuda.synchronize()
    memory_before = torch.cuda.memory_allocated() / 2**30

    print(f"\n--- отмена через {args.cancel_at}с ---")
    chunks, sample_rate, elapsed, was_cancelled = collect(
        synth, LONG_PHRASE, args.cancel_at
    )
    cut_duration = sum(len(c) for c in chunks) / max(sample_rate, 1)

    torch.cuda.synchronize()
    memory_after = torch.cuda.memory_allocated() / 2**30

    print(f"исключение Cancelled: {was_cancelled}")
    print(f"остановилась за:      {elapsed:.2f}с (просили {args.cancel_at}с)")
    print(f"успела наговорить:    {cut_duration:.2f}с из {full_duration:.2f}с")
    print(f"чанков:               {len(chunks)}")
    print(f"память до/после:      {memory_before:.2f} / {memory_after:.2f} GiB")

    if chunks:
        sf.write(args.out, np.concatenate(chunks), sample_rate)
        print(f"\nоборванная фраза:     {args.out}")


if __name__ == "__main__":
    main()
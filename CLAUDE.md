# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# lyralai-tts

A streaming layer on top of the `qwen-tts` package. The upstream package is
never patched — everything is done from the outside, by swapping classes on live
objects and intercepting `forward`.

The goal is to emit the first audio as early as possible and keep the stream
seamless, not to make synthesis faster overall.

## Commands

This project needs its own venv: the package pins `transformers==4.57.3` and
must not share an environment with newer branches. Run everything from the
repository root.

Server:
```
.venv/bin/python -u -m lyralai_tts.server --port 8083 --emit 4 --compile \
  --warmup-ref path/to/reference.wav
```

The TTFB/RTF benchmark is the primary tool. There is no conventional test
suite — the measurement *is* the test:
```
.venv/bin/python -m lyralai_tts.bench --ref path/to/reference.wav \
  --ref-text "$(cat path/to/reference.lab)" --compile
```
The benchmark defaults match the server configuration. Ablation flags:
`--compile-talker`, `--static-cache`, `--autotune`, `--mode MODE`,
`--no-codebook`, `--no-decoder`, `--no-fast-codebook`, `--emit N`,
`--window N`, `--warmup N`.
`--out-dir DIR` writes a wav per phrase plus `_seams.wav`
— the same chunks with a 120 ms gap at every boundary, which makes decoder
window seams audible.

`--compile-talker` is currently not merely useless but fatal: inductor traces
`talker.forward` all the way down to the predictor's `rotary_emb`, while the
loop runs under `@torch.inference_mode()` → `RuntimeError: Cannot set
version_counter for inference tensor`.

Cancellation check — memory leaks and how fast a cancel is honoured:
```
.venv/bin/python -m lyralai_tts.check_cancel --ref ... --ref-text ... --cancel-at 1.0
```

## Working configuration

Measured on an RTX 5070 Ti: TTFB 92 ms, RTF 0.235, 4.49 GB VRAM, 16 s warmup.

- bf16 + sdpa (not flash_attention_2 — slower here)
- compile_modules=True, compile_talker=True, fast_codebook=True
- static_kv_cache=False, emit_every_frames=4, decode_window_frames=80

Three settings in `acceleration.py` that everything depends on:

- `compile_talker` compiles the **inner** `talker.model.forward`. The outer
  `talker.forward` contains the codebook loop and sampling — dynamo tears it
  apart and rebuilds the graph on every step, 203 ms per call instead of 15.
- `cudagraph_skip_dynamic_graphs = False`. The codebook predictor walks a
  growing KV cache (lengths 1..7), which means dynamic shapes. With `True` no
  graphs are recorded for it at all and compiling in `reduce-overhead` becomes
  pointless: 18.4 ms instead of 12.5. The price is 9 recorded graphs and
  +0.27 GB VRAM.
- `dynamic=None` on the predictor, not `False` (another 1.8 ms).

These exact three places were once broken and produced a 92 → 122 ms regression
with ±10 ms of scatter between phrases. After the fix: 92/92/92, no scatter.

## Architecture

`server.py` — aiohttp, serves chunked WAV. The header carries a length of
0xFFFFFFFF so the client keeps playing until the connection closes; the client
strips the first 44 bytes. Inside the HTTP handler, generation runs on a
separate thread (`asyncio.to_thread`) and chunks travel through an
`asyncio.Queue(maxsize=8)` — that queue *is* the backpressure. Resampling
24k→44.1k is streaming, via `soxr.ResampleStream`; the tail is flushed with
`last=True` and gets a 12 ms fade. Language comes from the request body or is
guessed from the writing script. Only one phrase speaks at a time: `take_turn()`
sets a `threading.Event` on the previous one, which then dies with `Cancelled`.
Voice-cloning prompts are cached by sha1(first 2 KB of base64 + reference text).

`streaming.py` — the core. A manual generation loop instead of `model.generate`:
prefill the talker, then on each step call forward with `past_key_values` /
`past_hidden` / `generation_step`, sample the top codec with a local
`_pick_token`, and accumulate frames in a list. Every `emit_every_frames`
frames it decodes a **window** of the last `decode_window_frames` frames, but
only the fresh tail (`decode_tail`) is emitted — the decoder needs the context
for continuity, and re-emitting it would duplicate audio. A short window at the
start is left-padded with reference codes
(`_reference_prefix_for_cold_decoder`, ICL mode only), otherwise with zeros;
the padding is later trimmed by samples-per-frame. The tail after EOS goes out
through `_decode_remainder`. `_service_token_bias` mutes the last 1024
vocabulary tokens except EOS. EOS is checked on the GPU without syncing to the
host.

`talker_inputs.py` — manual assembly of the talker's input embeddings for a
batch of one: tts bos/eos/pad markers, the codec prefix (think / language /
speaker embed / bos), the role taken from the first 3 text tokens, then either
the ICL branch via `generate_icl_prompt` or the first text token plus the rest
of the text as `trailing_text_hidden`. This is the most brittle part — prompt
token offsets are hardcoded here (3 at the start, 5 from the end, 2 for the
reference).

`acceleration.py` — the single entry point for all accelerations,
`enable_all()`. Compilation modes are module-level globals (`DECODER_MODE` and
friends). `fast_codebook.py` and `fast_talker.py` swap `__class__` on a live
object for a subclass that overrides `generate` / `forward` — both return False
and log if the model does not expose the expected fields, breaking nothing.
`fast_codebook` unrolls the codebook predictor's `generate` into an explicit
loop over `lm_head[step]`; `fast_talker` concatenates the codebook embedding
tables into one tensor and sums them with a single `index_select`.
`fast_talker` is off by default.

`window_decoder.py` — bypasses `tokenizer.decode()`: calls
`tokenizer.model.decoder` directly when present, otherwise falls back to the
public API. The copy to host goes through a separate CUDA stream into a pinned
buffer.

`static_cache.py` — an optional `StaticCache`; it never paid off, and on the
first failure it is marked `_static_cache_failed` and never retried.

## Upstream contract

The list of private `qwen_tts` attributes this layer relies on lives in
`streaming.py:UPSTREAM_CONTRACT`. Check against it when upgrading the package —
it is the only place that records what exactly will break.

## Stage profile

RTX 5070 Ti, 5 runs, milliseconds per call. A baseline for future changes:

| stage | ms | share |
|---|---|---|
| talker.forward | 15.47 | 72% |
| └ codebook predictor | 12.52 | 57% |
| talker without the predictor | 2.95 | |
| decode | 9.43 | 11% |

The codebook predictor is the dominant cost, more than half the total. That is
where optimisation is worth spending time; the decoder and input assembly are
already cheap.

The profiler wraps `talker.forward`, `code_predictor.generate` and
`window_decoder.decode` in timers and prints a table. It was not committed —
rewrite it when needed, it is about twenty lines.

## Tried and rejected

`decode_padded`, `max-autotune`, `StaticCache`, batched embedding lookup — no
measurable gain.

Compiling the talker was once listed here by mistake: the wrong level was being
compiled. On the inner forward it is mandatory.

Patching inside the model gives 79 ms but degrades audio — 94 seams versus 16.

`fast_codebook` was also listed here by mistake. A/B within a single run: on
127 ms / 0.354, off 150 ms / 0.427. That is −15% TTFB and −17% RTF, so keep it
on.

Greedy decoding (`do_sample=False`) is unusable for comparisons: without EOS it
degenerates and runs to `max_frames`.

## Measurement methodology

Absolute numbers drift between runs: the same configuration produced 108 and
127 ms within one session. Trust only A/B comparisons inside a single run, with
the same reference and the same phrase set. Every number in this file is the
result of one specific run on one specific card, not a promise.

Always verify by ear. TTFB can be improved by changes that break decoder window
seams, and the metric will not show it — that is what `_seams.wav` is for.

## Code style

Self-documenting. No comments, no docstrings — everything lives in the names.
Log messages and error strings are currently in Russian.

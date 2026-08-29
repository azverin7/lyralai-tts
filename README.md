# lyralai-tts

A low-latency streaming layer on top of [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS),
built for a live AI VTuber that has to answer chat in real time.

**92 ms to first audio, RTF 0.235, 4.49 GB VRAM** on an RTX 5070 Ti with the
1.7B model and voice cloning enabled.

The upstream `qwen-tts` package is never patched. Everything here works from the
outside — swapping `__class__` on live objects, intercepting `forward`, and
replacing `model.generate` with a manual loop that can emit audio while it is
still generating.

## What problem this solves

`model.generate()` returns one finished waveform. For a conversational agent
that is the wrong shape: the listener waits for the entire phrase before hearing
a single sample, and a phrase that is no longer relevant cannot be stopped.

This layer generates frame by frame and decodes a sliding window every few
frames, so audio starts flowing about 90 ms after the request and keeps flowing
until the model emits EOS — or until something cancels it mid-sentence.

## Requirements

- NVIDIA GPU with roughly 5 GB of free VRAM (developed on an RTX 5070 Ti)
- CUDA-capable PyTorch
- `transformers==4.57.3` — the upstream package pins this, and it will not share
  an environment with newer branches. Use a dedicated venv.
- `qwen-tts`, `aiohttp`, `soxr`, `soundfile`, `numpy`
- A Qwen3-TTS checkpoint, by default under `models/qwen3-tts-1.7b-base`

## Running the server

```
.venv/bin/python -u -m lyralai_tts.server --port 8083 --emit 4 --compile \
  --warmup-ref path/to/reference.wav
```

| flag | default | meaning |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `8082` | bind address |
| `--model` | `models/qwen3-tts-1.7b-base` | checkpoint path |
| `--device` | `cuda:0` | target device |
| `--emit` | `4` | decode and emit every N frames |
| `--window` | `80` | decoder context window, in frames |
| `--compile` | off | enable `torch.compile` (strongly recommended) |
| `--warmup-ref` | — | reference wav to warm up with at startup |
| `--warmup-text` | — | transcript of that reference |

Warmup takes about 16 seconds with `--compile`. Until it finishes, `/health`
answers but synthesis will be slow, so wait for it before sending traffic.

### API

`POST /v1/tts` with a JSON body:

```json
{
  "text": "то, что нужно произнести",
  "references": [{ "audio": "<base64 wav>", "text": "transcript of that wav" }],
  "language": "ru",
  "temperature": 0.62,
  "repetition_penalty": 1.1,
  "seed": 1234
}
```

`text` and one `references` entry with an `audio` field are required — this
layer always runs in voice-cloning mode, and a request without a reference is
rejected with 400. `language` falls back to detection from the writing script.
Sampling parameters are optional and clamped to sane ranges: `temperature`
0.05–1.5, `top_p` 0.05–1.0, `top_k` 0–200, `repetition_penalty` 1.0–2.0.
Cloning prompts are cached, so reusing the same reference costs nothing after
the first request.

The response is a chunked WAV stream. The header declares a length of
0xFFFFFFFF so clients keep playing until the connection closes — strip the
first 44 bytes and feed the rest to your audio sink as it arrives.

Only one phrase is spoken at a time. A new request cancels the previous one:
the in-flight generation raises `Cancelled` and its connection closes. That is
deliberate — an agent that is interrupted should stop talking, not queue up.

`GET /health` reports whether the model is on the GPU, how many cloning prompts
are cached, and whether something is currently speaking.

## Benchmarking

There is no conventional test suite here. The measurement is the test:

```
.venv/bin/python -m lyralai_tts.bench --ref path/to/reference.wav \
  --ref-text "$(cat path/to/reference.lab)" --compile
```

It reports TTFB, RTF and peak VRAM. Ablation flags — `--no-fast-codebook`,
`--compile-talker`, `--static-cache`, `--no-decoder` and friends — let you
attribute a change to a specific mechanism.

`--out-dir DIR` writes each phrase as a wav plus a `_seams.wav`, which is the
same chunks with a 120 ms gap inserted at every boundary. That file exists
because latency work can quietly break the seams between decoder windows, and
no metric will tell you — you have to listen.

Cancellation behaviour and memory leaks:

```
.venv/bin/python -m lyralai_tts.check_cancel --ref ... --ref-text ... --cancel-at 1.0
```

### A warning about the numbers

Absolute figures drift between runs. The same configuration produced 108 ms and
127 ms within a single session on the same machine. Only A/B comparisons inside
one run, against the same reference and phrase set, mean anything. Every number
in this README is one run on one card, not a promise about yours.

## How it works

```
request ──▶ talker prefill
              │
              ▼
        manual step loop ──▶ frames ──┐
              ▲                       │  every --emit frames
              └── past_key_values     ▼
                                 window decode (last --window frames)
                                      │  emit only the fresh tail
                                      ▼
                            24k → 44.1k streaming resample
                                      ▼
                              chunked WAV over HTTP
```

The window matters: the decoder needs preceding frames for continuity, but only
the newly produced tail may be emitted. Re-emitting the context would duplicate
audio, and decoding without it produces audible seams.

Three settings carry most of the performance, and getting any of them wrong
costs 30% or more:

- compile the **inner** `talker.model.forward`, not the outer `talker.forward`
  — the outer one contains a Python loop that dynamo re-traces every step
  (203 ms per call instead of 15)
- keep `cudagraph_skip_dynamic_graphs = False` — the codebook predictor runs on
  a growing KV cache, so its shapes are dynamic, and skipping those graphs makes
  `reduce-overhead` compilation pointless (18.4 ms instead of 12.5)
- pass `dynamic=None` to the predictor, not `False`

`CLAUDE.md` documents the internals in detail, including a per-stage profile and
a list of optimisations that were tried and rejected.

## Upstream contract

This layer reaches into private attributes of `qwen_tts`. Every one of them is
listed in `streaming.py:UPSTREAM_CONTRACT`. When bumping the upstream package,
check that list first — it is the only record of what will break.

## Status

Built for one production system and shaped by its needs. It is not a general
purpose library, there is no packaging, and the log messages are in Russian.
Published because the acceleration findings took a long time to arrive at and
may save someone else the same search.

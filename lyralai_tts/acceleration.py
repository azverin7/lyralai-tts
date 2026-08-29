from __future__ import annotations

import logging
from typing import Any

import torch

from lyralai_tts import fast_codebook, fast_talker

log = logging.getLogger("lyralai-tts.compile")

DECODER_MODE = "reduce-overhead"
CODEBOOK_MODE = "reduce-overhead"
TALKER_MODE = "default"
AUTOTUNE_MODE = "max-autotune"


def set_modes(decoder: str, codebook: str, talker: str) -> None:
    global DECODER_MODE, CODEBOOK_MODE, TALKER_MODE
    DECODER_MODE, CODEBOOK_MODE, TALKER_MODE = decoder, codebook, talker
    log.info(f"режимы компиляции: декодер {decoder}, кодовые книги {codebook}, talker {talker}")

RECOMPILE_LIMIT = 128
CACHE_SIZE_LIMIT = 128


def relax_dynamo_limits() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    config = torch._dynamo.config
    config.recompile_limit = RECOMPILE_LIMIT
    config.accumulated_recompile_limit = CACHE_SIZE_LIMIT * 8
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = False
    torch._inductor.config.triton.cudagraph_trees = True
    log.info(
        f"dynamo: лимит перекомпиляций {RECOMPILE_LIMIT}, "
        "графы на динамических формах, деревья графов"
    )


def _compile_forward(module: Any, mode: str, label: str, dynamic: bool | None) -> bool:
    inner = getattr(module, "model", None)
    if inner is None or not hasattr(inner, "forward"):
        log.warning(f"{label}: нечего компилировать")
        return False
    inner.forward = torch.compile(
        inner.forward, mode=mode, fullgraph=False, dynamic=dynamic
    )
    log.info(f"{label}: {mode}, dynamic={dynamic}")
    return True


def compile_talker(model: Any) -> bool:
    return _compile_forward(model.talker, TALKER_MODE, "talker", dynamic=None)


def compile_codebook_predictor(model: Any) -> bool:
    predictor = getattr(model.talker, "code_predictor", None)
    if predictor is None:
        log.warning("code_predictor: не найден")
        return False
    return _compile_forward(predictor, CODEBOOK_MODE, "code_predictor", dynamic=None)


def compile_decoder(model: Any, decode_window_frames: int) -> bool:
    tokenizer = model.speech_tokenizer
    if tokenizer is None:
        log.warning("decoder: токенизатор не загружен")
        return False

    native = getattr(tokenizer, "enable_streaming_optimizations", None)
    if callable(native):
        native(
            decode_window_frames=decode_window_frames,
            use_compile=True,
            use_cuda_graphs=False,
            compile_mode=DECODER_MODE,
        )
        log.info(f"decoder: нативные оптимизации, окно {decode_window_frames}")
        return True

    inner = getattr(tokenizer, "decoder", None) or getattr(tokenizer, "model", None)
    if inner is None or not hasattr(inner, "forward"):
        log.warning("decoder: точка компиляции не найдена")
        return False
    inner.forward = torch.compile(
        inner.forward, mode=DECODER_MODE, fullgraph=False, dynamic=False
    )
    log.info(f"decoder: {DECODER_MODE}, dynamic=False")
    return True


def enable_all(
    model: Any,
    decode_window_frames: int,
    talker: bool = True,
    codebook: bool = True,
    decoder: bool = True,
    fast_codebook_path: bool = True,
    fast_talker_path: bool = True,
    compile_mode: str = "",
) -> dict[str, bool]:
    if compile_mode:
        set_modes(compile_mode, compile_mode, TALKER_MODE)
    relax_dynamo_limits()
    return {
        "fast_codebook": fast_codebook.install(model) if fast_codebook_path else False,
        "fast_talker": fast_talker.install(model) if fast_talker_path else False,
        "talker": compile_talker(model) if talker else False,
        "code_predictor": compile_codebook_predictor(model) if codebook else False,
        "decoder": compile_decoder(model, decode_window_frames) if decoder else False,
    }
from __future__ import annotations

import logging
from typing import Any

import torch

log = logging.getLogger("lyralai-tts.cache")

CACHE_HEADROOM_FRAMES = 64


def build(model: Any, max_frames: int, prompt_length: int) -> Any | None:
    try:
        from transformers.cache_utils import StaticCache
    except ImportError:
        log.warning("StaticCache недоступен в этой версии transformers")
        return None

    talker = model.talker
    capacity = prompt_length + max_frames + CACHE_HEADROOM_FRAMES

    try:
        cache = StaticCache(
            config=talker.config,
            max_batch_size=1,
            max_cache_len=capacity,
            device=talker.device,
            dtype=talker.dtype,
        )
    except TypeError:
        try:
            cache = StaticCache(
                config=talker.config,
                batch_size=1,
                max_cache_len=capacity,
                device=talker.device,
                dtype=talker.dtype,
            )
        except Exception as error:
            log.warning(f"статический кэш не собран: {error}")
            return None
    except Exception as error:
        log.warning(f"статический кэш не собран: {error}")
        return None

    log.info(f"статический кэш на {capacity} позиций")
    return cache


def positions(offset: int, length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(offset, offset + length, device=device)

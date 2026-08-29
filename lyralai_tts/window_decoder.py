from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

log = logging.getLogger("lyralai-tts.decoder")


class WindowDecoder:
    def __init__(self, tokenizer: Any, window_frames: int) -> None:
        self.tokenizer = tokenizer
        self.window_frames = window_frames

        inner = getattr(tokenizer, "model", None)
        self.decoder = getattr(inner, "decoder", None) if inner is not None else None
        self.sample_rate = int(tokenizer.get_output_sample_rate())
        self.fast_path = self.decoder is not None and hasattr(self.decoder, "forward")

        self.copy_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.staging: torch.Tensor | None = None
        self.pending: torch.cuda.Event | None = None
        self.pending_length = 0

        log.info(
            f"декодер окна: {'прямой' if self.fast_path else 'через chunked_decode'}, "
            f"{self.sample_rate} Гц, копирование "
            f"{'асинхронное' if self.copy_stream else 'синхронное'}"
        )

    def compile_forward(self, mode: str) -> bool:
        if not self.fast_path:
            return False
        self.decoder.forward = torch.compile(
            self.decoder.forward, mode=mode, fullgraph=False, dynamic=False
        )
        log.info(f"декодер окна: скомпилирован в режиме {mode}")
        return True

    def decode(self, window: torch.Tensor) -> tuple[np.ndarray, int]:
        if not self.fast_path:
            wavs, sample_rate = self.tokenizer.decode([{"audio_codes": window}])
            return wavs[0].astype(np.float32), sample_rate

        codes, padded_frames = self._pad_to_window(window.unsqueeze(0).transpose(1, 2))
        wav = self.decoder(codes).squeeze(0).squeeze(0)

        if padded_frames:
            samples_per_frame = wav.shape[-1] // self.window_frames
            wav = wav[padded_frames * samples_per_frame :]

        return self._to_host(wav), self.sample_rate

    def decode_tail(
        self, window: torch.Tensor, keep_samples: int
    ) -> tuple[np.ndarray, int]:
        if not self.fast_path:
            wavs, sample_rate = self.tokenizer.decode([{"audio_codes": window}])
            return wavs[0][-keep_samples:].astype(np.float32), sample_rate

        codes, _ = self._pad_to_window(window.unsqueeze(0).transpose(1, 2))
        wav = self.decoder(codes).squeeze(0).squeeze(0)[-keep_samples:]
        return self._to_host(wav), self.sample_rate

    def _pad_to_window(self, codes: torch.Tensor) -> tuple[torch.Tensor, int]:
        present = codes.shape[-1]
        missing = self.window_frames - present
        if missing <= 0:
            return codes.contiguous(), 0
        pad = torch.zeros(
            codes.shape[0],
            codes.shape[1],
            missing,
            dtype=codes.dtype,
            device=codes.device,
        )
        return torch.cat([pad, codes], dim=-1).contiguous(), missing

    def _to_host(self, wav: torch.Tensor) -> np.ndarray:
        length = wav.shape[-1]
        if self.copy_stream is None:
            return wav.to(torch.float32).cpu().numpy()

        self._ensure_staging(length)
        assert self.staging is not None

        self.copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.copy_stream):
            source = wav.to(torch.float32, non_blocking=True)
            self.staging[:length].copy_(source, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.copy_stream)

        event.synchronize()
        return self.staging[:length].numpy().copy()

    def _ensure_staging(self, length: int) -> None:
        if self.staging is not None and self.staging.shape[0] >= length:
            return
        self.staging = torch.empty(
            max(length, self.window_frames * 2048),
            dtype=torch.float32,
            pin_memory=True,
        )

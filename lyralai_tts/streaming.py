from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch

from lyralai_tts import static_cache
from lyralai_tts.acceleration import enable_all
from lyralai_tts.talker_inputs import build_single_talker_input
from lyralai_tts.window_decoder import WindowDecoder

log = logging.getLogger("lyralai-tts.stream")

UPSTREAM_CONTRACT = (
    "model.model.generate_speaker_prompt / generate_icl_prompt",
    "model.model.talker.text_projection / get_text_embeddings / get_input_embeddings",
    "model.model.talker.forward -> .past_key_values .past_hidden .generation_step "
    ".logits .hidden_states[1]",
    "model.model.speech_tokenizer.decode_streaming(codes, pad_to_size=...) "
    "или .decode([{'audio_codes': ...}])",
    "model.model.speech_tokenizer.get_decode_upsample_rate() -> int",
    "model.model.config.talker_config.codec_eos_token_id / vocab_size",
)

CODEC_FRAME_HZ = 12.5
SERVICE_TOKEN_TAIL = 1024


class Cancelled(Exception):
    pass


@dataclass
class VoiceParams:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None


@dataclass
class StreamConfig:
    emit_every_frames: int = 4
    decode_window_frames: int = 80
    max_frames: int = 10_000

    do_sample: bool = True
    top_k: int = 50
    top_p: float = 1.0
    temperature: float = 0.9

    subtalker_dosample: bool = True
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0
    subtalker_temperature: float = 0.9

    repetition_penalty: float = 1.0

    compile_modules: bool = False
    compile_talker: bool = True
    compile_codebook: bool = True
    compile_decoder: bool = True
    fast_codebook: bool = True
    fast_talker: bool = False
    autotune: bool = False
    static_kv_cache: bool = False
    compile_mode: str = ""

    def buffered_ms_before_first_chunk(self) -> float:
        return self.emit_every_frames / CODEC_FRAME_HZ * 1000.0

    def resolved(self, params: VoiceParams | None) -> Sampling:
        if params is None:
            return Sampling(
                self.temperature, self.top_p, self.top_k, self.repetition_penalty
            )
        return Sampling(
            self.temperature if params.temperature is None else params.temperature,
            self.top_p if params.top_p is None else params.top_p,
            self.top_k if params.top_k is None else params.top_k,
            self.repetition_penalty
            if params.repetition_penalty is None
            else params.repetition_penalty,
        )


class Sampling(NamedTuple):
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float


class StreamingSynth:
    def __init__(self, model: Any, config: StreamConfig | None = None) -> None:
        self.wrapper = model
        self.model = model.model
        self.cfg = config or StreamConfig()

        self.tokenizer = self.model.speech_tokenizer
        self.samples_per_frame: int = self.tokenizer.get_decode_upsample_rate()
        self.eos_id: int = self.model.config.talker_config.codec_eos_token_id
        self.token_bias = self._service_token_bias()
        self.window_decoder = WindowDecoder(
            self.tokenizer, self.cfg.decode_window_frames
        )

        log.info(
            f"{self.samples_per_frame} сэмплов/кадр, выдача каждые "
            f"{self.cfg.emit_every_frames} кадров "
            f"(~{self.cfg.buffered_ms_before_first_chunk():.0f} мс буфера)"
        )

        self._needs_graph_step_marks = self.cfg.compile_modules
        self._static_cache_failed = False
        if self.cfg.compile_modules:
            enable_all(
                self.model,
                self.cfg.decode_window_frames,
                talker=self.cfg.compile_talker,
                codebook=self.cfg.compile_codebook,
                decoder=False,
                fast_codebook_path=self.cfg.fast_codebook,
                fast_talker_path=self.cfg.fast_talker,
                compile_mode=self.cfg.compile_mode,
            )
            if self.cfg.compile_decoder:
                from lyralai_tts.acceleration import DECODER_MODE

                self.window_decoder.compile_forward(DECODER_MODE)

    def stream(
        self,
        text: str,
        clone_prompt: Any,
        language: str = "russian",
        cancel: threading.Event | None = None,
        params: VoiceParams | None = None,
    ) -> Iterator[tuple[np.ndarray, int]]:
        items = clone_prompt if isinstance(clone_prompt, list) else [clone_prompt]
        prompt = self.wrapper._prompt_items_to_voice_clone_prompt(items)

        input_ids = self.wrapper._tokenize_texts(
            [self.wrapper._build_assistant_text(text)]
        )
        reference_text = getattr(items[0], "ref_text", None)
        ref_ids = (
            self.wrapper._tokenize_texts([self.wrapper._build_ref_text(reference_text)])
            if reference_text
            else None
        )

        yield from self._generate(
            input_ids, ref_ids, prompt, [language], cancel, params
        )

    @torch.inference_mode()
    def _generate(
        self,
        input_ids: list[torch.Tensor],
        ref_ids: list[torch.Tensor] | None,
        prompt: dict[str, Any],
        languages: list[str],
        cancel: threading.Event | None,
        params: VoiceParams | None = None,
    ) -> Iterator[tuple[np.ndarray, int]]:
        cfg = self.cfg
        talker = self.model.talker
        sampling = cfg.resolved(params)
        if params is not None and params.seed is not None:
            torch.manual_seed(params.seed)
        seen_tokens = torch.zeros_like(self.token_bias, dtype=torch.bool)

        embeds, mask, trailing, pad_embed = build_single_talker_input(
            model=self.model,
            input_id=input_ids[0],
            ref_id=ref_ids[0] if ref_ids else None,
            prompt=prompt,
            language=languages[0],
        )

        if self._needs_graph_step_marks:
            torch.compiler.cudagraph_mark_step_begin()
        prompt_length = embeds.shape[1]
        prepared_cache = self._prepare_cache(prompt_length)

        prefill = talker.forward(
            inputs_embeds=embeds,
            attention_mask=mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            trailing_text_hidden=trailing,
            tts_pad_embed=pad_embed,
            generation_step=None,
            past_hidden=None,
            past_key_values=prepared_cache,
            **self._subtalker_kwargs(),
        )

        past_key_values = prefill.past_key_values
        past_hidden = prefill.past_hidden
        generation_step = prefill.generation_step
        token = self._pick_token(prefill.logits[:, -1, :], sampling, seen_tokens)

        cold_start_prefix = self._reference_prefix_for_cold_decoder(prompt)
        frames: list[torch.Tensor] = []
        frames_already_emitted = 0
        frames_since_emit = 0

        for _ in range(cfg.max_frames):
            if cancel is not None and cancel.is_set():
                log.info(f"Оборвано на кадре {len(frames)}")
                raise Cancelled

            if self._needs_graph_step_marks:
                torch.compiler.cudagraph_mark_step_begin()
            step = talker.forward(
                input_ids=token.unsqueeze(1),
                use_cache=True,
                return_dict=True,
                output_hidden_states=False,
                past_key_values=past_key_values,
                past_hidden=past_hidden,
                generation_step=generation_step,
                trailing_text_hidden=trailing,
                tts_pad_embed=pad_embed,
                **self._subtalker_kwargs(),
            )
            past_key_values = step.past_key_values
            past_hidden = step.past_hidden
            generation_step = step.generation_step

            codec_ids = step.hidden_states[1]
            if self._is_eos_without_host_sync(codec_ids):
                break

            frames.append(codec_ids[0].detach())
            seen_tokens[codec_ids[0, 0]] = True
            token = self._pick_token(step.logits[:, -1, :], sampling, seen_tokens)

            frames_since_emit += 1
            if frames_since_emit < cfg.emit_every_frames:
                continue
            frames_since_emit = 0

            yield self._decode_fresh_tail(frames, cold_start_prefix)
            frames_already_emitted = len(frames)

        if len(frames) > frames_already_emitted:
            yield self._decode_remainder(
                frames, frames_already_emitted, cold_start_prefix
            )

    def _prepare_cache(self, prompt_length: int) -> Any:
        if not self.cfg.static_kv_cache:
            return None
        if self._static_cache_failed:
            return None
        cache = static_cache.build(self.model, self.cfg.max_frames, prompt_length)
        if cache is None:
            self._static_cache_failed = True
        return cache

    def _subtalker_kwargs(self) -> dict[str, Any]:
        return {
            "subtalker_dosample": self.cfg.subtalker_dosample,
            "subtalker_top_k": self.cfg.subtalker_top_k,
            "subtalker_top_p": self.cfg.subtalker_top_p,
            "subtalker_temperature": self.cfg.subtalker_temperature,
        }

    def _service_token_bias(self) -> torch.Tensor:
        vocab_size = self.model.config.talker_config.vocab_size
        bias = torch.zeros(vocab_size, device=self.model.talker.device)
        bias[vocab_size - SERVICE_TOKEN_TAIL :] = float("-inf")
        bias[self.eos_id] = 0.0
        return bias

    def _is_eos_without_host_sync(self, codec_ids: torch.Tensor) -> bool:
        return bool(codec_ids[0, 0] == self.eos_id)

    def _pick_token(
        self, logits: torch.Tensor, sampling: Sampling, seen: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.cfg
        biased = logits + self.token_bias

        if sampling.repetition_penalty != 1.0:
            penalty = sampling.repetition_penalty
            penalised = torch.where(biased > 0, biased / penalty, biased * penalty)
            biased = torch.where(seen.unsqueeze(0), penalised, biased)

        if not cfg.do_sample or sampling.temperature <= 0:
            return torch.argmax(biased, dim=-1)

        if sampling.temperature != 1.0:
            biased = biased / sampling.temperature

        if sampling.top_k > 0:
            kth_largest = torch.topk(biased, sampling.top_k, dim=-1).values[:, -1:]
            biased = biased.masked_fill(biased < kth_largest, float("-inf"))

        if sampling.top_p < 1.0:
            ordered, original_index = torch.sort(biased, descending=True, dim=-1)
            cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
            beyond_nucleus = cumulative > sampling.top_p
            beyond_nucleus[:, 0] = False
            ordered = ordered.masked_fill(beyond_nucleus, float("-inf"))
            biased = torch.full_like(biased, float("-inf")).scatter(
                1, original_index, ordered
            )

        return torch.multinomial(torch.softmax(biased, dim=-1), 1).squeeze(1)

    def _reference_prefix_for_cold_decoder(
        self, prompt: dict[str, Any]
    ) -> torch.Tensor | None:
        reference_codes = (prompt.get("ref_code") or [None])[0]
        in_context_mode = (prompt.get("icl_mode") or [False])[0]
        if reference_codes is None or not in_context_mode:
            return None
        return reference_codes.to(self.model.talker.device)

    def _fixed_size_window(
        self,
        frames: list[torch.Tensor],
        start: int,
        prefix: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int]:
        window = torch.stack(frames[start:], dim=0)
        target = self.cfg.decode_window_frames
        missing = target - window.shape[0]
        if missing <= 0:
            return window, 0

        pieces: list[torch.Tensor] = []
        if prefix is not None:
            taken = prefix[-missing:]
            pieces.append(taken)
            missing -= taken.shape[0]
        if missing > 0:
            pieces.append(torch.zeros_like(window[:1]).expand(missing, -1))

        padding_frames = target - window.shape[0]
        return torch.cat(pieces + [window], dim=0), padding_frames

    def _decode(self, window: torch.Tensor) -> tuple[np.ndarray, int]:
        return self.window_decoder.decode(window.to(self.model.talker.device))

    def _decode_fresh_tail(
        self, frames: list[torch.Tensor], prefix: torch.Tensor | None
    ) -> tuple[np.ndarray, int]:
        start = max(0, len(frames) - self.cfg.decode_window_frames)
        window, _ = self._fixed_size_window(frames, start, prefix)
        fresh_samples = self.samples_per_frame * self.cfg.emit_every_frames
        return self.window_decoder.decode_tail(
            window.to(self.model.talker.device), fresh_samples
        )

    def _decode_remainder(
        self,
        frames: list[torch.Tensor],
        already_emitted: int,
        prefix: torch.Tensor | None,
    ) -> tuple[np.ndarray, int]:
        pending = len(frames) - already_emitted
        context = min(already_emitted, self.cfg.decode_window_frames - pending)
        start = already_emitted - context

        window, padding_frames = self._fixed_size_window(frames, start, prefix)
        wav, sample_rate = self._decode(window)

        stale_samples = (padding_frames + context) * self.samples_per_frame
        return wav[stale_samples:], sample_rate

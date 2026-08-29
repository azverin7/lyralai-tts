from __future__ import annotations

from typing import Any

import torch

TEXT_ROLE_TOKENS = 3
TEXT_TAIL_TOKENS = 5
REF_TAIL_TOKENS = 2


def _language_id(config: Any, language: str) -> int | None:
    if language.lower() == "auto":
        return None
    known = config.talker_config.codec_language_id
    if language.lower() not in known:
        raise NotImplementedError(f"Язык {language} не поддерживается моделью")
    return known[language.lower()]


def _tts_marker_embeds(model: Any, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    talker = model.talker
    markers = torch.tensor(
        [
            [
                model.config.tts_bos_token_id,
                model.config.tts_eos_token_id,
                model.config.tts_pad_token_id,
            ]
        ],
        device=talker.device,
        dtype=dtype,
    )
    return talker.text_projection(talker.get_text_embeddings()(markers)).chunk(3, dim=1)


def _codec_prefill_ids(config: Any, language_id: int | None) -> list[list[int]]:
    talker_config = config.talker_config
    if language_id is None:
        return [
            [
                talker_config.codec_nothink_id,
                talker_config.codec_think_bos_id,
                talker_config.codec_think_eos_id,
            ]
        ]
    return [
        [
            talker_config.codec_think_id,
            talker_config.codec_think_bos_id,
            language_id,
            talker_config.codec_think_eos_id,
        ]
    ]


def _codec_prefix_embed(
    model: Any,
    language_id: int | None,
    speaker_embed: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    talker = model.talker
    talker_config = model.config.talker_config

    think = talker.get_input_embeddings()(
        torch.tensor(
            _codec_prefill_ids(model.config, language_id),
            device=talker.device,
            dtype=dtype,
        )
    )
    opening = talker.get_input_embeddings()(
        torch.tensor(
            [[talker_config.codec_pad_id, talker_config.codec_bos_id]],
            device=talker.device,
            dtype=dtype,
        )
    )

    if speaker_embed is None:
        return torch.cat([think, opening], dim=1)
    return torch.cat([think, speaker_embed.view(1, 1, -1), opening], dim=1)


def _speaker_embed(model: Any, prompt: dict[str, Any]) -> torch.Tensor | None:
    if prompt is None:
        return None
    if not (prompt["x_vector_only_mode"][0] or prompt["icl_mode"][0]):
        return None
    return model.generate_speaker_prompt(prompt)[0]


def build_single_talker_input(
    model: Any,
    input_id: torch.Tensor,
    ref_id: torch.Tensor | None,
    prompt: dict[str, Any],
    language: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    talker = model.talker
    dtype = input_id.dtype

    language_id = _language_id(model.config, language)
    bos_embed, eos_embed, pad_embed = _tts_marker_embeds(model, dtype)
    codec_prefix = _codec_prefix_embed(
        model, language_id, _speaker_embed(model, prompt), dtype
    )

    role_embed = talker.text_projection(
        talker.get_text_embeddings()(input_id[:, :TEXT_ROLE_TOKENS])
    )
    opening_embed = (
        torch.cat(
            [pad_embed.expand(-1, codec_prefix.shape[1] - 2, -1), bos_embed], dim=1
        )
        + codec_prefix[:, :-1]
    )
    talker_input = torch.cat([role_embed, opening_embed], dim=1)

    reference_codes = (prompt or {}).get("ref_code", [None])[0]
    in_context_mode = bool((prompt or {}).get("icl_mode", [False])[0])

    if reference_codes is not None and in_context_mode and ref_id is not None:
        in_context_embed, trailing_text = model.generate_icl_prompt(
            text_id=input_id[:, TEXT_ROLE_TOKENS:-TEXT_TAIL_TOKENS],
            ref_id=ref_id[:, TEXT_ROLE_TOKENS:-REF_TAIL_TOKENS],
            ref_code=reference_codes.to(talker.device),
            tts_pad_embed=pad_embed,
            tts_eos_embed=eos_embed,
            non_streaming_mode=False,
        )
        talker_input = torch.cat([talker_input, in_context_embed], dim=1)
    else:
        first_text_embed = talker.text_projection(
            talker.get_text_embeddings()(
                input_id[:, TEXT_ROLE_TOKENS : TEXT_ROLE_TOKENS + 1]
            )
        )
        talker_input = torch.cat(
            [talker_input, first_text_embed + codec_prefix[:, -1:]], dim=1
        )
        trailing_text = torch.cat(
            [
                talker.text_projection(
                    talker.get_text_embeddings()(
                        input_id[:, TEXT_ROLE_TOKENS + 1 : -TEXT_TAIL_TOKENS]
                    )
                ),
                eos_embed,
            ],
            dim=1,
        )

    attention_mask = torch.ones(
        (1, talker_input.shape[1]), dtype=torch.long, device=talker_input.device
    )
    return talker_input, attention_mask, trailing_text, pad_embed
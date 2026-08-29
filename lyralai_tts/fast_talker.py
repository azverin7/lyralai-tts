from __future__ import annotations

import logging
from typing import Any

import torch

log = logging.getLogger("lyralai-tts.talker")


def stack_codebook_weights(predictor: Any) -> torch.Tensor:
    tables = predictor.get_input_embeddings()
    return torch.stack([table.weight for table in tables], dim=0).contiguous()


def build_fast_class(original: type, output_class: type) -> type:
    class FastTalker(original):
        codebook_weights: torch.Tensor | None = None

        def _summed_codec_hidden(
            self, last_id_hidden: torch.Tensor, sequences: torch.Tensor
        ) -> torch.Tensor:
            weights = self.codebook_weights
            groups, vocab, hidden = weights.shape
            indices = sequences[0, :groups]
            offsets = torch.arange(groups, device=indices.device) * vocab
            picked = weights.view(-1, hidden).index_select(0, indices + offsets)
            return last_id_hidden.squeeze(1) + picked.sum(0, keepdim=True)

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            cache_position=None,
            past_hidden=None,
            trailing_text_hidden=None,
            tts_pad_embed=None,
            generation_step=None,
            subtalker_dosample=None,
            subtalker_top_p=None,
            subtalker_top_k=None,
            subtalker_temperature=None,
            **kwargs: Any,
        ) -> Any:
            if self.codebook_weights is None or (
                inputs_embeds is not None and inputs_embeds.shape[1] > 1
            ):
                return super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    cache_position=cache_position,
                    past_hidden=past_hidden,
                    trailing_text_hidden=trailing_text_hidden,
                    tts_pad_embed=tts_pad_embed,
                    generation_step=generation_step,
                    subtalker_dosample=subtalker_dosample,
                    subtalker_top_p=subtalker_top_p,
                    subtalker_top_k=subtalker_top_k,
                    subtalker_temperature=subtalker_temperature,
                    **kwargs,
                )

            last_id_hidden = self.get_input_embeddings()(input_ids)
            predictor_result = self.code_predictor.generate(
                inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
                max_new_tokens=self.config.num_code_groups - 1,
                do_sample=subtalker_dosample,
                top_p=subtalker_top_p,
                top_k=subtalker_top_k,
                temperature=subtalker_temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            codec_ids = torch.cat((input_ids, predictor_result.sequences), dim=-1)
            inputs_embeds = self._summed_codec_hidden(
                last_id_hidden, predictor_result.sequences
            ).unsqueeze(1)

            if generation_step < trailing_text_hidden.shape[1]:
                inputs_embeds = inputs_embeds + trailing_text_hidden[
                    :, generation_step
                ].unsqueeze(1)
            else:
                inputs_embeds = inputs_embeds + tts_pad_embed

            batch_size, seq_length = input_ids.shape
            delta = (
                cache_position[0] + self.rope_deltas
                if cache_position is not None
                else self.rope_deltas
            )
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = outputs.last_hidden_state
            logits = self.codec_head(hidden_states)

            return output_class(
                loss=None,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=(outputs.hidden_states, codec_ids),
                attentions=outputs.attentions,
                past_hidden=hidden_states[:, -1:, :],
                generation_step=generation_step + 1,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
            )

    FastTalker.__name__ = f"Fast{original.__name__}"
    return FastTalker


def install(model: Any) -> bool:
    talker = model.talker
    predictor = getattr(talker, "code_predictor", None)
    if predictor is None:
        log.warning("code_predictor не найден")
        return False

    try:
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerOutputWithPast
    except ImportError as error:
        log.warning(f"класс вывода talker не найден: {error}")
        return False

    try:
        weights = stack_codebook_weights(predictor)
    except Exception as error:
        log.warning(f"таблицы эмбеддингов не склеились: {error}")
        return False

    if type(talker).__name__.startswith("Fast"):
        talker.codebook_weights = weights
        return True

    try:
        talker.__class__ = build_fast_class(
            type(talker), Qwen3TTSTalkerOutputWithPast
        )
    except TypeError as error:
        log.warning(f"подмена класса talker не удалась: {error}")
        return False

    talker.codebook_weights = weights
    log.info(
        f"talker: {weights.shape[0]} таблиц склеены в тензор {tuple(weights.shape)}"
    )
    return True

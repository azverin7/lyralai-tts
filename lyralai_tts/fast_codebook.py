from __future__ import annotations

import logging
from typing import Any

import torch

log = logging.getLogger("lyralai-tts.codebook")


class FastGenerationResult:
    def __init__(self, sequences: torch.Tensor) -> None:
        self.sequences = sequences


def _sample(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if not do_sample or temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    if temperature != 1.0:
        logits = logits / temperature

    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        ordered, index = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        beyond = cumulative > top_p
        beyond[..., 0] = False
        logits = logits.masked_fill(beyond.scatter(1, index, beyond), float("-inf"))

    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


def build_fast_class(original: type) -> type:
    class FastCodePredictor(original):
        @torch.inference_mode()
        def generate(self, *args: Any, **kwargs: Any) -> Any:
            inputs_embeds = kwargs.get("inputs_embeds")
            steps = kwargs.get("max_new_tokens")
            if inputs_embeds is None or steps is None:
                return super().generate(*args, **kwargs)

            do_sample = bool(kwargs.get("do_sample", True))
            temperature = float(kwargs.get("temperature") or 1.0)
            top_k = int(kwargs.get("top_k") or 0)
            top_p = float(kwargs.get("top_p") or 1.0)

            embeddings = self.model.get_input_embeddings()
            outputs = self.model(
                input_ids=None,
                inputs_embeds=self.small_to_mtp_projection(inputs_embeds),
                use_cache=True,
                output_hidden_states=False,
            )
            past = outputs.past_key_values
            hidden = outputs.last_hidden_state

            tokens: list[torch.Tensor] = []
            for step in range(steps):
                logits = self.lm_head[step](hidden[:, -1, :])
                token = _sample(logits, do_sample, temperature, top_k, top_p)
                tokens.append(token)

                if step == steps - 1:
                    break

                outputs = self.model(
                    input_ids=None,
                    inputs_embeds=self.small_to_mtp_projection(
                        embeddings[step](token)
                    ),
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=False,
                )
                past = outputs.past_key_values
                hidden = outputs.last_hidden_state

            return FastGenerationResult(torch.cat(tokens, dim=-1))

    FastCodePredictor.__name__ = f"Fast{original.__name__}"
    return FastCodePredictor


def install(model: Any) -> bool:
    predictor = getattr(model.talker, "code_predictor", None)
    if predictor is None:
        log.warning("code_predictor не найден")
        return False

    required = ("model", "small_to_mtp_projection", "lm_head")
    missing = [name for name in required if not hasattr(predictor, name)]
    if missing:
        log.warning(f"code_predictor без полей {missing}")
        return False

    if type(predictor).__name__.startswith("Fast"):
        return True

    try:
        predictor.__class__ = build_fast_class(type(predictor))
    except TypeError as error:
        log.warning(f"подмена класса не удалась: {error}")
        return False

    log.info("code_predictor: класс подменён на быстрый")
    return True
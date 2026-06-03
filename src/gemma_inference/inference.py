import time
from dataclasses import dataclass

import torch

from .model import get_model


@dataclass
class GenerationRequest:
    prompt: str
    system_prompt: str | None = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


@dataclass
class GenerationResult:
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    tokens_per_second: float


def generate(req: GenerationRequest) -> GenerationResult:
    model, tokenizer = get_model()

    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": req.prompt})

    # apply_chat_template handles Gemma's <start_of_turn>/<end_of_turn> format
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    prompt_len = input_ids.shape[-1]
    t0 = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature if req.do_sample else 1.0,
            top_p=req.top_p if req.do_sample else 1.0,
            do_sample=req.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.perf_counter() - t0
    new_token_count = output_ids.shape[-1] - prompt_len
    generated_text = tokenizer.decode(
        output_ids[0][prompt_len:], skip_special_tokens=True
    )

    return GenerationResult(
        generated_text=generated_text,
        prompt_tokens=prompt_len,
        generated_tokens=new_token_count,
        elapsed_seconds=round(elapsed, 3),
        tokens_per_second=round(new_token_count / elapsed, 2) if elapsed > 0 else 0.0,
    )

import time
from dataclasses import dataclass

from .model import get_engine


@dataclass
class GenerationRequest:
    prompt: str
    system_prompt: str | None = None


@dataclass
class GenerationResult:
    generated_text: str
    elapsed_seconds: float
    tokens_per_second: float  # approximate (word-based), LiteRT CPU API does not expose token counts


def generate(req: GenerationRequest) -> GenerationResult:
    import litert_lm

    engine = get_engine()

    # Set system prompt as an initial message if provided
    initial_messages = []
    if req.system_prompt:
        initial_messages.append(litert_lm.Message.system(req.system_prompt))

    t0 = time.perf_counter()
    collected_parts: list[str] = []

    with engine.create_conversation(messages=initial_messages) as conversation:
        stream = conversation.send_message_async(req.prompt)
        for chunk in stream:
            for item in chunk.get("content", []):
                if item.get("type") == "text":
                    collected_parts.append(item["text"])

    elapsed = time.perf_counter() - t0
    generated_text = "".join(collected_parts)

    # Approximate tokens/sec using whitespace-split word count as a proxy
    word_count = len(generated_text.split())
    tokens_per_second = round(word_count / elapsed, 2) if elapsed > 0 else 0.0

    return GenerationResult(
        generated_text=generated_text,
        elapsed_seconds=round(elapsed, 3),
        tokens_per_second=tokens_per_second,
    )

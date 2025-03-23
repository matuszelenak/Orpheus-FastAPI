import os
import time
from typing import Optional, Generator, AsyncGenerator

from openai import AsyncClient

from tts_engine.constants import API_URL, DEFAULT_VOICE, AVAILABLE_VOICES, TEMPERATURE, TOP_P, MAX_TOKENS, \
    REPETITION_PENALTY, CUSTOM_TOKEN_PREFIX
from tts_engine.log import get_logger
from tts_engine.speechpipe import convert_to_audio
from tts_engine.string_utils import split_text_into_sentences

logger = get_logger(__name__)

client = AsyncClient(base_url=API_URL, api_key='none')


def format_prompt(prompt: str, voice: str = DEFAULT_VOICE) -> str:
    """Format prompt for Orpheus model with voice prefix and special tokens."""
    # Validate voice and provide fallback
    if voice not in AVAILABLE_VOICES:
        logger.debug(f"Warning: Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead.")
        voice = DEFAULT_VOICE

    # Format similar to how engine_class.py does it with special tokens
    formatted_prompt = f"{voice}: {prompt}"

    # Add special token markers for the Orpheus-FASTAPI
    special_start = "<|audio|>"  # Using the additional_special_token from config
    special_end = "<|eot_id|>"  # Using the eos_token from config

    return f"{special_start}{formatted_prompt}{special_end}"


async def generate_tokens_from_api(
    prompt: str,
    voice: str = DEFAULT_VOICE,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    repetition_penalty: float = REPETITION_PENALTY
) -> AsyncGenerator[str, None]:
    formatted_prompt = format_prompt(prompt, voice)

    token_count = 0
    async for part in await client.completions.create(
            model=os.environ.get('ORPHEUS_MODEL_NAME'),
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=repetition_penalty,
            stream=True
    ):
        token_count += 1
        yield part.choices[0].text

    logger.debug(f'Received {token_count} tokens')


token_id_cache = {}
MAX_CACHE_SIZE = 10000


def turn_token_into_id(token_string: str, index: int) -> Optional[int]:
    """Optimized token-to-ID conversion with caching."""
    cache_key = (token_string, index % 7)
    if cache_key in token_id_cache:
        return token_id_cache[cache_key]

    if CUSTOM_TOKEN_PREFIX not in token_string:
        return None

    token_string = token_string.strip()
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)

    if last_token_start == -1:
        return None

    last_token = token_string[last_token_start:]

    if not (last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">")):
        return None

    try:
        number_str = last_token[14:-1]
        token_id = int(number_str) - 10 - ((index % 7) * 4096)

        if len(token_id_cache) < MAX_CACHE_SIZE:
            token_id_cache[cache_key] = token_id

        return token_id
    except (ValueError, IndexError):
        return None


async def tokens_decoder(token_gen: AsyncGenerator[str, None]) -> Generator[bytes, None, None]:
    """Simplified token decoder without complex ring buffer to ensure reliable output."""
    buffer = []
    count = 0

    min_frames = 28
    process_every = 7

    start_time = time.time()
    last_log_time = start_time
    token_count = 0

    async for token_text in token_gen:
        token = turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1
            token_count += 1

            current_time = time.time()
            if current_time - last_log_time > 5.0:  # Every 5 seconds
                elapsed = current_time - start_time
                if elapsed > 0:
                    logger.debug(f"Token processing rate: {token_count / elapsed:.1f} tokens/second")
                last_log_time = current_time

            if count % process_every == 0 and count >= min_frames:
                buffer_to_proc = buffer[-min_frames:]

                if count % 28 == 0:
                    logger.debug(f"Processing buffer with {len(buffer_to_proc)} tokens, total collected: {len(buffer)}")

                # Process the tokens
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    yield audio_samples


async def generate_speech_chunks_from_api(
    prompt, voice=DEFAULT_VOICE,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS,
    repetition_penalty=None,
    use_batching=True,
    max_batch_chars=1000
):
    """Generate speech from text using Orpheus model with performance optimizations."""
    logger.debug(f"Starting speech generation for '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")

    # For shorter text, use the standard non-batched approach
    if not use_batching or len(prompt) < max_batch_chars:
        async for audio_chunk in tokens_decoder(
                generate_tokens_from_api(
                    prompt=prompt,
                    voice=voice,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=REPETITION_PENALTY  # Always use hardcoded value
                )
        ):
            yield audio_chunk
        return

    # For longer text, use sentence-based batching
    logger.debug(f"Using sentence-based batching for text with {len(prompt)} characters")

    sentences = split_text_into_sentences(prompt)
    logger.debug(f"Split text into {len(sentences)} segments")

    batches = []
    current_batch = ""

    for sentence in sentences:
        if len(current_batch) + len(sentence) > max_batch_chars and current_batch:
            batches.append(current_batch)
            current_batch = sentence
        else:
            if current_batch:
                current_batch += " "
            current_batch += sentence

    if current_batch:
        batches.append(current_batch)

    logger.debug(f"Created {len(batches)} batches for processing")

    for i, batch in enumerate(batches):
        logger.debug(f"Processing batch {i + 1}/{len(batches)} ({len(batch)} characters)")

        async for audio_chunk in tokens_decoder(
                generate_tokens_from_api(
                    prompt=batch,
                    voice=voice,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=REPETITION_PENALTY
                )
        ):
            yield audio_chunk

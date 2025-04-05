import os
from typing import Generator, AsyncGenerator, Optional

from openai import AsyncClient
from sentence_splitter import SentenceSplitter

from tts_engine.constants import API_URL, DEFAULT_VOICE, AVAILABLE_VOICES, TEMPERATURE, TOP_P, MAX_TOKENS, \
    REPETITION_PENALTY, CUSTOM_TOKEN_PREFIX
from tts_engine.log import get_logger
from tts_engine.speechpipe import convert_to_audio

logger = get_logger(__name__)


def format_prompt(prompt: str, voice: str = DEFAULT_VOICE) -> str:
    """Format prompt for Orpheus model with voice prefix and special tokens."""
    if voice not in AVAILABLE_VOICES:
        logger.debug(f"Warning: Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead.")
        voice = DEFAULT_VOICE

    formatted_prompt = f"{voice}: {prompt}"

    special_start = "<|audio|>"
    special_end = "<|eot_id|>"

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

    client = AsyncClient(base_url=API_URL, api_key='None')

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
    """Optimized token decoder with early first-chunk processing for lower latency"""
    buffer = []
    count = 0

    first_chunk_processed = False

    # Use different thresholds for first chunk vs. subsequent chunks
    min_frames_first = 7  # Just one chunk (7 tokens) for first audio - ultra-low latency
    min_frames_subsequent = 28  # Standard minimum (4 chunks of 7 tokens) after first audio
    ideal_frames = 49  # Ideal standard frame size (7×7 window) - unchanged
    process_every_n = 7  # Process every 7 tokens (standard for Orpheus model) - unchanged

    async for token_sim in token_gen:
        token = turn_token_into_id(token_sim, count)

        if token is not None and token > 0:
            buffer.append(token)
            count += 1

            # Different processing logic based on whether first chunk has been processed
            if not first_chunk_processed:
                if count >= min_frames_first:
                    buffer_to_proc = buffer[-min_frames_first:]

                    audio_samples = convert_to_audio(buffer_to_proc)
                    if audio_samples is not None:
                        first_chunk_processed = True  # Mark first chunk as processed
                        yield audio_samples
            else:
                # For subsequent chunks, use original processing with proper batching
                if count % process_every_n == 0:
                    # Use same prioritization logic as before
                    if len(buffer) >= ideal_frames:
                        buffer_to_proc = buffer[-ideal_frames:]
                    elif len(buffer) >= min_frames_subsequent:
                        buffer_to_proc = buffer[-min_frames_subsequent:]
                    else:
                        continue

                    audio_samples = convert_to_audio(buffer_to_proc)
                    if audio_samples is not None:
                        yield audio_samples

    if len(buffer) >= ideal_frames:
        buffer_to_proc = buffer[-ideal_frames:]
        audio_samples = convert_to_audio(buffer_to_proc)
        if audio_samples is not None:
            yield audio_samples

    elif len(buffer) >= min_frames_subsequent:
        buffer_to_proc = buffer[-min_frames_subsequent:]
        audio_samples = convert_to_audio(buffer_to_proc)
        if audio_samples is not None:
            yield audio_samples

    # Final special case: even if we don't have minimum frames, try to process
    # what we have by padding with silence tokens that won't affect the audio
    elif len(buffer) >= process_every_n:
        # Pad to minimum frame requirement with copies of the final token
        # This is more continuous than using unrelated tokens from the beginning
        last_token = buffer[-1]
        padding_needed = min_frames_subsequent - len(buffer)

        # Create a padding array of copies of the last token
        # This maintains continuity much better than circular buffering
        padding = [last_token] * padding_needed
        padded_buffer = buffer + padding

        audio_samples = convert_to_audio(padded_buffer)
        if audio_samples is not None:
            yield audio_samples


async def generate_speech_chunks_from_api(
        prompt,
        voice=DEFAULT_VOICE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        repetition_penalty=None,
        use_batching=True,
        max_batch_chars=500
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

    splitter = SentenceSplitter(language='en')
    sentences = splitter.split(prompt)
    logger.debug(f"Split text into {len(sentences)} segments")

    batches = []
    current_batch = ""

    for sentence in sentences:
        logger.debug(sentence)
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

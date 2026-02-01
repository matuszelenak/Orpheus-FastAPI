import os
from typing import AsyncGenerator, Optional

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
    logger.debug(f'Submitting to LLM: {prompt}')

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


async def tokens_decoder_original(token_gen: AsyncGenerator[str, None]) -> AsyncGenerator[bytes, None]:
    buffer = []
    count = 0

    async for token_sim in token_gen:
        token = turn_token_into_id(token_sim, count)
        if token is None:
            pass
        else:
            if token > 0:
                buffer.append(token)
                count += 1

                if count % 7 == 0 and count > 27:
                    buffer_to_proc = buffer[-28:]
                    audio_samples = convert_to_audio(buffer_to_proc)
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

    logger.debug(f"Using sentence-based batching for text with {len(prompt)} characters")

    splitter = SentenceSplitter(language='en')
    sentences = splitter.split(prompt)
    logger.debug(f"Split text into {len(sentences)} segments")

    for i, sentence in enumerate(sentences):
        logger.debug(f"Processing batch {i + 1}/{len(sentences)} ({len(sentence)} characters)")

        async for audio_chunk in tokens_decoder_original(
                generate_tokens_from_api(
                    prompt=sentence,
                    voice=voice,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=REPETITION_PENALTY
                )
        ):
            yield audio_chunk

import asyncio
import time
import wave
from dotenv import load_dotenv

load_dotenv()

from tts_engine.constants import SAMPLE_RATE
from tts_engine.inference import generate_speech_chunks_from_api
from tts_engine.log import get_logger

logger = get_logger(__name__)


async def gather_chunks(prompt: str):

    wav_file = wave.open('outputs/output.wav', "wb")
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(SAMPLE_RATE)

    start_time = time.time()
    samples = 0
    async for chunk in generate_speech_chunks_from_api(
            prompt=prompt,
            voice='tara',
            use_batching=True,
            max_batch_chars=150
    ):
        samples += len(chunk) // 2

        wav_file.writeframes(chunk)

    end_time = time.time()
    duration_s = end_time - start_time
    logger.debug(f'Generation took {duration_s:.2f}s for {(samples / SAMPLE_RATE):.2f}s of audio')
    if wav_file:
        wav_file.close()


async def main():
    with open('copypasta', 'r') as f:
        content = f.read()

    await gather_chunks(content)

if __name__ == '__main__':
    asyncio.run(main())
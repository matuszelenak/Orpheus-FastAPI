# Orpheus-FASTAPI by Lex-au
# https://github.com/Lex-au/Orpheus-FastAPI
# Description: Main FastAPI server for Orpheus Text-to-Speech

import wave
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from tts_engine.constants import SAMPLE_RATE
from tts_engine.inference import generate_speech_chunks_from_api, DEFAULT_VOICE
from tts_engine.log import get_logger

app = FastAPI(
    title="Orpheus-FASTAPI",
    description="High-performance Text-to-Speech server using Orpheus-FASTAPI",
    version="1.0.0"
)


logger = get_logger(__name__)


class SpeechRequest(BaseModel):
    input: str
    model: str = "orpheus"
    voice: str = DEFAULT_VOICE
    response_format: str = "wav"
    speed: float = 1.0
    stream: bool = False


@app.post("/v1/audio/speech")
async def create_speech_api(request: SpeechRequest):
    """
    Generate speech from text using the Orpheus TTS model.
    Compatible with OpenAI's /v1/audio/speech endpoint.
    
    For longer texts (>1000 characters), batched generation is used
    to improve reliability and avoid truncation issues.
    """
    if not request.input:
        raise HTTPException(status_code=400, detail="Missing input text")

    # Check if we should use batched generation
    use_batching = len(request.input) > 1000
    if use_batching:
        logger.debug(f"Using batched generation for long text ({len(request.input)} characters)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{request.voice}_{timestamp}.wav"

    async def gather_chunks():
        if not request.stream:
            wav_file = wave.open(output_path, "wb")
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
        else:
            wav_file = None

        async for chunk in generate_speech_chunks_from_api(
                prompt=request.input,
                voice=request.voice,
                use_batching=use_batching,
                max_batch_chars=1000
        ):
            logger.debug(f'Chunk {len(chunk)}')
            yield chunk

            if wav_file:
                wav_file.writeframes(chunk)

        if wav_file:
            wav_file.close()

    if request.stream:
        return StreamingResponse(
            gather_chunks()
        )
    else:
        async for _ in gather_chunks():
            pass
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=f"{request.voice}_{timestamp}.wav"
        )

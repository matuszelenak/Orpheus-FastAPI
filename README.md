# Orpheus-FASTAPI

[![GitHub](https://img.shields.io/github/license/Lex-au/Orpheus-FastAPI)](https://github.com/Lex-au/Orpheus-FastAPI/blob/main/LICENSE.txt)

High-performance Text-to-Speech server with OpenAI-compatible API, 8 voices, emotion tags, and modern web UI. Optimized for RTX GPUs.

## Changelog

**v1.1.0** (2025-03-23)
- ✨ Added long-form audio support with sentence-based batching and crossfade stitching
- 🔊 Improved short audio quality with optimized token buffer handling
- 🔄 Enhanced environment variable support with .env file loading (configurable via UI)
- 🖥️ Added automatic hardware detection and optimization for different GPUs
- 📊 Implemented detailed performance reporting for audio generation

[GitHub Repository](https://github.com/Lex-au/Orpheus-FastAPI)

## Voice Demos

Listen to sample outputs with different voices and emotions:
- [Default Test Sample](https://lex-au.github.io/Orpheus-FastAPI/DefaultTest.mp3) - Standard neutral tone
- [Leah Happy Sample](https://lex-au.github.io/Orpheus-FastAPI/LeahHappy.mp3) - Cheerful, upbeat demo
- [Tara Sad Sample](https://lex-au.github.io/Orpheus-FastAPI/TaraSad.mp3) - Emotional, melancholic demo
- [Zac Contemplative Sample](https://lex-au.github.io/Orpheus-FastAPI/ZacContemplative.mp3) - Thoughtful, measured tone


## Features

- **OpenAI API Compatible**: Drop-in replacement for OpenAI's `/v1/audio/speech` endpoint
- **High Performance**: Optimized for RTX GPUs with parallel processing
- **Multiple Voices**: 8 different voice options with different characteristics
- **Emotion Tags**: Support for laughter, sighs, and other emotional expressions
- **Unlimited Audio Length**: Generate audio of any length through intelligent batching
- **Smooth Transitions**: Crossfaded audio segments for seamless listening experience

## Project Structure

```
Orpheus-FastAPI/
├── main.py                # FastAPI server and endpoints
├── requirements.txt      # Dependencies
└── tts_engine/           # Core TTS functionality
    ├── inference.py      # Token generation and API handling
    └── speechpipe.py     # Audio conversion pipeline
```

## Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: RTX series for best performance)
- Separate LLM inference server running the Orpheus model (e.g., LM Studio or llama.cpp server)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Lex-au/Orpheus-FastAPI.git
cd Orpheus-FastAPI
```

2. Create a Python virtual environment:
```bash
# Using venv (Python's built-in virtual environment)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n orpheus-tts python=3.10
conda activate orpheus-tts
```

3. Install PyTorch with CUDA support:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

4. Install other dependencies:
```bash
pip3 install -r requirements.txt
```


### Starting the Server

Run the FastAPI server:
```bash
python main.py
```

Or with specific host/port:
```bash
uvicorn main:app --host 0.0.0.0 --port 5005 --reload
```

Access:
- API documentation: http://localhost:5005/docs (or http://127.0.0.1:5005/docs)

![API Documentation](https://lex-au.github.io/Orpheus-FastAPI/docs.png)

## API Usage

### OpenAI-Compatible Endpoint

The server provides an OpenAI-compatible API endpoint at `/v1/audio/speech`:

```bash
curl http://localhost:5005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "orpheus",
    "input": "Hello world! This is a test of the Orpheus TTS system.",
    "voice": "tara",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output speech.wav
```

### Parameters

- `input` (required): The text to convert to speech
- `model` (optional): The model to use (default: "orpheus")
- `voice` (optional): Which voice to use (default: "tara")
- `response_format` (optional): Output format (currently only "wav" is supported)
- `speed` (optional): Speed factor (0.5 to 1.5, default: 1.0)

### Available Voices

- `tara`: Female, conversational, clear
- `leah`: Female, warm, gentle
- `jess`: Female, energetic, youthful
- `leo`: Male, authoritative, deep
- `dan`: Male, friendly, casual
- `mia`: Female, professional, articulate
- `zac`: Male, enthusiastic, dynamic
- `zoe`: Female, calm, soothing

### Emotion Tags

You can insert emotion tags into your text to add expressiveness:

- `<laugh>`: Add laughter
- `<sigh>`: Add a sigh
- `<chuckle>`: Add a chuckle
- `<cough>`: Add a cough sound
- `<sniffle>`: Add a sniffle sound
- `<groan>`: Add a groan
- `<yawn>`: Add a yawning sound
- `<gasp>`: Add a gasping sound

Example: "Well, that's interesting <laugh> I hadn't thought of that before."

## Technical Details

This server works as a frontend that connects to an external LLM inference server. It sends text prompts to the inference server, which generates tokens that are then converted to audio using the SNAC model. The system has been optimised for RTX 4090 GPUs with:

- Vectorised tensor operations
- Parallel processing with CUDA streams
- Efficient memory management
- Token and audio caching
- Optimised batch sizes

For best performance, adjust the API_URL in `tts_engine/inference.py` to point to your LLM inference server endpoint.

### Hardware Detection and Optimization

The system features intelligent hardware detection that automatically optimizes performance based on your hardware capabilities:

- **High-End GPU Mode** (dynamically detected based on capabilities):
  - Triggered by either: 16GB+ VRAM, compute capability 8.0+, or 12GB+ VRAM with 7.0+ compute capability
  - Advanced parallel processing with 4 workers
  - Optimized batch sizes (32 tokens)
  - High-throughput parallel file I/O
  - Full hardware details displayed (name, VRAM, compute capability)
  - GPU-specific optimizations automatically applied

- **Standard GPU Mode** (other CUDA-capable GPUs):
  - Efficient parallel processing
  - GPU-optimized parameters
  - CUDA acceleration where beneficial
  - Detailed GPU specifications

- **CPU Mode** (when no GPU is available):
  - Conservative processing with 2 workers
  - Optimized memory usage
  - Smaller batch sizes (16 tokens)
  - Sequential file I/O
  - Detailed CPU cores, threads, and RAM information

No manual configuration is needed - the system automatically detects hardware capabilities and adapts for optimal performance across different generations of GPUs and CPUs.

### Token Processing Optimization

The token processing system has been optimized with mathematically aligned parameters:
- Uses a context window of 49 tokens (7²)
- Processes in batches of 7 tokens (Orpheus model standard)
- This square relationship ensures complete token processing with no missed tokens
- Results in cleaner audio generation with proper token alignment
- Repetition penalty fixed at 1.1 for optimal quality generation (cannot be changed)

### Long Text Processing

The system features efficient batch processing for texts of any length:
- Automatically detects longer inputs (>1000 characters) 
- Splits text at logical points to create manageable chunks
- Processes each chunk independently for reliability
- Combines audio segments with smooth 50ms crossfades
- Intelligently stitches segments in-memory for consistent output
- Handles texts of unlimited length with no truncation
- Provides detailed progress reporting for each batch

**Note about long-form audio**: While the system now supports texts of unlimited length, there may be slight audio discontinuities between segments due to architectural constraints of the underlying model. The Orpheus model was designed for short to medium text segments, and our batching system works around this limitation by intelligently splitting and stitching content with minimal audible impact.

- `ORPHEUS_API_URL`: URL of the LLM inference API (tts_engine/inference.py)
- `ORPHEUS_API_TIMEOUT`: Timeout in seconds for API requests (default: 120)
- `ORPHEUS_MAX_TOKENS`: Maximum tokens to generate (default: 8192)
- `ORPHEUS_TEMPERATURE`: Temperature for generation (default: 0.6)
- `ORPHEUS_TOP_P`: Top-p sampling parameter (default: 0.9)
- `ORPHEUS_SAMPLE_RATE`: Audio sample rate in Hz (default: 24000)
- `ORPHEUS_PORT`: Web server port (default: 5005)
- `ORPHEUS_HOST`: Web server host (default: 0.0.0.0)

The system now supports loading environment variables from a `.env` file in the project root, making it easier to configure without modifying system-wide environment settings. See `.env.example` for a template.

Note: Repetition penalty is hardcoded to 1.1 and cannot be changed through environment variables as this is the only value that produces stable, high-quality output.

Make sure the `ORPHEUS_API_URL` points to your running inference server.

## Development

### Project Components

- **main.py**: FastAPI server that handles HTTP requests and serves the web UI
- **tts_engine/inference.py**: Handles token generation and API communication 
- **tts_engine/speechpipe.py**: Converts token sequences to audio using the SNAC model

### Adding New Voices

To add new voices, update the `AVAILABLE_VOICES` list in `tts_engine/inference.py`

## Using with llama.cpp

When running the Orpheus model with llama.cpp, use these parameters to ensure optimal performance:

```bash
./llama-server -m models/Orpheus-3b-FT-Q8_0.gguf \
  --ctx-size={{your ORPHEUS_MAX_TOKENS from .env}} \
  --n-predict={{your ORPHEUS_MAX_TOKENS from .env}} \
  --rope-scaling=linear
```

Important parameters:
- `--ctx-size`: Sets the context window size, should match your ORPHEUS_MAX_TOKENS setting
- `--n-predict`: Maximum tokens to generate, should match your ORPHEUS_MAX_TOKENS setting
- `--rope-scaling=linear`: Required for optimal positional encoding with the Orpheus model

For extended audio generation (books, long narrations), you may want to increase your token limits:
1. Set ORPHEUS_MAX_TOKENS to 32768 or higher in your .env file (or via the Web UI)
2. Increase ORPHEUS_API_TIMEOUT to 1800 for longer processing times
3. Use the same values in your llama.cpp parameters (if you're using llama.cpp)

## License

This project is licensed under the Apache License 2.0 - see the LICENSE.txt file for details.

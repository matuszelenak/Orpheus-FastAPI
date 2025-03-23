from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import time

# Try to enable torch.compile if PyTorch 2.0+ is available
TORCH_COMPILE_AVAILABLE = False
try:
    if hasattr(torch, 'compile'):
        TORCH_COMPILE_AVAILABLE = True
        print("PyTorch 2.0+ detected, torch.compile is available")
except:
    pass

# Try to enable CUDA graphs if available
CUDA_GRAPHS_AVAILABLE = False
try:
    if torch.cuda.is_available() and hasattr(torch.cuda, 'make_graphed_callables'):
        CUDA_GRAPHS_AVAILABLE = True
        print("CUDA graphs support is available")
except:
    pass

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

# Check if CUDA is available and set device accordingly
snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {snac_device}")
model = model.to(snac_device)

# Disable torch.compile as it requires Triton which isn't installed
# We'll use regular PyTorch optimization techniques instead
print("Using standard PyTorch optimizations (torch.compile disabled)")

# Prepare CUDA streams for parallel processing if available
cuda_stream = None
if snac_device == "cuda":
    cuda_stream = torch.cuda.Stream()
    print("Using CUDA stream for parallel processing")


def convert_to_audio(multiframe, count):
    """
    Optimized version of convert_to_audio that eliminates inefficient tensor operations
    and reduces CPU-GPU transfers for much faster inference on high-end GPUs.
    """
    if len(multiframe) < 7:
        return None

    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames * 7]

    # Pre-allocate tensors instead of incrementally building them
    codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=snac_device)
    codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=snac_device)
    codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=snac_device)

    # Use vectorized operations where possible
    frame_tensor = torch.tensor(frame, dtype=torch.int32, device=snac_device)

    # Direct indexing is much faster than concatenation in a loop
    for j in range(num_frames):
        idx = j * 7

        # Code 0 - single value per frame
        codes_0[j] = frame_tensor[idx]

        # Code 1 - two values per frame
        codes_1[j * 2] = frame_tensor[idx + 1]
        codes_1[j * 2 + 1] = frame_tensor[idx + 4]

        # Code 2 - four values per frame
        codes_2[j * 4] = frame_tensor[idx + 2]
        codes_2[j * 4 + 1] = frame_tensor[idx + 3]
        codes_2[j * 4 + 2] = frame_tensor[idx + 5]
        codes_2[j * 4 + 3] = frame_tensor[idx + 6]

    # Reshape codes into expected format
    codes = [
        codes_0.unsqueeze(0),
        codes_1.unsqueeze(0),
        codes_2.unsqueeze(0)
    ]

    # Check tokens are in valid range
    if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or
            torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or
            torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
        return None

    # Use CUDA stream for parallel processing if available
    stream_ctx = torch.cuda.stream(cuda_stream) if cuda_stream is not None else torch.no_grad()

    with stream_ctx, torch.inference_mode():
        # Decode the audio
        audio_hat = model.decode(codes)

        # Extract the relevant slice and efficiently convert to bytes
        # Keep data on GPU as long as possible
        audio_slice = audio_hat[:, :, 2048:4096]

        # Process on GPU if possible, with minimal data transfer
        if snac_device == "cuda":
            # Scale directly on GPU
            audio_int16_tensor = (audio_slice * 32767).to(torch.int16)
            # Only transfer the final result to CPU
            audio_bytes = audio_int16_tensor.cpu().numpy().tobytes()
        else:
            # For non-CUDA devices, fall back to the original approach
            detached_audio = audio_slice.detach().cpu()
            audio_np = detached_audio.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

    return audio_bytes


def turn_token_into_id(token_string, index):
    """Optimized token-to-id conversion with early returns and minimal string operations"""
    token_string = token_string.strip()

    # Early return for obvious mismatches
    if "<custom_token_" not in token_string:
        return None

    # Find the last token in the string
    last_token_start = token_string.rfind("<custom_token_")
    if last_token_start == -1:
        return None

    # Check if the token ends properly
    if not token_string.endswith(">"):
        return None

    try:
        # Extract and convert the number directly
        number_str = token_string[last_token_start + 14:-1]
        return int(number_str) - 10 - ((index % 7) * 4096)
    except (ValueError, IndexError):
        return None


# Cache for frequently processed tokens to avoid redundant computation
token_cache = {}
MAX_CACHE_SIZE = 1000  # Limit cache size to prevent memory bloat


async def tokens_decoder(token_gen):
    """Optimized token decoder with reliable end-of-buffer handling for complete audio generation"""
    buffer = []
    count = 0
    # Use a smaller minimum frame requirement to allow more flexible processing
    min_frames_required = 28  # Lower requirement (4 chunks of 7 tokens)
    ideal_frames = 49  # Ideal standard frame size (7×7 window)
    process_every_n = 7  # Process every 7 tokens (standard for Orpheus model)

    start_time = time.time()
    token_count = 0
    last_log_time = start_time

    async for token_sim in token_gen:
        token_count += 1

        # Check cache first to avoid redundant computation
        cache_key = (token_sim, count % 7)
        if cache_key in token_cache:
            token = token_cache[cache_key]
        else:
            token = turn_token_into_id(token_sim, count)
            # Add to cache if valid token
            if token is not None and len(token_cache) < MAX_CACHE_SIZE:
                token_cache[cache_key] = token

        if token is not None and token > 0:
            buffer.append(token)
            count += 1

            # Log throughput periodically
            current_time = time.time()
            if current_time - last_log_time > 5.0:  # Every 5 seconds
                elapsed = current_time - last_log_time
                if elapsed > 0:
                    recent_tokens = token_count
                    tokens_per_sec = recent_tokens / elapsed
                    print(f"Token processing rate: {tokens_per_sec:.1f} tokens/second")
                last_log_time = current_time
                token_count = 0

            # Process standard batches when we have enough tokens
            if count % process_every_n == 0:
                # Best case: we have enough for the ideal frame size
                if len(buffer) >= ideal_frames:
                    buffer_to_proc = buffer[-ideal_frames:]
                # Fallback: we have enough for the minimum requirement
                elif len(buffer) >= min_frames_required:
                    buffer_to_proc = buffer[-min_frames_required:]
                # For the first few frames, we may not have enough yet
                else:
                    continue

                # Debug output to help diagnose issues
                if count % 28 == 0:
                    print(f"Processing buffer with {len(buffer_to_proc)} tokens, total collected: {len(buffer)}")

                # Process the tokens
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    yield audio_samples

    # CRITICAL: End-of-generation handling - process all remaining frames
    # Process remaining complete frames (ideal size)
    if len(buffer) >= ideal_frames:
        buffer_to_proc = buffer[-ideal_frames:]
        audio_samples = convert_to_audio(buffer_to_proc, count)
        if audio_samples is not None:
            yield audio_samples

    # Process any additional complete frames (minimum size)
    elif len(buffer) >= min_frames_required:
        buffer_to_proc = buffer[-min_frames_required:]
        audio_samples = convert_to_audio(buffer_to_proc, count)
        if audio_samples is not None:
            yield audio_samples

    # Final special case: even if we don't have minimum frames, try to process
    # what we have by padding with silence tokens that won't affect the audio
    elif len(buffer) >= process_every_n:
        # Pad to minimum frame requirement with copies of the final token
        # This is more continuous than using unrelated tokens from the beginning
        last_token = buffer[-1]
        padding_needed = min_frames_required - len(buffer)

        # Create a padding array of copies of the last token
        # This maintains continuity much better than circular buffering
        padding = [last_token] * padding_needed
        padded_buffer = buffer + padding

        print(f"Processing final partial frame: {len(buffer)} tokens + {padding_needed} repeated-token padding")
        audio_samples = convert_to_audio(padded_buffer, count)
        if audio_samples is not None:
            yield audio_samples


# ------------------ Synchronous Tokens Decoder Wrapper ------------------ #
def tokens_decoder_sync(syn_token_gen):
    """Optimized synchronous decoder with larger queue and parallel processing"""
    # Use a larger queue for RTX 4090 to maximize GPU utilization
    max_queue_size = 32 if snac_device == "cuda" else 8
    audio_queue = queue.Queue(maxsize=max_queue_size)

    # Collect tokens in batches for higher throughput
    batch_size = 16 if snac_device == "cuda" else 4

    # Convert the synchronous token generator into an async generator with batching
    async def async_token_gen():
        token_batch = []
        for token in syn_token_gen:
            token_batch.append(token)
            # Process in batches for efficiency
            if len(token_batch) >= batch_size:
                for t in token_batch:
                    yield t
                token_batch = []
        # Process any remaining tokens
        for t in token_batch:
            yield t

    async def async_producer():
        # Start timer for performance logging
        start_time = time.time()
        chunk_count = 0

        try:
            # Process audio chunks from the token decoder
            async for audio_chunk in tokens_decoder(async_token_gen()):
                if audio_chunk:  # Validate audio chunk before adding to queue
                    audio_queue.put(audio_chunk)
                    chunk_count += 1

                    # Log performance stats periodically
                    if chunk_count % 10 == 0:
                        elapsed = time.time() - start_time
                        print(
                            f"Generated {chunk_count} chunks in {elapsed:.2f}s ({chunk_count / elapsed:.2f} chunks/sec)")
        except Exception as e:
            print(f"Error in audio producer: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Signal completion
            print("Audio producer completed - finalizing all chunks")
            audio_queue.put(None)  # Sentinel

    def run_async():
        asyncio.run(async_producer())

    # Use a higher priority thread for RTX 4090 to ensure it stays fed with work
    thread = threading.Thread(target=run_async)
    thread.daemon = True  # Allow the thread to be terminated when the main thread exits
    thread.start()

    # Use larger buffer for final audio assembly
    buffer_size = 5
    audio_buffer = []

    while True:
        audio = audio_queue.get()
        if audio is None:
            break

        audio_buffer.append(audio)
        # Yield buffered audio chunks for smoother playback
        if len(audio_buffer) >= buffer_size:
            for chunk in audio_buffer:
                yield chunk
            audio_buffer = []

    # Yield any remaining audio in the buffer
    for chunk in audio_buffer:
        yield chunk

    thread.join()

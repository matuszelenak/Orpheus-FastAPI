import numpy as np
import torch
from snac import SNAC

from tts_engine.log import get_logger

logger = get_logger(__name__)

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
logger.debug(f"Using device: {snac_device}")
model = model.to(snac_device)


cuda_stream = None
if snac_device == "cuda":
    cuda_stream = torch.cuda.Stream()
    logger.debug("Using CUDA stream for parallel processing")


def convert_to_audio(multiframe):
    """
    Optimized version of convert_to_audio that eliminates inefficient tensor operations
    and reduces CPU-GPU transfers for much faster inference on high-end GPUs.
    """
    if len(multiframe) < 7:
        logger.debug("Not enough frames to convert to audio")
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
        logger.debug("Invalid tokens")
        print(codes)
        logger.debug(str(multiframe))
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

import asyncio
import websockets
import json
import base64
import os
import subprocess
import wave

# Target URL pointing directly to Ticket 1 on your live Render backend
BACKEND_WS_URL = "wss://notollm-zendesk-backend.onrender.com/stream/twilio/?ticket_id=1&api_key=cust_live_123abc"

# Pure Python linear PCM 16-bit to G.711 u-law converter (Zero external dependencies)
BIAS = 0x84
CLIP = 32635
EXPONENT_LUT = [
    0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
]

def pcm16_to_ulaw_byte(sample: int) -> int:
    sign = (sample >> 8) & 0x80
    if sign != 0:
        sample = -sample
    if sample > CLIP:
        sample = CLIP
    sample += BIAS
    exponent = EXPONENT_LUT[(sample >> 7) & 0xFF]
    mantissa = (sample >> (exponent + 3)) & 0x0F
    ulawbyte = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return ulawbyte

def get_sample_speech_mulaw() -> bytes:
    """Uses macOS built-in speech synthesis to generate real-time test audio."""
    print("Synthesizing test speech using macOS 'say' command...")
    
    # You can change this text to test different Gemini RAG responses
    text = "Hello, I am calling about a recent purchase. What is your return policy on damaged items?"
    filepath = "mock_speech.wav"
    
    # Trigger macOS 'say' command to generate an 8000Hz 16-bit PCM WAV file
    subprocess.run([
        "say",
        "-o", filepath,
        "--data-format=LEI16@8000",
        text
    ], check=True)

    # Read the generated WAV file
    with wave.open(filepath, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        raw_pcm = wav_file.readframes(wav_file.getnframes())
        
    # Clean up the local file
    if os.path.exists(filepath):
        os.remove(filepath)

    # Convert 16-bit signed PCM frames to 8-bit mu-law
    ulaw_bytes = bytearray()
    step = 2 * n_channels
    for i in range(0, len(raw_pcm), step):
        sample = int.from_bytes(raw_pcm[i:i+2], byteorder='little', signed=True)
        ulaw_bytes.append(pcm16_to_ulaw_byte(sample))

    return bytes(ulaw_bytes)

async def run_mock_call():
    audio_data = get_sample_speech_mulaw()
    print(f"Audio ready: {len(audio_data)} bytes (~{len(audio_data)/8000:.1f} seconds).")

    print(f"Connecting to: {BACKEND_WS_URL}")
    async with websockets.connect(BACKEND_WS_URL) as ws:
        print("Connected to Render WebSocket backend.")

        # 1. Send Twilio start event
        start_payload = {
            "event": "start",
            "sequenceNumber": "1",
            "start": {
                "streamSid": "MZ_test_simulation_stream",
                "callSid": "CA_mock_call_12345",
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}
            }
        }
        await ws.send(json.dumps(start_payload))

        # 2. Stream in 20ms chunks (160 bytes of 8000Hz mu-law = 20ms)
        CHUNK_SIZE = 160
        print("Streaming speech audio frames in real-time...")

        for i in range(0, len(audio_data), CHUNK_SIZE):
            chunk = audio_data[i:i + CHUNK_SIZE]
            payload = base64.b64encode(chunk).decode("utf-8")
            media_msg = {
                "event": "media",
                "sequenceNumber": str(i // CHUNK_SIZE + 2),
                "media": {"payload": payload}
            }
            await ws.send(json.dumps(media_msg))
            await asyncio.sleep(0.02)  # 20ms real-time pacing

        # 3. Send stop event
        stop_payload = {"event": "stop", "sequenceNumber": "9999"}
        await ws.send(json.dumps(stop_payload))
        print("Audio stream finished. Awaiting Gemini and Firebase response...")

        await asyncio.sleep(4)

if __name__ == "__main__":
    asyncio.run(run_mock_call())
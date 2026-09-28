import asyncio
import websockets
import json
import base64
import requests
import io
import wave
import audioop

# 1. Target URL pointing directly to Ticket 1 on your live Render backend
BACKEND_WS_URL = "wss://notollm-zendesk-backend.onrender.com/stream/twilio/?ticket_id=1&api_key=cust_live_123abc"

def generate_sample_mulaw() -> bytes:
    """
    Fetches a brief speech audio sample and converts it to 8000Hz mono μ-law 
    (matching standard Twilio / Telephony streams).
    """
    print("Preparing test audio sample...")
    # Public domain sample utterance: "Can I get a refund for my order?"
    # Using a reliable raw wav fixture or generating standard telephone audio
    url = "https://raw.githubusercontent.com/voxserv/audio-samples/master/speech/speech-male-16k.wav"
    r = requests.get(url)
    
    with wave.open(io.BytesIO(r.content), 'rb') as wav_in:
        n_channels = wav_in.getnchannels()
        sampwidth = wav_in.getsampwidth()
        framerate = wav_in.getframerate()
        frames = wav_in.readframes(wav_in.getnframes())

    # Downmix to mono if stereo
    if n_channels == 2:
        frames = audioop.tomono(frames, sampwidth, 0.5, 0.5)

    # Resample to 8000 Hz if needed
    if framerate != 8000:
        frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 8000, None)

    # Convert linear PCM 16-bit to 8-bit μ-law (standard G.711u / Twilio payload)
    mulaw_data = audioop.lin2ulaw(frames, sampwidth)
    return mulaw_data


async def run_mock_call():
    audio_data = generate_sample_mulaw()
    print(f"Total audio ready: {len(audio_data)} bytes (~{len(audio_data) / 8000:.1f} seconds).")

    print(f"Connecting to: {BACKEND_WS_URL}")
    async with websockets.connect(BACKEND_WS_URL) as ws:
        print("Connected to Render WebSocket backend.")

        # Step 1: Send Twilio 'start' event
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

        # Step 2: Stream in 20ms chunks (160 bytes of μ-law at 8000 samples/sec = 20ms)
        CHUNK_SIZE = 160 
        print("Streaming speech audio frames in real-time...")

        for i in range(0, len(audio_data), CHUNK_SIZE):
            chunk = audio_data[i:i + CHUNK_SIZE]
            payload = base64.b64encode(chunk).decode("utf-8")
            
            media_msg = {
                "event": "media",
                "sequenceNumber": str(i // CHUNK_SIZE + 2),
                "media": {
                    "payload": payload
                }
            }
            await ws.send(json.dumps(media_msg))
            await asyncio.sleep(0.02) # Paced to simulate live caller speech

        # Step 3: Send 'stop' event
        stop_payload = {"event": "stop", "sequenceNumber": "9999"}
        await ws.send(json.dumps(stop_payload))
        print("Call finished. Audio stream completed.")

        # Keep open briefly to allow final speech transcripts to settle
        await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(run_mock_call())
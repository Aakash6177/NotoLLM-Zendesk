import os
import json
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore

# 1. Initialize Firebase Admin securely for both Local & Render
# Render mounts Secret Files in the /etc/secrets/ directory for Docker environments
firebase_key_path = "/etc/secrets/firebase-adminsdk.json" 

if not os.path.exists(firebase_key_path):
    # Fallback to local directory for local development
    firebase_key_path = "firebase-adminsdk.json"

try:
    cred = credentials.Certificate(firebase_key_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully.")
except Exception as e:
    print(f"Failed to initialize Firebase: {e}")
    db = None

# 2. Initialize FastAPI, Deepgram & Gemini
app = FastAPI()
deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Using Flash because it is optimized for high-frequency, low-latency tasks
gemini_model = genai.GenerativeModel('gemini-1.5-flash')


async def generate_and_push_suggestion(transcript: str, ticket_id: str):
    """The RAG Engine: Runs Gemini and pushes to Firebase"""
    if not db:
        print("Skipping Firebase push: Firebase is not initialized.")
        return

    prompt = f"""You are an agent assist AI. The customer just said: '{transcript}'
    Based on our company policy, generate a very brief, 1-2 sentence helpful recommendation for the support agent."""

    try:
        # Use Gemini's async generation to avoid blocking the websocket stream
        response = await gemini_model.generate_content_async(prompt)
        suggestion_text = response.text

        # Push to Firebase State Store
        doc_ref = db.collection("tickets").document(ticket_id).collection("suggestions").document()
        doc_ref.set({
            "title": "✨ AI Recommendation",
            "text": suggestion_text.strip(),
            "source": "Knowledge Base (Auto-generated)",
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        print(f"Pushed Gemini suggestion to Firebase for ticket {ticket_id}")
        
    except Exception as e:
        print(f"Gemini or Firebase Error: {e}")


# 3. Health Check Endpoint for Render Deployments
@app.get("/")
def health_check():
    """Render pings the root URL to verify the container is alive."""
    return {"status": "healthy", "service": "ai-agent-assist"}


# 4. The WebSocket Audio Stream
@app.websocket("/media/{ticket_id}")
async def media_stream(websocket: WebSocket, ticket_id: str):
    """The Audio Backend: Handles the live call stream"""
    await websocket.accept()
    print(f"WebSocket connected for ticket: {ticket_id}")

    dg_connection = None
    try:
        # Connect to Deepgram Live Transcription
        dg_connection = deepgram.listen.websocket.v("1")
        
        def on_message(self, result, **kwargs):
            # Extract transcript from Deepgram payload
            if 'channel' in result:
                transcript = result.channel.alternatives[0].transcript
                if transcript and result.speech_final:
                    print(f"Final Transcript: {transcript}")
                    # Fire and forget the LLM/Firebase push so we don't block audio processing
                    asyncio.create_task(generate_and_push_suggestion(transcript, ticket_id))

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        
        # Configure Deepgram for Twilio's standard audio format (mulaw, 8000Hz)
        options = LiveOptions(model="nova-2", encoding="mulaw", sample_rate=8000)
        dg_connection.start(options)

        # Continuous loop to receive audio from CTI and forward to Deepgram
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            if msg["event"] == "media":
                # Decode the base64 audio and push it to Deepgram
                audio = base64.b64decode(msg["media"]["payload"])
                dg_connection.send(audio)
            elif msg["event"] == "stop":
                print(f"Received stop event for ticket {ticket_id}")
                break

    except WebSocketDisconnect:
        print(f"Client disconnected for ticket {ticket_id}")
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        # Always clean up the Deepgram connection when the call drops
        if dg_connection:
            dg_connection.finish()
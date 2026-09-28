import os
import json
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, status
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

# Mock database of valid customer API keys
VALID_API_KEYS = {"cust_live_123abc", "cust_live_456def"}

async def authenticate_connection(api_key: str = Query(None)):
    """Validates the customer's API key passed in the WebSocket URL query."""
    if api_key not in VALID_API_KEYS:
        # In WebSockets, we cannot return standard HTTP 401s easily after accept,
        # so we validate it before fully upgrading the connection.
        return False
    return True

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
@app.websocket("/stream/{provider}/{ticket_id}")
async def universal_media_stream(
    websocket: WebSocket, 
    provider: str, 
    ticket_id: str, 
    api_key: str = Query(None)
):
    """
    The BYOT Ingress Router:
    URL Format: wss://your-backend.com/stream/twilio/12345?api_key=cust_live_123abc
    """
    
    # 1. Enforce Authentication
    is_valid = await authenticate_connection(api_key)
    if not is_valid:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        print(f"Rejected connection for ticket {ticket_id}: Invalid API Key")
        return
        
    await websocket.accept()
    print(f"[{provider.upper()}] Stream connected for ticket: {ticket_id}")

    dg_connection = None
    try:
        dg_connection = deepgram.listen.websocket.v("1")
        
        def on_message(self, result, **kwargs):
            if 'channel' in result:
                transcript = result.channel.alternatives[0].transcript
                if transcript and result.speech_final:
                    print(f"[{ticket_id}] Final Transcript: {transcript}")
                    asyncio.create_task(generate_and_push_suggestion(transcript, ticket_id))

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        
        # In a full production app, you might map the encoding based on the provider 
        # (e.g., Twilio is mulaw, others might be linear16)
        options = LiveOptions(model="nova-2", encoding="mulaw", sample_rate=8000)
        dg_connection.start(options)

        # 2. The Normalization Loop
        while True:
            # receive() gets either text (JSON) or bytes based on what the client sends
            message = await websocket.receive()
            
            if provider.lower() == "twilio":
                # Twilio sends base64 encoded audio wrapped in JSON strings
                if "text" in message:
                    msg = json.loads(message["text"])
                    if msg["event"] == "media":
                        audio = base64.b64decode(msg["media"]["payload"])
                        dg_connection.send(audio)
                    elif msg["event"] == "stop":
                        break
                        
            elif provider.lower() in ["raw", "genesys", "amazon"]:
                # Enterprise CTI systems often stream raw binary frames natively
                if "bytes" in message:
                    dg_connection.send(message["bytes"])
                elif "text" in message and message["text"] == "stop":
                    break

    except WebSocketDisconnect:
        print(f"Client disconnected for ticket {ticket_id}")
    except Exception as e:
        print(f"Stream processing error: {e}")
    finally:
        if dg_connection:
            dg_connection.finish()
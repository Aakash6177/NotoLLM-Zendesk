import os
import json
import base64
import httpx
import urllib.parse
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, status
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
from google import genai
from google.genai import types
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

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# Using Flash because it is optimized for high-frequency, low-latency tasks
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

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
        response = await gemini_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        suggestion_text = response.text or ""

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
# Replace Section 4 in your main.py with this:

@app.websocket("/stream/{provider}/")
async def universal_media_stream(
    websocket: WebSocket, 
    provider: str, 
    ticket_id: str = Query(None),       
    caller_phone: str = Query(None),    
    api_key: str = Query(None)
):
    # 1. Enforce Authentication
    is_valid = await authenticate_connection(api_key)
    if not is_valid:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        print("Rejected connection: Invalid API Key")
        return
        
    await websocket.accept()

    # 2. Resolve Target Ticket ID
    target_ticket_id = ticket_id
    if not target_ticket_id and caller_phone:
        target_ticket_id = await get_ticket_id_by_phone(caller_phone)
    elif not target_ticket_id:
        target_ticket_id = "1" 

    print(f"[{provider.upper()}] Stream connected and mapped to Ticket ID: {target_ticket_id}")

    # Capture the main thread's asyncio loop to safely execute Gemini later
    loop = asyncio.get_running_loop()
    dg_connection = None
    
    try:
        dg_connection = deepgram.listen.live.v("1")
        
        def on_message(self, result, **kwargs):
            if hasattr(result, 'channel') and result.channel:
                transcript = result.channel.alternatives[0].transcript
                # Use is_final instead of speech_final to catch abrupt clip endings
                if transcript and result.is_final:
                    print(f"[{target_ticket_id}] Final Transcript: {transcript}")
                    
                    # Safely dispatch the RAG Engine task back to the main thread
                    asyncio.run_coroutine_threadsafe(
                        generate_and_push_suggestion(transcript, target_ticket_id), 
                        loop
                    )

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        options = LiveOptions(model="nova-2", encoding="mulaw", sample_rate=8000)
        dg_connection.start(options)

        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                print(f"Client disconnected for ticket {target_ticket_id}")
                break
            
            if provider.lower() == "twilio":
                if "text" in message:
                    msg = json.loads(message["text"])
                    if msg["event"] == "media":
                        audio = base64.b64decode(msg["media"]["payload"])
                        dg_connection.send(audio)
                    elif msg["event"] == "stop":
                        print(f"Stop event received for ticket {target_ticket_id}")
                        # Force Deepgram to flush its final transcript buffer, 
                        # but DO NOT break out of the loop yet. Allow the client 
                        # to naturally disconnect so we can catch the final text.
                        dg_connection.finish()
                        
            elif provider.lower() in ["raw", "genesys", "amazon"]:
                if "bytes" in message:
                    dg_connection.send(message["bytes"])
                elif "text" in message and message["text"] == "stop":
                    dg_connection.finish()

    except WebSocketDisconnect:
        print(f"Client stream disconnected for ticket {target_ticket_id}")
    except Exception as e:
        print(f"Stream processing error: {e}")
    finally:
        # Failsafe cleanup
        if dg_connection:
            try:
                dg_connection.finish()
            except Exception:
                pass

# 1. Fetch Ticket ID from Zendesk
async def get_ticket_id_by_phone(caller_phone: str) -> str:
    """Queries the Zendesk Search API to find an open ticket for the calling phone number."""
    subdomain = os.getenv("ZENDESK_SUBDOMAIN")
    email = os.getenv("ZENDESK_EMAIL")
    token = os.getenv("ZENDESK_API_TOKEN")
    
    if not all([subdomain, email, token]):
        print("Zendesk credentials missing. Falling back to default ticket ID.")
        return "DEFAULT_TICKET"

    # Zendesk requires URL encoding for phone numbers (e.g., +14155551212 becomes %2B14155551212)
    encoded_phone = urllib.parse.quote(caller_phone)
    
    # Query: Find open tickets where the requester matches this phone number
    search_query = f"type:ticket requester:{encoded_phone} status<solved"
    url = f"https://{subdomain}.zendesk.com/api/v2/search.json?query={search_query}"
    
    auth = (f"{email}/token", token)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, auth=auth)
            response.raise_for_status()
            data = response.json()
            
            # If a ticket is found, return the ID of the most recent one
            if data.get("results") and len(data["results"]) > 0:
                ticket_id = str(data["results"][0]["id"])
                print(f"Found active Zendesk Ticket: {ticket_id} for phone: {caller_phone}")
                return ticket_id
            else:
                print(f"No open ticket found for phone {caller_phone}. Generating fallback ID.")
                return f"UNMATCHED_{caller_phone.replace('+', '')}"
                
        except Exception as e:
            print(f"Zendesk API Search Error: {e}")
            return "ERROR_TICKET"
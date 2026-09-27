import os
import json
import base64
from fastapi import FastAPI, WebSocket
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore

# 1. Initialize Firebase Admin
cred = credentials.Certificate("firebase-adminsdk.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# 2. Initialize API Clients (Deepgram & Gemini)
app = FastAPI()
deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Flash is the fastest model for real-time RAG
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

async def generate_and_push_suggestion(transcript: str, ticket_id: str):
    """The RAG Engine: Runs Gemini and pushes to Firebase"""
    
    prompt = f"""You are an agent assist AI. The customer just said: '{transcript}'
    Based on our return policy, generate a short, helpful recommendation for the agent."""

    try:
        # Use Gemini's async generation to avoid blocking the websocket
        response = await gemini_model.generate_content_async(prompt)
        suggestion_text = response.text

        # 3. Push to Firebase State Store
        doc_ref = db.collection("tickets").document(ticket_id).collection("suggestions").document()
        doc_ref.set({
            "title": "✨ AI Recommendation",
            "text": suggestion_text,
            "source": "Knowledge Base (Auto-generated)",
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        print(f"Pushed Gemini suggestion to Firebase for ticket {ticket_id}")
        
    except Exception as e:
        print(f"Gemini API error: {e}")

@app.websocket("/media/{ticket_id}")
async def media_stream(websocket: WebSocket, ticket_id: str):
    """The Audio Backend: Handles the live call stream"""
    await websocket.accept()

    try:
        # Connect to Deepgram
        dg_connection = deepgram.listen.websocket.v("1")
        
        def on_message(self, result, **kwargs):
            transcript = result.channel.alternatives[0].transcript
            if transcript and result.speech_final:
                # Fire and forget the LLM/Firebase push so we don't block the audio stream
                import asyncio
                asyncio.create_task(generate_and_push_suggestion(transcript, ticket_id))

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        
        options = LiveOptions(model="nova-2", encoding="mulaw", sample_rate=8000)
        dg_connection.start(options)

        # Receive audio from CTI and forward to Deepgram
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            if msg["event"] == "media":
                audio = base64.b64decode(msg["media"]["payload"])
                dg_connection.send(audio)
            elif msg["event"] == "stop":
                break

    except Exception as e:
        print(f"Connection closed: {e}")
    finally:
        dg_connection.finish()
import os
import json
import base64
import httpx
import urllib.parse
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, status
from fastapi.middleware.cors import CORSMiddleware
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
from google import genai
from google.genai.errors import APIError
import firebase_admin
from firebase_admin import credentials, firestore
from pydantic import BaseModel

class AgentQuery(BaseModel):
    query: str
    ticket_id: str
    requester_id: str

# 1. Initialize Firebase Admin securely for both Local & Render
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

# Add CORS Middleware so the Zendesk frontend can call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from Zendesk iframe domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

VALID_API_KEYS = {"cust_live_123abc", "cust_live_456def"}

async def authenticate_connection(api_key: str = Query(None)):
    """Validates the customer's API key passed in the WebSocket URL query."""
    if api_key not in VALID_API_KEYS:
        return False
    return True

async def search_bookstack(query: str) -> str:
    """Searches BookStack KB on localhost / remote and returns markdown context."""
    domain = os.getenv("BOOKSTACK_DOMAIN", "localhost:8080")
    token_id = os.getenv("BOOKSTACK_TOKEN_ID")
    token_secret = os.getenv("BOOKSTACK_TOKEN_SECRET")
    
    if not all([domain, token_id, token_secret]):
        print("BookStack credentials or domain missing.")
        return ""
        
    encoded_query = urllib.parse.quote(query)
    
    # 1. Determine HTTP vs HTTPS
    protocol = "http" if any(h in domain for h in ["localhost", "127.0.0.1", "host.docker.internal"]) else "https"
    
    search_url = f"{protocol}://{domain}/api/search?query={encoded_query}"
    headers = {
        "Authorization": f"Token {token_id}:{token_secret}"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            # 2. Search for matching pages
            response = await client.get(search_url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            results = [item for item in data.get("data", []) if item.get("type") == "page"]
            if not results:
                return ""
                
            # 3. Pull top 2 articles and their markdown body
            context_parts = []
            for article in results[:2]:
                title = article.get("name", "")
                page_id = article.get("id")
                
                page_url = f"{protocol}://{domain}/api/pages/{page_id}"
                page_resp = await client.get(page_url, headers=headers)
                
                if page_resp.status_code == 200:
                    page_data = page_resp.json()
                    body = page_data.get("markdown", "") or page_data.get("preview_html", "")
                    context_parts.append(f"Source: {title}\nContent: {body[:2000]}")
                
            return "\n\n".join(context_parts)
            
        except Exception as e:
            print(f"Error fetching from BookStack: {e}")
            return ""

async def handle_custom_agent_query(query: str, ticket_id: str):
    print(f"Processing query for ticket {ticket_id}: {query}")
    
    # Swap out search_help_center with search_bookstack
    kb_context = await search_bookstack(query)
    
    prompt = f"""You are an internal AI assistant for a customer support agent. The agent asked: "{query}"

Company Knowledge Base (BookStack):
{kb_context if kb_context else "No relevant articles found."}

Instructions:
1. Answer the agent's question clearly and concisely.
2. Ground your answer STRICTLY in the Company Knowledge Base text provided above.
3. If the answer is not in the knowledge base, say "I cannot find this information in the Knowledge Base."
4. Always cite the article title at the end of your answer like this:."""

    try:
        response = await gemini_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        ai_text = response.text.strip()
        
        if db:
            doc_ref = db.collection("tickets").document(ticket_id).collection("suggestions").document()
            doc_ref.set({
                "title": "✨ AI Answer",
                "text": ai_text,
                "source": "BookStack KB",
                "timestamp": firestore.SERVER_TIMESTAMP
            })
            print(f"Pushed answer to ticket {ticket_id}")
    except Exception as e:
        print(f"Error generating or saving answer: {e}")

async def generate_and_push_suggestion(transcript: str, ticket_id: str):
    """The Live Audio RAG Engine: Runs Gemini on live caller transcripts."""
    if not db:
        print("Skipping Firebase push: Firebase is not initialized.")
        return

    kb_context = await search_help_center(transcript)

    prompt = f"""You are an agent assist AI. The customer just said: '{transcript}'
    
    Company Knowledge Base:
    {kb_context if kb_context else "No relevant knowledge base articles found."}
    
    Instructions:
    1. Generate a very brief, 1-2 sentence helpful recommendation for the support agent.
    2. Ground your recommendation strictly in the Company Knowledge Base if available.
    3. Cite the article title in brackets at the end if used, e.g. [Source: Article Title]."""

    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            response = await gemini_client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            suggestion_text = response.text or ""

            doc_ref = db.collection("tickets").document(ticket_id).collection("suggestions").document()
            doc_ref.set({
                "title": "✨ AI Recommendation",
                "text": suggestion_text.strip(),
                "source": "Knowledge Base (Auto-generated)" if kb_context else "General Best Practice",
                "timestamp": firestore.SERVER_TIMESTAMP
            })
            print(f"Pushed Gemini suggestion to Firebase for ticket {ticket_id}")
            return
            
        except APIError as e:
            if e.code == 503:
                print(f"Gemini 503 Unavailable (Attempt {attempt + 1}/{max_retries}). Retrying...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2 ** attempt))
                else:
                    print(f"Failed to generate suggestion after {max_retries} attempts due to high demand.")
            else:
                print(f"Gemini API Error: {e}")
                break
        except Exception as e:
            print(f"Unexpected Error: {e}")
            break

# 3. HTTP Endpoints
@app.get("/")
def health_check():
    """Render pings the root URL to verify the container is alive."""
    return {"status": "healthy", "service": "ai-agent-assist"}

@app.post("/ask")
async def process_agent_query(payload: AgentQuery):
    # Run the RAG pipeline in the background so we don't timeout the frontend
    asyncio.create_task(handle_custom_agent_query(payload.query, payload.ticket_id))
    return {"status": "processing"}

# 4. The WebSocket Audio Stream
@app.websocket("/stream/{provider}/")
async def universal_media_stream(
    websocket: WebSocket, 
    provider: str, 
    ticket_id: str = Query(None),       
    caller_phone: str = Query(None),    
    api_key: str = Query(None)
):
    is_valid = await authenticate_connection(api_key)
    if not is_valid:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        print("Rejected connection: Invalid API Key")
        return
        
    await websocket.accept()

    target_ticket_id = ticket_id
    if not target_ticket_id and caller_phone:
        target_ticket_id = await get_ticket_id_by_phone(caller_phone)
    elif not target_ticket_id:
        target_ticket_id = "1" 

    print(f"[{provider.upper()}] Stream connected and mapped to Ticket ID: {target_ticket_id}")

    loop = asyncio.get_running_loop()
    dg_connection = None
    
    try:
        dg_connection = deepgram.listen.live.v("1")
        
        def on_message(self, result, **kwargs):
            if hasattr(result, 'channel') and result.channel:
                transcript = result.channel.alternatives[0].transcript
                if transcript and result.is_final:
                    print(f"[{target_ticket_id}] Final Transcript: {transcript}")
                    
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
        if dg_connection:
            try:
                dg_connection.finish()
            except Exception:
                pass

# 5. Fetch Ticket ID from Zendesk
async def get_ticket_id_by_phone(caller_phone: str) -> str:
    """Queries the Zendesk Search API to find an open ticket for the calling phone number."""
    subdomain = os.getenv("ZENDESK_SUBDOMAIN")
    email = os.getenv("ZENDESK_EMAIL")
    token = os.getenv("ZENDESK_API_TOKEN")
    
    if not all([subdomain, email, token]):
        print("Zendesk credentials missing. Falling back to default ticket ID.")
        return "DEFAULT_TICKET"

    encoded_phone = urllib.parse.quote(caller_phone)
    search_query = f"type:ticket requester:{encoded_phone} status<solved"
    url = f"https://{subdomain}.zendesk.com/api/v2/search.json?query={search_query}"
    
    auth = (f"{email}/token", token)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, auth=auth)
            response.raise_for_status()
            data = response.json()
            
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
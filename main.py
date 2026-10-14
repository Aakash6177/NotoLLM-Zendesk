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
from typing import Optional, Union
import re
from bs4 import BeautifulSoup

class AgentQuery(BaseModel):
    query: str
    ticket_id: str
    requester_id: str

class AgentQuery(BaseModel):
    query: str
    ticket_id: Union[str, int]
    requester_id: Optional[Union[str, int]] = None

STOP = {"what","is","the","a","an","for","to","how","do","i","we","of","and","in","on","are",
        "about","should","can","my","our","does","this","that","matter","if","when","who"}

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

def bookstack_base() -> str:
    raw = (os.getenv("BOOKSTACK_URL") or os.getenv("BOOKSTACK_DOMAIN") or "").strip().rstrip("/")
    if not raw:
        return ""
    if raw.startswith(("http://", "https://")):
        return raw
    local = any(h in raw for h in ("localhost", "127.0.0.1", "host.docker.internal"))
    return f"{'http' if local else 'https'}://{raw}"

def html_to_text(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text(" ", strip=True)

async def search_bookstack(query: str) -> tuple[str, list[dict]]:
    """Returns (context_text, sources) where sources = [{'title','url'}]."""
    base = bookstack_base()
    token_id = os.getenv("BOOKSTACK_TOKEN_ID")
    token_secret = os.getenv("BOOKSTACK_TOKEN_SECRET")

    missing = [n for n, v in [("BOOKSTACK_URL", base),
                              ("BOOKSTACK_TOKEN_ID", token_id),
                              ("BOOKSTACK_TOKEN_SECRET", token_secret)] if not v]
    if missing:
        print(f"BookStack missing env vars: {missing}")
        return "", []

    headers = {"Authorization": f"Token {token_id}:{token_secret}"}
    keywords = " ".join(w for w in re.findall(r"[a-z0-9]+", query.lower())
                        if w not in STOP and len(w) > 2)

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            hits = []
            for q in dict.fromkeys([query, keywords]):      # full question, then keywords
                if not q:
                    continue
                r = await client.get(f"{base}/api/search",
                                     params={"query": q, "count": 10}, headers=headers)
                print(f"BookStack search '{q}' -> {r.status_code}")
                r.raise_for_status()
                hits = [i for i in r.json().get("data", []) if i.get("type") == "page"]
                if hits:
                    break

            parts, sources = [], []
            for h in hits[:3]:
                p = await client.get(f"{base}/api/pages/{h['id']}", headers=headers)
                if p.status_code != 200:
                    continue
                d = p.json()
                n = len(parts) + 1
                parts.append(f"[{n}] {h['name']}\n{body[:3000]}")
                sources.append({"title": h["name"], "url": f"{base}/link/{h['id']}"})

            print(f"BookStack returned {len(parts)} pages")
            return "\n\n".join(parts), sources
        except Exception as e:
            print(f"Error fetching from BookStack: {e}")
            return "", []

async def handle_custom_agent_query(query: str, ticket_id):
    ticket_id = str(ticket_id)
    print(f"Processing query for ticket {ticket_id}: {query}")
    if not db:
        print("Skipping Firebase push: Firebase is not initialized.")
        return

    ref = db.collection("tickets").document(ticket_id).collection("suggestions").document()
    context, sources = await search_bookstack(query)

    if not context:
        ref.set({"title": "AI Answer",
                 "text": "No matching pages were found in the Knowledge Base.",
                 "source": "BookStack KB", "sources": [],
                 "timestamp": firestore.SERVER_TIMESTAMP})
        return

    prompt = f"""You are an internal AI assistant for a customer support agent.
The agent asked: "{query}"

Knowledge Base pages:
{context}

Instructions:
1. Answer clearly and concisely, using a numbered list for steps (one per line).
2. Use ONLY the Knowledge Base pages above.
3. If the answer is not there, say "I cannot find this information in the Knowledge Base."
4. Cite using the page number in brackets right after the claim, like [1] or [2]. Do not write page titles or a sources list."""

    text = "The AI service is busy. Please try again in a moment."
    for attempt in range(4):
        try:
            resp = await gemini_client.aio.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            text = (resp.text or "I cannot find this information in the Knowledge Base.").strip()
            break
        except APIError as e:
            if e.code in (429, 503) and attempt < 3:
                await asyncio.sleep(2 ** attempt)
                continue
            print(f"Gemini error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break

    ref.set({"title": "AI Answer", "text": text, "source": "BookStack KB",
             "sources": sources, "timestamp": firestore.SERVER_TIMESTAMP})
    print(f"Pushed answer to ticket {ticket_id}")

async def search_zendesk(query: str, start: int = 1) -> tuple[str, list[dict]]:
    sub = (os.getenv("ZENDESK_SUBDOMAIN") or "").strip()
    if not sub:
        print("Zendesk missing env var: ZENDESK_SUBDOMAIN")
        return "", []

    base = f"https://{sub}.zendesk.com/api/v2/help_center"
    email, token = os.getenv("ZENDESK_EMAIL"), os.getenv("ZENDESK_API_TOKEN")
    auth = (f"{email}/token", token) if email and token else None   # optional; public articles need none

    kws = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in STOP and len(w) > 2]

    async with httpx.AsyncClient(timeout=20, auth=auth) as client:
        try:
            arts = []
            for q in dict.fromkeys([query, " ".join(kws)]):
                if not q:
                    continue
                r = await client.get(f"{base}/articles/search.json", params={"query": q, "per_page": 5})
                print(f"Zendesk search '{q}' -> {r.status_code}")
                if r.status_code == 200:
                    arts = r.json().get("results", [])
                if arts:
                    break

            if not arts and kws:   # fallback: rank all articles by keyword overlap
                r = await client.get(f"{base}/en-us/articles.json", params={"per_page": 100})
                print(f"Zendesk list fallback -> {r.status_code}")
                if r.status_code == 200:
                    def score(a):
                        text = (a["title"] + " " + html_to_text(a.get("body", ""))).lower()
                        return sum(text.count(k) for k in kws)
                    ranked = sorted(r.json().get("articles", []), key=score, reverse=True)
                    arts = [a for a in ranked if score(a) > 0]

            parts, sources = [], []
            for a in arts[:3]:
                n = start + len(parts)
                parts.append(f"[{n}] {a['title']}\n{html_to_text(a.get('body', ''))[:3000]}")
                sources.append({"title": a["title"], "url": a["html_url"]})

            print(f"Zendesk returned {len(parts)} articles")
            return "\n\n".join(parts), sources
        except Exception as e:
            print(f"Error fetching from Zendesk: {e}")
            return "", []

async def search_kb(query: str) -> tuple[str, list[dict]]:
    mode = os.getenv("KB_SOURCE", "bookstack").lower()   # bookstack | zendesk | both

    if mode == "zendesk":
        return await search_zendesk(query)
    if mode == "both":
        bs_ctx, bs_src = await search_bookstack(query)
        zd_ctx, zd_src = await search_zendesk(query, start=len(bs_src) + 1)
        return "\n\n".join(c for c in (bs_ctx, zd_ctx) if c), bs_src + zd_src
    return await search_bookstack(query)

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
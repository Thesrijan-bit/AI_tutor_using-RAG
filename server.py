import os
import json
import asyncio
import aiosqlite
import uuid
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# Removed unused llama_index imports

import torchvision.transforms as T
import torch
from dotenv import load_dotenv
from rag_engine import RAGEngine

load_dotenv()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(lifespan=lifespan)
rag = RAGEngine()

# --- Async Database Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "tutor_memory.db")

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                reasoning TEXT,
                image_url TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id)')
        await db.commit()

async def save_message(role, content, session_id="default", reasoning=None, image_url=None):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO messages (session_id, role, content, reasoning, image_url) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, reasoning, image_url)
        )
        await db.commit()
    # Also index in Vector DB for semantic retrieval (Long term memory)
    if content:
        rag.add_chat_message(role, content, session_id)

async def get_recent_history(session_id: str, limit=5) -> List[Any]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT role, content, image_url FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit)
        ) as cursor:
            rows = await cursor.fetchall()
            
    history = []
    for row in reversed(rows):
        if row['role'] == "user":
            if row['image_url']:
                history.append(HumanMessage(content=[
                    {"type": "text", "text": row['content']},
                    {"type": "image_url", "image_url": {"url": row['image_url']}}
                ]))
            else:
                history.append(HumanMessage(content=row['content']))
        elif row['role'] == "ai":
            history.append(AIMessage(content=row['content']))
    return history

# --- AI Setup ---
KIMI_API_KEY = os.getenv("KIMI_API_KEY")
client = ChatOpenAI(
  model="moonshotai/kimi-k2.5", # NVIDIA NIM model name for Kimi 2.5
  api_key=KIMI_API_KEY,
  base_url="https://integrate.api.nvidia.com/v1",
  temperature=0.85,
)

# Initialize RAG Engine
try:
    rag = RAGEngine()
except Exception as e:
    print(f"Warning: RAGEngine failed to initialize: {e}")
    rag = None

class QuestionRequest(BaseModel):
    question: Any = ""
    image: Any = None
    session_id: str = "default"

def optimize_image(b64_string: str) -> str:
    """Uses Pillow & Torchvision to optimize and resize image for the Vision API."""
    try:
        if "," in b64_string:
            header, b64_string = b64_string.split(",", 1)
        
        img_data = base64.b64decode(b64_string)
        img = Image.open(BytesIO(img_data))
        
        # Convert to RGB
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            
        # --- Torchvision Pre-processing (FIXED: NO CROPPING) ---
        # We resize while maintaining the full content of the image.
        transform = T.Compose([
            T.Resize(1024, interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.ToPILImage()
        ])
        img = transform(img)
        
        print(f"DEBUG: Optimized Image Size -> {img.size}")
        
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=95) # High quality for math
        optimized_b64 = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{optimized_b64}"
    except Exception as e:
        print(f"ERROR optimizing image: {e}")
        return b64_string

@app.post("/api/ask")
async def ask_nova(req: QuestionRequest):
    async def generate():
        try:
            prompt_text = str(req.question) if req.question else "Analyze the image."
            print(f"DEBUG: Nova receiving query -> '{prompt_text}'")
            
            # 1. Fetch raw history (Last 10 messages)
            raw_history = await get_recent_history(req.session_id, limit=10)
            
            # 2. Preserve Multi-Modal Structure (Logic provided by user)
            recent_history = []
            for msg in raw_history:
                if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
                    # Keep the content as a list so it preserves both 'text' and 'image_url' types
                    recent_history.append(HumanMessage(content=msg.content))
                else:
                    recent_history.append(msg)

            # 3. Context Router (Suppress history for greetings)
            should_include_history = True
            clean_text = prompt_text.lower().strip()
            is_greeting = any(clean_text.startswith(g) for g in ["hi", "hello", "hey", "yo"]) and len(clean_text) < 10
            
            semantic_context = ""
            if is_greeting:
                should_include_history = False
            else:
                # --- TRIGGER-BASED RAG ---
                needs_history = any(w in clean_text for w in ["remember", "previous", "earlier", "told", "said"])
                needs_docs = any(w in clean_text for w in ["document", "pdf", "data", "file"])
                
                context_parts = []
                if needs_history and rag:
                    relevant_history = rag.query_relevant_history(prompt_text, limit=10)
                    if relevant_history:
                        context_parts.append("\n".join([f"Past conversation: {doc.page_content}" for doc in relevant_history]))
                if needs_docs and rag:
                    relevant_docs = rag.query_documents(prompt_text, limit=10)
                    if relevant_docs:
                        context_parts.append("\n".join([f"Document excerpt: {doc.page_content}" for doc in relevant_docs]))
                if context_parts:
                    semantic_context = "\n\n".join(context_parts)
            image_description = ""
            active_image = req.image
            
            if not active_image:
                async with aiosqlite.connect(DB_PATH) as db:
                    db.row_factory = aiosqlite.Row
                    async with db.execute(
                        "SELECT image_url FROM messages WHERE session_id = ? AND image_url IS NOT NULL ORDER BY timestamp DESC LIMIT 1",
                        (req.session_id,)
                    ) as cursor:
                        row = await cursor.fetchone()
                        if row: active_image = row['image_url']

            if active_image:
                print("DEBUG: Gemini Native Vision activating...")
                optimized_img = optimize_image(active_image)

            # 4. Construct message chain
            system_content = (
                "You are Nova, an Elite AI Reasoning Tutor. "
                "CRITICAL: You are a native multi-modal AI. You can see images directly. "
                "If an image is attached to the user's query or in the history, analyze it meticulously. "
                "Never claim you cannot see."
                "\n\nPROMPT_MANDATE: Output must be elegant, professional, and visual. Use Markdown tables for formatting."
                "\nCRITICAL: NEVER use LaTeX format (like $$ or \\( \\)) for math or chemistry! Use plain text and unicode characters only."
                "\nFor chemistry structures, you MUST generate them using SMILES notation wrapped inside a markdown code block with the language 'smiles'. Example:\n```smiles\nCCO\n```"
                "\n\n*** CRITICAL FORMATTING INSTRUCTION ***\n"
                "1. You MUST think step-by-step and write out your internal reasoning first.\n"
                "2. When you are finished reasoning, you MUST output the exact line `===ANSWER===`.\n"
                "3. After `===ANSWER===`, write your final response to the user."
            )
            
            if semantic_context:
                system_content += f"\n\n[RELEVANT PAST CONTEXT]:\n{semantic_context}"

            # Brain Dump for diagnostics (Verify what Nova is actually told)
            try:
                with open("nova_brain_dump.txt", "w", encoding="utf-8") as f:
                    f.write(f"SYSTEM_PROMPT:\n{system_content}\n\nQUERY: {prompt_text}")
            except: pass

            messages = [SystemMessage(content=system_content)]
            if should_include_history:
                messages += recent_history
            
            # 5. Final Query (Multi-Modal with OPTIMIZED image)
            if active_image:
                # Use the sharpened, compressed version to avoid payload limits
                img_to_send = optimized_img if 'optimized_img' in locals() else optimize_image(active_image)
                messages.append(HumanMessage(content=[
                    {"type": "text", "text": f"STUDENT_QUERY: {prompt_text}"},
                    {"type": "image_url", "image_url": {"url": img_to_send}}
                ]))
            else:
                messages.append(HumanMessage(content=f"STUDENT_QUERY: {prompt_text}"))

            full_reasoning = ""
            full_answer = ""
            in_think_block = True
            
            buffer = ""
            async for chunk in client.astream(messages):
                content = chunk.content if chunk.content else ""
                buffer += content
                
                if in_think_block:
                    if "===ANSWER===" in buffer:
                        in_think_block = False
                        reasoning_part, buffer = buffer.split("===ANSWER===", 1)
                        full_reasoning += reasoning_part
                        yield json.dumps({"reasoning": reasoning_part, "answer": ""}) + "\n"
                    else:
                        if len(buffer) > 12:
                            safe_reasoning = buffer[:-12]
                            buffer = buffer[-12:]
                            full_reasoning += safe_reasoning
                            yield json.dumps({"reasoning": safe_reasoning, "answer": ""}) + "\n"
                else:
                    if buffer:
                        full_answer += buffer
                        yield json.dumps({"reasoning": "", "answer": buffer}) + "\n"
                        buffer = ""

            if buffer:
                if in_think_block:
                    full_reasoning += buffer
                    yield json.dumps({"reasoning": buffer, "answer": ""}) + "\n"
                else:
                    full_answer += buffer
                    yield json.dumps({"reasoning": "", "answer": buffer}) + "\n"
                
            # Background save
            asyncio.create_task(save_message("ai", full_answer, session_id=req.session_id, reasoning=full_reasoning))
            asyncio.create_task(save_message("user", prompt_text, session_id=req.session_id, image_url=req.image))

        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.get("/api/history/{session_id}")
async def fetch_history(session_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT role, content, reasoning, image_url FROM messages WHERE session_id = ? ORDER BY timestamp ASC", (session_id,)) as cursor:
            rows = await cursor.fetchall()
            return [{"role": r['role'], "content": r['content'], "reasoning": r['reasoning'], "image": r['image_url']} for r in rows]

@app.post("/api/new_session")
async def new_session():
    return {"session_id": str(uuid.uuid4())}

app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

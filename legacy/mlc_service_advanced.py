import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.responses import StreamingResponse
import re
# DLL matching logic from run_mlc_chat.py
def base_key_from_model_name(model_name: str) -> str:
    s = model_name
    s = re.sub(r'([_-]mlc)$', '', s, flags=re.IGNORECASE)
    s = re.split(r"-q\d+f?\d*[_-]?\d*-?mlc", model_name, flags=re.IGNORECASE)
    return s[0] if s else model_name

def match_model_dlls(model_name: str, dlls):
    base_key = base_key_from_model_name(model_name).lower()
    matches = []
    for dll in dlls:
        name = dll.name.lower()
        if base_key and base_key in name:
            matches.append(dll)
    if not matches:
        loose_key = model_name.split("-q")[0].lower()
        for dll in dlls:
            if loose_key in dll.name.lower():
                matches.append(dll)
    return matches
from pathlib import Path
# Import FastAPI here for clarity
from fastapi import FastAPI, HTTPException, Depends, Request

# Utility to find available models
def find_models(dist_root: Path = Path("dist")):
    models = []
    for cfg in dist_root.rglob("mlc-chat-config.json"):
        model_dir = cfg.parent
        models.append({
            "model_name": model_dir.name,
            "model_dir": str(model_dir),
            "config_path": str(cfg)
        })
    return models

# Cross-platform model_lib (DLL) resolution
import platform
from threading import Lock

def resolve_model_lib(model_name: str, libs_dir: Path) -> str:
    ext = { "Windows": ".dll", "Darwin": ".dylib" }.get(platform.system(), ".so")
    if not libs_dir.exists():
        raise HTTPException(status_code=404, detail=f"Libs dir not found: {libs_dir}")
    dlls = list(libs_dir.glob(f"*{ext}"))
    matches = match_model_dlls(model_name, dlls)
    if matches:
        return str(matches[0])
    # fallback: try generic mlc_llm lib
    generic = libs_dir / f"mlc_llm{ext}"
    if generic.exists():
        return str(generic)
    raise HTTPException(status_code=404, detail=f"No matching model lib found for {model_name} in {libs_dir}")

# --- Thread-Safe Engine Cache ---
# Using a simple dictionary and a lock is more robust for managing stateful
# objects like MLCEngine than relying on lru_cache's automatic eviction,
# which can lead to race conditions and access violations in a threaded server.
_engine_cache: dict = {}
_cache_lock = Lock()

def get_engine_safely(model_dir, device, dll_path):
    """
    Initializes and returns a cached MLCEngine in a thread-safe manner.
    """
    cache_key = (model_dir, device, dll_path)
    with _cache_lock:
        if cache_key in _engine_cache:
            return _engine_cache[cache_key]
        
        # To prevent the cache from growing indefinitely, we'll limit it.
        # If the cache is full, evict all engines before adding a new one.
        if len(_engine_cache) >= 2: # maxsize=2
            # This is a simplified eviction strategy. A more sophisticated LRU
            # cache could be implemented for better performance.
            for engine in _engine_cache.values():
                engine.terminate()
            _engine_cache.clear()

        engine = MLCEngine(model=model_dir, 
                           model_lib=dll_path, 
                           device=device)
        _engine_cache[cache_key] = engine
        return engine

def clear_engine_cache():
    """Safely clears the engine cache and unloads all engines."""
    with _cache_lock:
        for engine in _engine_cache.values():
            engine.terminate()
        _engine_cache.clear()
# --- End Thread-Safe Engine Cache ---

"""
mlc_service_advanced.py
Session-based MLC-LLM API with SQLite persistence.
"""
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Any
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session as OrmSession
from mlc_llm import MLCEngine
import json



# Database setup

DATABASE_URL = os.environ.get("MLC_SQLITE_URL", "sqlite:///mlc_sessions.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
# Enable WAL mode for concurrency
from sqlalchemy import text
with engine.connect() as conn:
    conn.execute(text("PRAGMA journal_mode=WAL"))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class SessionDB(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_active = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    meta = Column(Text, nullable=True)  # Renamed from 'metadata'
    messages = relationship("MessageDB", back_populates="session", cascade="all, delete-orphan")

class MessageDB(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    session = relationship("SessionDB", back_populates="messages")

# Add index for (session_id, id)
from sqlalchemy import Index
Index("ix_messages_session_id_id", MessageDB.session_id, MessageDB.id)

Base.metadata.create_all(bind=engine)

# FastAPI app
# FastAPI app



from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    yield
    # On shutdown, explicitly unload all engines to ensure graceful termination.
    clear_engine_cache()

app = FastAPI(title="MLC-LLM Session Service", version="0.1.0", lifespan=lifespan)
# Auto-discover all tool modules so register_tool decorators run
import importlib, pkgutil
import tools
for m in pkgutil.iter_modules(tools.__path__):
    importlib.import_module(f"{tools.__name__}.{m.name}")

@app.get("/healthz")
def healthz():
    return {"ok": True}

# Endpoint to list available models
@app.get("/models")
def list_models():
    return {"models": find_models()}

from tools import get_all_tool_definitions

@app.get("/tools")
def list_tools():
    return {"tools": get_all_tool_definitions()}

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models
class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: datetime

class Message(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

class GenerateRequest(BaseModel):
    session_id: str = Field(...)
    prompt: str = Field(...)
    model_name: str = Field(...)
    device: Optional[str] = "auto"
    dll: Optional[str] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95
    # Add tools parameter for OpenAI-compatible tool definitions
    tools: Optional[List[dict]] = None
    # The mcp_server config is now handled by the client, which passes `tools`
    mcp_server: Optional[dict] = None

class GenerateResponse(BaseModel):
    session_id: str
    reply: str
    messages: List[Message]

class UnloadModelRequest(BaseModel):
    model_name: str
    device: Optional[str] = "auto"

@app.post("/sessions", response_model=CreateSessionResponse)
def create_session(db: OrmSession = Depends(get_db)):
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    session = SessionDB(id=session_id, created_at=now, last_active=now)
    db.add(session)
    db.commit()
    db.refresh(session)
    return CreateSessionResponse(session_id=getattr(session, 'id'), created_at=getattr(session, 'created_at'))

@app.get("/sessions")
def list_sessions(db: OrmSession = Depends(get_db)):
    sessions = db.query(SessionDB).all()
    return [{"session_id": s.id, "created_at": s.created_at, "last_active": s.last_active} for s in sessions]

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, db: OrmSession = Depends(get_db)):
    session = db.query(SessionDB).filter(SessionDB.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete(session)
    db.commit()
    return {"detail": "Session deleted"}

@app.delete("/sessions")
def delete_all_sessions(db: OrmSession = Depends(get_db)):
    try:
        num_deleted = db.query(SessionDB).delete()
        db.commit()
        return {"detail": f"Deleted {num_deleted} sessions."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting sessions: {e}")

@app.get("/sessions/{session_id}/messages", response_model=List[Message])
def get_session_messages(session_id: str, db: OrmSession = Depends(get_db)):
    session = db.query(SessionDB).filter(SessionDB.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return [Message(id=m.id, role=m.role, content=m.content, created_at=m.created_at) for m in session.messages]

import json
# # --- HARD-CODED: force this tool on every request ---
# FORCE_TOOL_NAME = "get_current_time"  # change to your tool's name
# # ----------------------------------------------------

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, db: OrmSession = Depends(get_db)):
    session = db.query(SessionDB).filter(SessionDB.id == req.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    # Add user message
    user_msg = MessageDB(session_id=session.id, role="user", content=req.prompt)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)
    # Prepare history (all messages in order)
    history: List[dict[str, Any]] = [{"role": m.role, "content": m.content}  # type: ignore
        for m in db.query(MessageDB).filter(MessageDB.session_id == session.id).order_by(MessageDB.id).all()]
    model_dir = os.path.join("dist", req.model_name)
    if not os.path.isdir(model_dir):
        raise HTTPException(status_code=404, detail=f"Model directory not found: {model_dir}")
    device = req.device if req.device is not None else "auto"
    libs_dir = Path("dist") / "libs"
    if req.dll:
        candidate = libs_dir / req.dll
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail=f"Model lib not found: {candidate}")
        dll_path = str(candidate)
    else:
        dll_path = resolve_model_lib(req.model_name, libs_dir)

    mlc_engine = get_engine_safely(model_dir, device, dll_path)
    available_tools = get_all_tool_definitions()
    # Filter client-provided tools to those registered, else use all available
    if req.tools:
        reg_names = {t["function"]["name"] for t in available_tools}
        requested = [t.get("function", {}).get("name") for t in req.tools]
        missing = [n for n in requested if n not in reg_names]
        if missing:
            # optional: use your logger instead of print
            print(f"[tools] Ignoring unregistered tools requested by client: {missing}")
        tools_to_use = [t for t in req.tools if t.get("function", {}).get("name") in reg_names]
    else:
        tools_to_use = available_tools

    # # --- HARD-CODED: ensure the forced tool definition is included ---
    # forced_defs = [t for t in available_tools if t["function"]["name"] == FORCE_TOOL_NAME]
    # if not forced_defs:
    #     # Optional: be loud if you expect it to exist
    #     raise HTTPException(status_code=500, detail=f"Forced tool '{FORCE_TOOL_NAME}' not registered")
    # forced_def = forced_defs[0]
    # if all(t.get("function", {}).get("name") != FORCE_TOOL_NAME for t in tools_to_use):
    #     tools_to_use.append(forced_def)
    # # ---------------------------------------------------------------


    # OpenAI-style tool loop
    reply = ""
    max_tool_loops = 3
    tool_loop_count = 0

    while True:
        tool_choice = "auto" if tools_to_use else "none"
        out = mlc_engine.chat.completions.create(
            messages=history,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            tools=tools_to_use,
            tool_choice="auto",  # <- important
        )

        # # --- HARD-CODED: force the function call every turn ---
        # print("[tools] names:", [t["function"]["name"] for t in tools_to_use])
        # print("[tools] forcing:", tools_to_use)
        # print("[tools] messages_last_role:", history[-1]["role"] if history else None)

        # out = mlc_engine.chat.completions.create(
        #     messages=history,
        #     max_tokens=req.max_tokens,
        #     temperature=req.temperature,
        #     top_p=req.top_p,
        #     tools=tools_to_use,
        #     tool_choice={"type": "function", "function": {"name": FORCE_TOOL_NAME}},
        # )
        # # ------------------------------------------------------
        print(f"[mlc_service] Model output: {out}")  # optional: use your logger instead of print
        print("[mlc_service] use_function_calling:", getattr(out, "use_function_calling", None))
        first = out.choices[0] if out and out.choices else None
        print("[mlc_service] tool_calls on first choice:", getattr(first.message, "tool_calls", None) if first else None)
        print("[mlc_service] finish_reason:", getattr(first, "finish_reason", None) if first else None)

        message = out.choices[0].message if out and out.choices else None

        if message and getattr(message, "tool_calls", None):
            # 1) record assistant tool-call message
            history.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": (
                                tc.function.arguments
                                if isinstance(tc.function.arguments, str)
                                else json.dumps(tc.function.arguments or {})
                            )
                        }
                    }
                    for tc in (message.tool_calls or [])
                ],
            })

            # 2) execute each tool and append a tool message
            from tools import execute_tool
            for tc in (message.tool_calls or []):
                name = tc.function.name
                args = tc.function.arguments
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except: args = {}
                elif args is None:
                    args = {}
                try:
                    result = execute_tool(name, args)
                except Exception as e:
                    result = f"Tool execution error: {e}"
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": str(result),
                })

            tool_loop_count += 1
            if tool_loop_count >= max_tool_loops:
                reply = (
                    "I ran tools several times but couldn’t finish the reasoning. "
                    "Try rephrasing or reducing steps."
                )
                break

            continue

        # no more tool_calls → final assistant reply
        reply = (message.content or "") if message else ""
        break

    # Add assistant message to DB
    assistant_msg = MessageDB(session_id=session.id, role="assistant", content=reply)
    db.add(assistant_msg)
    setattr(session, 'last_active', datetime.now(timezone.utc))
    db.commit()
    db.refresh(assistant_msg)
    # Return with full message history
    messages = db.query(MessageDB).filter(MessageDB.session_id == session.id).order_by(MessageDB.id).all()
    return GenerateResponse(
        session_id=str(session.id),
        reply=str(reply),
        messages=[
            Message(
                id=int(m.id),  # type: ignore
                role=str(m.role),  # type: ignore
                content=str(m.content),  # type: ignore
                created_at=m.created_at  # type: ignore
            ) for m in messages
        ]
    )

@app.post("/models/unload")
def unload_model(req: UnloadModelRequest):
    """
    Unloads a specific model from the cache to free up memory.
    """
    # With lru_cache, we can't easily target a specific entry for removal
    # without inspecting the cache's internal structure, which is not robust.
    # A full cache clear is the simplest way if memory needs to be freed up.
    # For more granular control, the old manual cache might be needed, but with
    # much more careful locking.
    clear_engine_cache()
    return {"detail": f"All models in the cache have been cleared to unload '{req.model_name}'."}




# Streaming endpoint for token-by-token output
@app.post("/generate_stream")
async def generate_stream(request: Request, req: GenerateRequest, db: OrmSession = Depends(get_db)):
    session = db.query(SessionDB).filter(SessionDB.id == req.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if req.tools:
        raise HTTPException(status_code=400, detail="Streaming endpoint does not support tools. Use /generate.")
    # Add user message
    user_msg = MessageDB(session_id=session.id, role="user", content=req.prompt)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)
    # Prepare history (all messages in order)
    history = [{"role": m.role, "content": m.content} for m in db.query(MessageDB).filter(MessageDB.session_id == getattr(session, 'id')).order_by(MessageDB.id).all()]
    # Model and DLL logic (reuse from /generate)
    model_dir = os.path.join("dist", req.model_name)
    if not os.path.isdir(model_dir):
        raise HTTPException(status_code=404, detail=f"Model directory not found: {model_dir}")
    device = req.device if req.device is not None else "auto"
    libs_dir = Path("dist") / "libs"
    if req.dll:
        candidate = libs_dir / req.dll
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail=f"Model lib not found: {candidate}")
        dll_path = str(candidate)
    else:
        dll_path = resolve_model_lib(req.model_name, libs_dir)
    # Model setup: catch errors before streaming starts
    try:
        mlc_engine = get_engine_safely(model_dir, device, dll_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize model: {e}")

    async def token_stream():
        reply_chunks = []
        streamed = False
        client_disconnected = False
        try:
            for response in mlc_engine.chat.completions.create(
                messages=history,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stream=True,
            ):
                for choice in response.choices:
                    chunk = choice.delta.content or ""
                    # Ensure chunk is a string
                    if not isinstance(chunk, str):
                        chunk = json.dumps(chunk)
                    reply_chunks.append(chunk)
                    streamed = True
                    yield chunk
                    # Check for client disconnect
                    if request is not None and hasattr(request, 'is_disconnected'):
                        if await request.is_disconnected():
                            client_disconnected = True
                            break
                if client_disconnected:
                    break
        finally:
            # Only persist assistant message if at least one chunk was streamed and client is still connected
            if streamed and reply_chunks and not client_disconnected:
                assistant_msg = MessageDB(session_id=session.id, role="assistant", content="".join(reply_chunks))
                db.add(assistant_msg)
                setattr(session, 'last_active', datetime.now(timezone.utc))
                db.commit()

    return StreamingResponse(token_stream(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mlc_service_advanced:app", host="127.0.0.1", port=8090, reload=False, workers=1)

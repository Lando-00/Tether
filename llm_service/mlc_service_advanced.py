from fastapi.responses import StreamingResponse
import re
# DLL matching logic from run_mlc_chat.py
def base_key_from_model_name(model_name: str) -> str:
    m = re.split(r"-q\d+f?\d*[_-]?\d*-?mlc", model_name, flags=re.IGNORECASE)
    return m[0] if m else model_name

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
from functools import lru_cache
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

# LRU cache for MLCEngine
_engine_lock = Lock()
_ENGINE_CACHE_SIZE = 2
_engine_cache = {}

def _engine_cache_key(model_dir, device, dll_path):
    return (str(model_dir), str(device), str(dll_path))

def get_engine(model_dir, device, dll_path):
    key = _engine_cache_key(model_dir, device, dll_path)
    with _engine_lock:
        if key in _engine_cache:
            return _engine_cache[key]
        # If cache is full, evict the oldest
        if len(_engine_cache) >= _ENGINE_CACHE_SIZE:
            old_key, old_engine = next(iter(_engine_cache.items()))
            try:
                old_engine.terminate()
            except Exception:
                pass
            del _engine_cache[old_key]
        engine = MLCEngine(model=model_dir, model_lib=dll_path, device=device)
        _engine_cache[key] = engine
        return engine

"""
mlc_service_advanced.py
Session-based MLC-LLM API with SQLite persistence.
"""
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Request
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
    # On shutdown
    with _engine_lock:
        for engine in _engine_cache.values():
            try:
                engine.terminate()
            except Exception:
                pass
        _engine_cache.clear()

app = FastAPI(title="MLC-LLM Session Service", version="0.1.0", lifespan=lifespan)

# Endpoint to list available models
@app.get("/models")
def list_models():
    return {"models": find_models()}

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
    device: Optional[str] = "opencl"
    dll: Optional[str] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95

class GenerateResponse(BaseModel):
    session_id: str
    reply: str
    messages: List[Message]

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

@app.get("/sessions/{session_id}/messages", response_model=List[Message])
def get_session_messages(session_id: str, db: OrmSession = Depends(get_db)):
    session = db.query(SessionDB).filter(SessionDB.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return [Message(id=m.id, role=m.role, content=m.content, created_at=m.created_at) for m in session.messages]

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
    history = [{"role": m.role, "content": m.content} for m in db.query(MessageDB).filter(MessageDB.session_id == session.id).order_by(MessageDB.id).all()]
    # Run LLM
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
    mlc_engine = get_engine(model_dir, device, dll_path)
    out = mlc_engine.chat.completions.create(
        messages=history,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    reply = out.choices[0].message.content if out and out.choices else ""
    # Add assistant message
    assistant_msg = MessageDB(session_id=session.id, role="assistant", content=reply)
    db.add(assistant_msg)
    setattr(session, 'last_active', datetime.now(timezone.utc))
    db.commit()
    db.refresh(assistant_msg)
    # Return
    messages = db.query(MessageDB).filter(MessageDB.session_id == getattr(session, 'id')).order_by(MessageDB.id).all()
    return GenerateResponse(
        session_id=getattr(session, 'id'),
        reply=str(reply) if reply is not None else "",
        messages=[
            Message(
                id=getattr(m, 'id'),
                role=getattr(m, 'role'),
                content=getattr(m, 'content'),
                created_at=getattr(m, 'created_at')
            ) for m in messages
        ]
    )

# Streaming endpoint for token-by-token output
@app.post("/generate_stream")
async def generate_stream(request: Request, req: GenerateRequest, db: OrmSession = Depends(get_db)):
    session = db.query(SessionDB).filter(SessionDB.id == req.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
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
        mlc_engine = get_engine(model_dir, device, dll_path)
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

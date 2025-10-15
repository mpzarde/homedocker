# -*- coding: utf-8 -*-
"""
Ollama Router — v5.9.3 (with Header Logging)
"""

import os
import re
import json
import time
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, AsyncIterator, List, Optional, Tuple
from collections import OrderedDict

import httpx
import yaml
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging

CFG_PATH = os.getenv("ROUTER_CONFIG", "config.yaml")
_cfg: Dict[str, Any] = {}
if os.path.exists(CFG_PATH):
    try:
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            _cfg = yaml.safe_load(f) or {}
    except Exception:
        _cfg = {}

models_cfg = (_cfg.get("models") or {})
router_cfg = (_cfg.get("router") or {})
ollama_cfg = (_cfg.get("ollama") or {})

OLLAMA_BASE: str = os.getenv("OLLAMA_BASE", ollama_cfg.get("base_url", "http://localhost:11434"))
MODEL_DEV: str = os.getenv("MODEL_DEV", models_cfg.get("dev", "qwen2.5-coder:32b"))
MODEL_WRITING: str = os.getenv("MODEL_WRITING", models_cfg.get("writing", "llama3.1:8b"))
MODEL_GENERAL: str = os.getenv("MODEL_GENERAL", models_cfg.get("general", "codestral:22b"))
CLASSIFIER_MODEL: str = os.getenv("CLASSIFIER", models_cfg.get("classifier", "qwen2.5:7b-instruct"))

DEFAULT_LOCAL_ON_ERROR: bool = str(os.getenv("DEFAULT_LOCAL_ON_ERROR", str(router_cfg.get("default_to_local_on_error", True)))).lower() == "true"
VERBOSE: bool = str(os.getenv("VERBOSE_LOGGING", str(router_cfg.get("verbose_logging", False)))).lower() == "true"
LOG_HEADERS: bool = str(os.getenv("LOG_HEADERS", str(router_cfg.get("log_headers", False)))).lower() == "true"
SESSION_TTL_MIN: int = int(router_cfg.get("session_timeout", 30))

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE: str = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")

TIMEOUT = httpx.Timeout(120.0, connect=15.0, read=120.0)
client = httpx.AsyncClient(timeout=TIMEOUT, limits=httpx.Limits(max_connections=50, max_keepalive_connections=25))

app = FastAPI(title="Ollama Router (v5.9.3)")

LOG_LEVEL = logging.DEBUG if VERBOSE else logging.INFO
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ollama-router")

# Header Logging Middleware
class HeaderLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if LOG_HEADERS:
            # Log all request headers
            headers_dict = {k: v for k, v in request.headers.items()}
            logger.info(f"REQUEST HEADERS for {request.method} {request.url.path}: {json.dumps(headers_dict, indent=2)}")
        
        response = await call_next(request)
        return response

# Add the middleware to the app
app.add_middleware(HeaderLoggingMiddleware)

def _log_chat(kind: str, requested_model: Optional[str], intent: str,
              resolved_model: str, backend: str, stream: bool,
              sess: Optional[str] = None, aux: bool = False, key_src: Optional[str] = None):
    try:
        sid = f" sess='{sess}'" if sess else ""
        src = f" keysrc='{key_src}'" if key_src else ""
        line = f"{kind} start:{sid}{src} requested='{requested_model}' intent='{intent}' -> resolved='{resolved_model}' backend={backend} stream={stream}"
        (logger.debug if aux else logger.info)(line)
    except Exception:
        pass

DEV_HINT = re.compile(r"\b(code|bug|stacktrace|api|sql|regex|compile|unit test|refactor|java|spring|typescript|python|golang|kotlin|maven|gradle)\b", re.I)
WRITE_HINT = re.compile(r"\b(rewrite|edit|tone|copy|outline|chapter|scene|synopsis|blurb|style|novel|prose|blog|essay|draft)\b", re.I)
NET_HINT = re.compile(r"\b(weather|forecast|news|today|latest|price|stock|score|currency|exchange rate|now|current)\b", re.I)

async def classify_intent(text: str) -> str:
    if DEV_HINT.search(text or ""): return "dev"
    if WRITE_HINT.search(text or ""): return "writing"
    if NET_HINT.search(text or ""): return "internet"
    try:
        r = await client.post(f"{OLLAMA_BASE}/api/generate",
                              json={"model": CLASSIFIER_MODEL, "prompt": f"Classify as one of [dev, writing, general]: {text}", "stream": False})
        label = (r.json().get("response", "") or "").lower()
        if "dev" in label: return "dev"
        if "write" in label: return "writing"
    except Exception:
        pass
    return "general"

def pick_local_model(label: str) -> str:
    return MODEL_DEV if label == "dev" else MODEL_WRITING if label == "writing" else "openai/gpt-4o-mini" if label == "internet" else MODEL_GENERAL

VIRTUAL_ALIASES = {"auto","auto:latest","dev","dev:latest","writing","writing:latest","general","general:latest"}

def is_virtual_model(name: Optional[str]) -> bool:
    return bool(name) and name.strip().lower() in VIRTUAL_ALIASES

def resolve_virtual_model(name: Optional[str], intent_hint: Optional[str] = None) -> str:
    n = (name or "").strip().lower()
    if n in ("auto","auto:latest"): return pick_local_model(intent_hint or "general")
    if n in ("dev","dev:latest"): return MODEL_DEV
    if n in ("writing","writing:latest"): return MODEL_WRITING
    if n in ("general","general:latest"): return MODEL_GENERAL
    return name or MODEL_GENERAL

def choose_backend(model_name: str) -> str:
    return "openai" if model_name and model_name.lower().startswith("openai/") else "ollama"

class _LRUCache:
    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self.store: "OrderedDict[str, tuple[str, float]]" = OrderedDict()
    def get(self, key: str) -> Optional[str]:
        if not key:
            return None
        item = self.store.get(key)
        if item is None: return None
        val, ts = item
        self.store.move_to_end(key)
        if (time.time() - ts) > SESSION_TTL_MIN * 60:
            try: del self.store[key]
            except KeyError: pass
            return None
        return val
    def set(self, key: str, value: str):
        if not key:
            return
        self.store[key] = (value, time.time())
        self.store.move_to_end(key)
        if len(self.store) > self.capacity:
            self.store.popitem(last=False)

SESSION_CACHE = _LRUCache()

_CHAT_ID_RE = re.compile(r"/c/([0-9a-fA-F-]{36})|/chat/([A-Za-z0-9\-_]+)")

def _earliest_user_text(messages: Optional[List[Dict[str, Any]]]) -> str:
    if not messages:
        return ""
    for m in messages:
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""

def _conversation_key_from_messages(messages: List[Dict[str, Any]]) -> str:
    sys = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    usr = _earliest_user_text(messages)
    base_key = (sys + "\n---\n" + usr).strip()
    return hashlib.sha256(base_key.encode("utf-8")).hexdigest() if base_key else ""

def _deep_find_id(obj, keys=("conversation_id","chat_id","thread_id","threadId","id")) -> Optional[str]:
    try:
        if isinstance(obj, dict):
            for k in keys:
                if k in obj and isinstance(obj[k], (str, int)):
                    v = str(obj[k]).strip()
                    if v:
                        return v
            for v in obj.values():
                found = _deep_find_id(v, keys)
                if found:
                    return found
        elif isinstance(obj, list):
            for it in obj:
                found = _deep_find_id(it, keys)
                if found:
                    return found
    except Exception:
        pass
    return None

def _extract_conv_from_headers(req: Request) -> Tuple[Optional[str], Optional[str]]:
    # Only look for the exact "x-openwebui-chat-id" header
    v = req.headers.get("x-openwebui-chat-id")
    if v:
        return (f"webui:{v}", "x-openwebui-chat-id")
    return (None, None)

def _session_key_from_request(req: Request, messages: Optional[List[Dict[str, Any]]], prompt: Optional[str], body: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    # First try to get chat ID from x-openwebui-chat-id header
    sid, src = _extract_conv_from_headers(req)
    if sid:
        return sid, src or "header"
    
    # Fall back to peer+firstmsg hash ONLY
    ref = req.headers.get("Referer") or req.headers.get("X-Forwarded-Referer") or ""
    client_host = getattr(req.client, "host", "") or ""
    ua = req.headers.get("user-agent", "")
    origin = req.headers.get("origin", "")
    path_hint = ref.split("?", 1)[0]
    peer_raw = f"{client_host}|{ua}|{origin}|{path_hint}"
    first_msg = _earliest_user_text(messages) if messages is not None else (prompt or "")
    fp = ""
    if first_msg.strip():
        fp = hashlib.sha1(first_msg.strip().encode("utf-8")).hexdigest()[:6]
    if peer_raw.strip():
        return ("peer:" + hashlib.sha256(peer_raw.encode("utf-8")).hexdigest() + (f"+fp:{fp}" if fp else ""),
                "peer+firstmsg" if fp else "peer")
    return "", "none"

def ndjson_chunk(model: str, text: str) -> bytes:
    return (json.dumps({"model": model, "message": {"role": "assistant", "content": text}, "done": False}) + "\n").encode()

def ndjson_done() -> bytes:
    return (json.dumps({"done": True}) + "\n").encode()

async def stream_openai_chat_as_ndjson(messages: Any, model: str):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    body = {"model": model, "messages": messages, "stream": True}
    async with client.stream("POST", f"{OPENAI_BASE}/chat/completions", headers=headers, json=body) as resp:
        async for raw in resp.aiter_lines():
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                yield ndjson_done()
                break
            try:
                ev = json.loads(data)
                delta = ev["choices"][0]["delta"].get("content", "")
                if delta:
                    yield ndjson_chunk(model, delta)
            except Exception:
                continue

async def nonstream_openai_chat(messages: Any, model: str) -> Response:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    body = {"model": model, "messages": messages, "stream": False}
    r = await client.post(f"{OPENAI_BASE}/chat/completions", headers=headers, json=body)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    return JSONResponse({"model": model, "message": {"role": "assistant", "content": text}, "done": True})

async def stream_ollama(endpoint: str, payload: Dict[str, Any]):
    async with client.stream("POST", f"{OLLAMA_BASE}{endpoint}", json=payload) as upstream:
        async for chunk in upstream.aiter_raw():
            if not chunk:
                continue
            yield chunk

async def nonstream_ollama(endpoint: str, payload: Dict[str, Any]) -> Response:
    r = await client.post(f"{OLLAMA_BASE}{endpoint}", json=payload)
    return Response(content=r.content, media_type=r.headers.get("content-type", "application/json"))

def last_user_message(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""

def _is_aux_call(body: dict, messages: list[dict]) -> bool:
    if not body or body.get("stream") is True:
        return False
    txt = ""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            txt = (m.get("content") or "").strip()
            break
    return txt == ""

@app.post("/api/chat")
async def api_chat(req: Request):
    body = await req.json()
    messages: List[Dict[str, Any]] = body.get("messages", []) or []
    stream: bool = bool(body.get("stream", True))
    requested_model: Optional[str] = body.get("model")
    force = bool(body.get("force_reclassify")) or (req.headers.get("X-Router-Reclassify") == "1")

    use_auto = (requested_model is None) or (is_virtual_model(requested_model) and requested_model.strip().lower() in ("auto","auto:latest"))
    conv_key, key_src = _session_key_from_request(req, messages, None, body)
    aux = _is_aux_call(body, messages)

    cached = SESSION_CACHE.get(conv_key) if (use_auto and conv_key and not force) else None
    if cached:
        model = cached
        intent = "(skip)"
        source = "cache"
    else:
        if use_auto:
            intent = await classify_intent(last_user_message(messages))
            model = resolve_virtual_model("auto:latest", intent_hint=intent)
            source = "classified"
            if conv_key:
                SESSION_CACHE.set(conv_key, model)
        else:
            intent = "(explicit)"
            model = resolve_virtual_model(requested_model, intent_hint="general") if is_virtual_model(requested_model) else (requested_model or MODEL_GENERAL)
            source = None

    backend = choose_backend(model)
    log_model = model if not use_auto else (model + f" (auto:{source})")
    _log_chat("chat", requested_model, intent, log_model, backend, stream, sess=conv_key, aux=aux, key_src=key_src)

    if backend == "openai":
        real = model.split("/", 1)[1] if "/" in model else OPENAI_MODEL
        if not OPENAI_API_KEY:
            if DEFAULT_LOCAL_ON_ERROR:
                payload = dict(body); payload["model"] = MODEL_GENERAL
                return StreamingResponse(stream_ollama("/api/chat", payload), media_type="application/x-ndjson") if stream \
                       else await nonstream_ollama("/api/chat", payload)
            return JSONResponse({"error": "OPENAI_API_KEY not set"}, status_code=400)
        return StreamingResponse(stream_openai_chat_as_ndjson(messages, real), media_type="application/x-ndjson") if stream \
               else await nonstream_openai_chat(messages, real)

    payload = dict(body); payload["model"] = model
    return StreamingResponse(stream_ollama("/api/chat", payload), media_type="application/x-ndjson") if stream \
           else await nonstream_ollama("/api/chat", payload)

@app.post("/api/generate")
async def api_generate(req: Request):
    body = await req.json()
    prompt: str = body.get("prompt", "")
    stream: bool = bool(body.get("stream", True))
    requested_model: Optional[str] = body.get("model")
    force = bool(body.get("force_reclassify")) or (req.headers.get("X-Router-Reclassify") == "1")

    use_auto = (requested_model is None) or (is_virtual_model(requested_model) and requested_model.strip().lower() in ("auto","auto:latest"))
    conv_key, key_src = _session_key_from_request(req, None, prompt, body)
    aux = (not stream) and (not (prompt or "").strip())

    cached = SESSION_CACHE.get(conv_key) if (use_auto and conv_key and not force) else None
    if cached:
        model = cached
        intent = "(skip)"
        source = "cache"
    else:
        if use_auto:
            intent = await classify_intent(prompt)
            model = resolve_virtual_model("auto:latest", intent_hint=intent)
            source = "classified"
            if conv_key:
                SESSION_CACHE.set(conv_key, model)
        else:
            intent = "(explicit)"
            model = resolve_virtual_model(requested_model, intent_hint="general") if is_virtual_model(requested_model) else (requested_model or MODEL_GENERAL)
            source = None

    backend = choose_backend(model)
    log_model = model if not use_auto else (model + f" (auto:{source})")
    _log_chat("generate", requested_model, intent, log_model, backend, stream, sess=conv_key, aux=aux, key_src=key_src)

    if backend == "openai":
        real = model.split("/", 1)[1] if "/" in model else OPENAI_MODEL
        if not OPENAI_API_KEY:
            if DEFAULT_LOCAL_ON_ERROR:
                payload = dict(body); payload["model"] = MODEL_GENERAL
                return StreamingResponse(stream_ollama("/api/generate", payload), media_type="application/x-ndjson") if stream \
                       else await nonstream_ollama("/api/generate", payload)
            return JSONResponse({"error": "OPENAI_API_KEY not set"}, status_code=400)
        messages = [{"role": "user", "content": prompt}]
        return StreamingResponse(stream_openai_chat_as_ndjson(messages, real), media_type="application/x-ndjson") if stream \
               else await nonstream_openai_chat(messages, real)

    payload = dict(body); payload["model"] = model
    return StreamingResponse(stream_ollama("/api/generate", payload), media_type="application/x-ndjson") if stream \
           else await nonstream_ollama("/api/generate", payload)

def _virtual_tag(name: str) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {"name": name, "model": name, "modified_at": now, "size": 0, "digest": f"virtual-{name}",
            "details": {"parameter_size": "virtual", "quantization_level": "n/a"}}

def _configured_models_set() -> set[str]:
    s = set()
    for m in (MODEL_DEV, MODEL_WRITING, MODEL_GENERAL):
        if m: s.add(m)
    return s

def _tag_from_upstream(name: str, upstream_models: List[Dict[str, Any]]) -> Dict[str, Any]:
    for it in upstream_models:
        if it.get("name") == name or it.get("model") == name:
            it = dict(it)
            it["name"] = name
            it["model"] = name
            return it
    return {
        "name": name,
        "model": name,
        "modified_at": datetime.now(timezone.utc).isoformat(),
        "size": 0,
        "digest": f"configured-{name}",
        "details": {"parameter_size": "unknown", "quantization_level": "unknown"}
    }

@app.get("/api/tags")
async def api_tags_get():
    r = await client.get(f"{OLLAMA_BASE}/api/tags")
    try:
        upstream = r.json().get("models", [])
    except Exception:
        upstream = []
    conf = _configured_models_set()
    result: List[Dict[str, Any]] = []
    for name in conf:
        result.append(_tag_from_upstream(name, upstream))
    result.append(_virtual_tag("auto:latest"))
    return JSONResponse({"models": result})

@app.post("/api/tags")
async def api_tags_post():
    return await api_tags_get()

@app.get("/api/ps")
async def api_ps_get():
    r = await client.get(f"{OLLAMA_BASE}/api/ps")
    try:
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception:
        return Response(content=r.content, media_type=r.headers.get("content-type", "application/json"),
                        status_code=r.status_code)

@app.post("/api/ps")
async def api_ps_post():
    return await api_ps_get()

@app.get("/api/version")
async def api_version_get():
    r = await client.get(f"{OLLAMA_BASE}/api/version")
    try:
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception:
        return Response(content=r.content, media_type=r.headers.get("content-type", "application/json"),
                        status_code=r.status_code)

@app.post("/api/show")
async def api_show(req: Request):
    body = await req.json()
    name = body.get("name") or body.get("model")
    if not name:
        return JSONResponse({"error": "missing name"}, status_code=400)
    resolved = resolve_virtual_model(name, intent_hint="general") if is_virtual_model(name) else name
    r = await client.post(f"{OLLAMA_BASE}/api/show", json={"name": resolved})
    try:
        data = r.json()
    except Exception:
        return Response(content=r.content, media_type=r.headers.get("content-type", "application/json"),
                        status_code=r.status_code)
    data["name"] = name
    data["model"] = name
    return JSONResponse(data, status_code=200)

@app.get("/debug/echo-headers")
async def echo_headers(req: Request):
    return JSONResponse({"headers": {k: v for k, v in req.headers.items()}})

@app.get("/healthz")
async def healthz():
    return JSONResponse({
        "ok": True,
        "ollama": OLLAMA_BASE,
        "models": {"dev": MODEL_DEV, "writing": MODEL_WRITING, "general": MODEL_GENERAL, "classifier": CLASSIFIER_MODEL},
        "openai": bool(OPENAI_API_KEY),
        "config": CFG_PATH,
        "log_headers": LOG_HEADERS
    })

@app.on_event("shutdown")
async def on_shutdown():
    await client.aclose()

"""Chat endpoints for the ClimAID assistant in the browser interface.

The same rule-based assistant as `climaid ai` (climaid.assistant), one per browser session.
Runs can take minutes, so each message is handled in a background thread: the page posts a
message, then polls for new replies.

    POST /assistant/message   {"text": "..."}          -> {"accepted": true}
    POST /assistant/upload    file + kind              -> uploaded file is passed to the assistant
    GET  /assistant/messages?after=N                   -> {"messages": [...], "busy": bool}
    POST /assistant/reset
"""
from __future__ import annotations

import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

from climaid.assistant import Assistant, Runner

from .api import REPORT_DIR

router = APIRouter(prefix="/assistant")

ASSISTANT_REPORTS = REPORT_DIR / "assistant"
MAX_SESSIONS = 50
UPLOAD_KINDS = {"disease": ("my data file is", (".csv", ".xlsx", ".xls")),
                "weather": ("my climate data file is", (".csv", ".xlsx", ".xls", ".parquet")),
                "projection": ("my projection data file is", (".csv", ".xlsx", ".xls", ".parquet"))}
FILE_PROMPT = ("Please upload your disease data with the 📎 button below (CSV or Excel, with a date column and a "
               "case-count column).")


class WebRunner(Runner):
    """Saves each report under /reports/assistant/<session>/ with a unique name, and leaves ClimAID's
    progress messages in the server terminal (as the rest of the dashboard does)."""

    def __init__(self, session_id: str):
        super().__init__(log_path=None)
        self.out_dir = ASSISTANT_REPORTS / session_id

    def _unique(self, result: dict, label: str) -> dict:
        path = result.get("report_path")
        if path and Path(path).exists():
            dst = self.out_dir / f"climaid_{label}_{uuid.uuid4().hex[:8]}.html"
            Path(path).replace(dst)
            result["report_path"] = str(dst)
        return result

    def forecast(self, dm, settings):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self._unique(super().forecast(dm, dict(settings, output_dir=str(self.out_dir))), "forecast")

    def project(self, dm, settings):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self._unique(super().project(dm, dict(settings, output_dir=str(self.out_dir))), "outlook")


def _report_url(path: str) -> str:
    try:
        return "/reports/" + Path(path).resolve().relative_to(REPORT_DIR.resolve()).as_posix()
    except ValueError:
        return Path(path).resolve().as_uri()


class ChatSession:
    def __init__(self, sid: str):
        self.sid = sid
        self.assistant = Assistant(runner=WebRunner(sid), report_link=_report_url,
                                   docs_base="/documentation/", file_prompt=FILE_PROMPT)
        self.messages: list[dict] = [{"role": "assistant", "text": self._intro()}]
        self.busy = False
        self.lock = threading.Lock()
        self.last_used = time.time()

    def _intro(self) -> str:
        return (self.assistant.greet()
                .replace("where your disease data file is", "upload your disease data file with the 📎 button")
                .replace('Type "help" at any time, or "quit" to leave.', 'Type "help" at any time.'))

    def add(self, role: str, text: str):
        with self.lock:
            self.messages.append({"role": role, "text": text})

    def handle(self, text: str, shown: str | None = None):
        """Answer `text` in a background thread (`shown` is what the chat displays for it)."""
        with self.lock:
            if self.busy:
                raise HTTPException(status_code=409, detail="Still working on the previous request.")
            self.busy = True
            self.messages.append({"role": "user", "text": shown or text})
        self.last_used = time.time()

        def work():
            try:
                for reply in self.assistant.respond(text):
                    self.add("assistant", reply)
            except Exception as exc:          # never leave the chat stuck
                self.add("assistant", f"Something went wrong: {type(exc).__name__}: {exc}")
            finally:
                with self.lock:
                    self.busy = False
                    if self.assistant.finished:      # "quit" in the browser just starts afresh
                        self.assistant.finished = False

        threading.Thread(target=work, daemon=True).start()


_sessions: dict[str, ChatSession] = {}
_sessions_lock = threading.Lock()


def _session(session_id: str | None) -> ChatSession:
    sid = (session_id or "").strip() or uuid.uuid4().hex
    with _sessions_lock:
        s = _sessions.get(sid)
        if s is None:
            if len(_sessions) >= MAX_SESSIONS:     # forget the least recently used idle session
                idle = [x for x in _sessions.values() if not x.busy]
                if idle:
                    _sessions.pop(min(idle, key=lambda x: x.last_used).sid, None)
            s = _sessions[sid] = ChatSession(sid)
        return s


class Message(BaseModel):
    text: str


@router.post("/message")
def post_message(msg: Message, x_climaid_session: str | None = Header(default=None, alias="X-ClimAID-Session")):
    text = msg.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty message.")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Message too long (2000 characters at most).")
    _session(x_climaid_session).handle(text)
    return {"accepted": True}


@router.get("/messages")
def get_messages(after: int = 0, x_climaid_session: str | None = Header(default=None, alias="X-ClimAID-Session")):
    s = _session(x_climaid_session)
    with s.lock:
        return {"messages": s.messages[max(0, after):], "total": len(s.messages), "busy": s.busy}


@router.post("/upload")
async def upload(file: UploadFile, kind: str = Form("disease"),
                 x_climaid_session: str | None = Header(default=None, alias="X-ClimAID-Session")):
    if kind not in UPLOAD_KINDS:
        raise HTTPException(status_code=400, detail=f"Unknown upload type {kind!r}.")
    phrase, suffixes = UPLOAD_KINDS[kind]
    name = Path(file.filename or "").name
    ext = Path(name).suffix.lower()
    if ext not in suffixes:
        raise HTTPException(status_code=400, detail=f"Use a {', '.join(suffixes)} file for this upload.")
    s = _session(x_climaid_session)
    folder = Path(tempfile.gettempdir()) / "climaid_assistant" / s.sid
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{kind}_{uuid.uuid4().hex[:8]}{ext}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    s.handle(f'{phrase} "{dest}"', shown=f"📎 Uploaded {kind} data: {name}")
    return {"accepted": True, "filename": name}


@router.post("/reset")
def reset(x_climaid_session: str | None = Header(default=None, alias="X-ClimAID-Session")):
    sid = (x_climaid_session or "").strip()
    with _sessions_lock:
        old = _sessions.get(sid)
        if old is not None and old.busy:
            raise HTTPException(status_code=409, detail="Still working; try again when the run has finished.")
        _sessions.pop(sid, None)
    _session(sid)
    return {"reset": True}

from __future__ import annotations
"""
SSE streaming micro-server for token output (FastAPI/Starlette). Optional dependency.

Run:
    uvicorn wee.stream:app --reload

Then GET /stream?text=hello

If FastAPI isn't installed, pip install:
    pip install fastapi uvicorn
"""
import asyncio
from typing import AsyncGenerator
try:
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
except Exception as e:
    FastAPI = None
    StreamingResponse = None

async def token_generator(text: str) -> AsyncGenerator[bytes, None]:
    for i, ch in enumerate(text):
        await asyncio.sleep(0.05)
        yield f"data: {ch}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"

if FastAPI is not None:
    app = FastAPI()

    @app.get("/stream")
    async def stream(text: str = "hello world"):
        return StreamingResponse(token_generator(text), media_type="text/event-stream")
else:
    app = None

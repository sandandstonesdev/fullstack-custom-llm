"""Pydantic models for WebSocket chat messages.

These match the frontend contract exactly (see frontend/src/schemas/message.schema.ts):
- client sends: {"text": string, "isUser": bool}
- server replies: {"text": string, "isUser": false} or {"error": string, "isUser": false}
"""
from pydantic import BaseModel


class ChatMessage(BaseModel):
    text: str
    isUser: bool = False


class ChatError(BaseModel):
    error: str
    isUser: bool = False

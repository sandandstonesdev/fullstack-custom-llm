"""
FastAPI serving layer for the Polish LLM chat app.

Model + tokenizer load once at startup via lifespan and are never written
to. Serving knobs (CORS origins, host/port, generation) come from configs
(env overridable). Generation is CPU-bound: it runs in a worker thread
(asyncio.to_thread, torch releases the GIL) under a lock, so concurrent
clients don't interleave. Redis/Valkey publishing is async, fire-and-forget.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.broker import EventBroker
from app.schemas import ChatError, ChatMessage
from configs import API_HOST, API_PORT, CORS_ORIGINS, GEN_MAX_TOKENS
from model.inference import generate, load_trained_model

state = {"model": None, "tokenizer": None, "broker": None, "gen_lock": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, tokenizer = load_trained_model()
    if model is None:
        print("WARNING: starting without a model; /ws will echo messages")
    state["model"], state["tokenizer"] = model, tokenizer
    state["broker"] = EventBroker()
    state["gen_lock"] = asyncio.Lock()
    try:
        yield
    finally:
        await state["broker"].close()


app = FastAPI(lifespan=lifespan)

# allow_credentials=False: wildcard origins are incompatible with
# credentialed requests per the CORS spec, and this app uses no cookies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)


manager = ConnectionManager()


async def generate_reply(prompt: str) -> str:
    """Generate a bot reply off the event loop, one request at a time."""
    async with state["gen_lock"]:
        return await asyncio.to_thread(
            generate,
            state["model"],
            state["tokenizer"],
            prompt,
            max_new_tokens=GEN_MAX_TOKENS,
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()

            try:
                message = ChatMessage.model_validate_json(data)
            except Exception as exc:
                await manager.send_personal_message(
                    ChatError(error=f"Invalid message: {exc}").model_dump_json(),
                    websocket,
                )
                continue

            # Observability event (fire-and-forget)
            await state["broker"].publish(message.model_dump())

            if state["model"] is not None and state["tokenizer"] is not None:
                try:
                    bot_response = await generate_reply(message.text)
                    bot_message = ChatMessage(text=bot_response, isUser=False)
                    await websocket.send_text(bot_message.model_dump_json())
                    await state["broker"].publish(bot_message.model_dump())
                except Exception as exc:
                    print(f"Error generating model response: {exc}")
                    await websocket.send_text(
                        ChatMessage(
                            text="Sorry, I encountered an error generating a response.",
                            isUser=False,
                        ).model_dump_json()
                    )
            else:
                # No model available: relay to other clients (testing mode)
                for connection in manager.active_connections:
                    if connection != websocket:
                        await connection.send_text(message.model_dump_json())
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/healthz")
def healthz():
    """Liveness/readiness probe for k8s."""
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/")
def read_root():
    return {"message": "WebSocket server is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)

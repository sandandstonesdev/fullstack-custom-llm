from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import redis
import json
import os
from typing import List
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load the language model
try:
    from model.inference import generate, load_trained_model
    model, tokenizer = load_trained_model()
    print("✅ Language model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load language model: {e}")
    model, tokenizer = None, None

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis/Valkey client
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received raw data: {data}")  # Debug: Log raw data
            
            # Parse the incoming message
            try:
                message = json.loads(data)
                print(f"Parsed message: {message}")  # Debug: Log parsed message
                
                # Validate message structure
                if not isinstance(message, dict) or "text" not in message:
                    await manager.send_personal_message(
                        json.dumps({
                            "error": "Message must contain 'text' field",
                            "isUser": False
                        }),
                        websocket
                    )
                    continue
                    
                # Set default isUser value if not provided
                message["isUser"] = message.get("isUser", False)
                
                # Publish the message to Valkey
                redis_client.publish("chat_messages", json.dumps(message))
                
                # Generate response from language model if available
                if model is not None and tokenizer is not None:
                    try:
                        # Generate response from the model
                        prompt = message["text"]
                        bot_response = generate(model, tokenizer, prompt, max_new_tokens=50)
                        
                        # Create bot response message
                        bot_message = {
                            "text": bot_response,
                            "isUser": False
                        }
                        
                        # Send bot response back to the sender
                        await websocket.send_text(json.dumps(bot_message))
                        
                        # Also publish to Valkey
                        redis_client.publish("chat_messages", json.dumps(bot_message))
                        
                    except Exception as e:
                        print(f"❌ Error generating model response: {e}")
                        await websocket.send_text(json.dumps({
                            "text": "Sorry, I encountered an error generating a response.",
                            "isUser": False
                        }))
                else:
                    # If model not available, just echo the message (for testing)
                    print("⚠️  Language model not available, echoing message")
                    for connection in manager.active_connections:
                        if connection != websocket:  # Don't echo back to sender
                            await connection.send_text(json.dumps(message))
                        
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")  # Debug: Log JSON errors
                await manager.send_personal_message(
                    json.dumps({
                        "error": f"Invalid JSON format: {str(e)}",
                        "isUser": False
                    }),
                    websocket
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")  # Debug: Log disconnections

@app.get("/")
def read_root():
    return {"message": "WebSocket server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
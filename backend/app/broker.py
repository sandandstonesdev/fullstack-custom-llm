"""Valkey/Redis integration for the API.

One role only: structured logging/observability. The WebSocket handler
generates responses directly; prompt/reply events are published to a
channel so anything that wants to observe chat traffic can subscribe.

Fully async (redis.asyncio) so publishing never blocks the event loop.
If Valkey is down, serving continues - publish failures are logged and
swallowed. Do not add queueing/worker logic here unless the architecture
changes (see project README).
"""
import json

from redis import asyncio as aioredis

from configs import VALKEY_URL


class EventBroker:
    """Publishes chat events to a Valkey/Redis channel (fire-and-forget)."""

    def __init__(self, url=VALKEY_URL, channel="chat_messages"):
        self.channel = channel
        self.client = aioredis.Redis.from_url(url, decode_responses=True)

    async def publish(self, event: dict):
        """Publish an event; never raises."""
        try:
            await self.client.publish(self.channel, json.dumps(event))
        except Exception as exc:
            print(f"broker: publish failed ({exc}); event dropped")

    async def close(self):
        """Close the connection pool on shutdown; never raises."""
        try:
            await self.client.aclose()
        except Exception:
            pass

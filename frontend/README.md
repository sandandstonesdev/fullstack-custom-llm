# Chatbot Frontend with Astro, Tailwind CSS, and Valkey

A modern chatbot interface built with Astro, styled with Tailwind CSS, and powered by Valkey (Redis) for real-time communication.

# Frontend - Polish LLM Chat

## Quick Start

```bash
pnpm install
pnpm run dev
```

Opens at: `http://localhost:3000`

## Configuration

Create `.env` file:
```
VITE_WEBSOCKET_URL=ws://localhost:8000/ws
```

## Commands

- `pnpm install` - Install dependencies
- `pnpm run dev` - Start dev server
- `pnpm run build` - Production build
- `pnpm test` - Run tests

## Podman

```bash
podman build -t polish-llm-frontend:latest .
podman run -p 3000:80 polish-llm-frontend:latest
```

## Kubernetes

```bash
kubectl apply -f ../deploy/k8s/
```

## Structure

- `src/components/ChatWindow.jsx` - Main chat component
- `src/pages/index.astro` - Main page
- `../deploy/k8s/` - Kubernetes manifests (repo root)

## Features

- Real-time chat with WebSocket
- Polish language model integration
- Clean chat (🗑️ button)
- Responsive design with Tailwind CSS

## License
MIT
# Chatbot Frontend with Astro, Tailwind CSS, and Valkey

A modern chatbot interface built with Astro, styled with Tailwind CSS, and powered by Valkey (Redis) for real-time communication.

## Features

- Real-time chat interface with Valkey (Redis) integration
- Responsive design with Tailwind CSS
- Auto-scrolling chat messages
- Clean, modern UI with message bubbles
- About page with project information

## Prerequisites

- Node.js (v16 or later)
- pnpm (recommended)
- Valkey (Redis) server running on `localhost:6379`
- Backend API that publishes/receives messages from Valkey

## Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies using pnpm:
   ```bash
   pnpm install
   ```

   If you don't have pnpm installed:
   ```bash
   npm install -g pnpm
   ```

## Development

1. Start Valkey (Redis) server:
```bash
valkey-server
# or
redis-server
```

2. Start the development server:
```bash
pnpm dev
```

The app will be available at `http://localhost:3000`

## Configuration

### Tailwind CSS

Tailwind is configured in `tailwind.config.js`. You can customize the theme, colors, and plugins as needed.

## Project Structure

```
frontend/
├── src/
│   ├── components/       # Reusable components
│   │   ├── Header.astro
│   │   ├── ChatMessage.astro
│   │   └── ChatbotWindowFixed.astro
│   ├── pages/            # Page components
│   │   ├── index.astro   # Main chat page
│   │   └── about.astro   # About page
│   └── styles/           # Global styles
│       └── global.css
├── package.json
├── astro.config.mjs      # Astro configuration
├── tailwind.config.js    # Tailwind configuration
└── README.md
```

## Components

### ChatbotWindowFixed.astro

The main chat interface component with:
- Message display with auto-scrolling
- Input form for sending messages
- Connection status handling

### ChatMessage.astro

Displays individual chat messages with:
- Different styling for user vs bot messages
- Responsive layout

### Header.astro

Navigation header with links to:
- Home (chat interface)
- About page

## Styling

The app uses Tailwind CSS for styling:
- Responsive design
- Modern color scheme
- Custom scrollbars for chat messages
- Focus states for accessibility

## Deployment

Build the app for production:
```bash
pnpm build
```

The output will be in the `dist/` directory.

Note: This project uses `output: "static"` mode, which generates static HTML files. For server-side rendering or hybrid rendering, you would need to install an appropriate adapter.

## Troubleshooting

### Valkey Connection Issues

If you see connection errors in the console:
1. Ensure Valkey (Redis) server is running
2. Check the connection settings in `src/utils/valkeyClient.js`
3. Verify your Valkey server is accessible on the configured port
4. Check your network connection

### Tailwind Not Working

If Tailwind styles aren't applying:
1. Ensure you've installed all dependencies
2. Check that `tailwind.config.js` includes all necessary file paths
3. Verify that `global.css` imports Tailwind directives

## License

MIT
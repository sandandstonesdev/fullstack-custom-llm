import React, { useState, useEffect, useRef } from 'react';

const ChatWindow = ({ initialMessages }) => {
  const [messages, setMessages] = useState(initialMessages);
  const [inputValue, setInputValue] = useState('');
  const [connected, setConnected] = useState(false);
  const socketRef = useRef(null);
  const messagesEndRef = useRef(null);
  // Message queued while the socket is still connecting.
  const pendingMessageRef = useRef(null);

  useEffect(() => {
    // Initialize WebSocket connection
    const websocketUrl = import.meta.env.VITE_WEBSOCKET_URL || 'ws://localhost:8000/ws';
    const socket = new WebSocket(websocketUrl);
    socketRef.current = socket;

    socket.onopen = () => {
      setConnected(true);
      // Flush any message that was submitted while connecting
      if (pendingMessageRef.current) {
        socket.send(JSON.stringify(pendingMessageRef.current));
        pendingMessageRef.current = null;
      }
    };

    // Handle incoming messages
    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        // Ignore malformed frames (e.g. error payloads without text)
        if (typeof message.text === 'string') {
          setMessages((prevMessages) => [...prevMessages, message]);
        }
      } catch (error) {
        console.error('Error parsing message:', error);
      }
    };

    // Handle WebSocket errors
    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    // Handle WebSocket close
    socket.onclose = () => {
      setConnected(false);
      console.log('WebSocket connection closed');
    };

    // Clean up WebSocket connection on component unmount
    return () => {
      socketRef.current = null;
      pendingMessageRef.current = null;
      socket.close();
    };
  }, []);

  useEffect(() => {
    // Scroll to bottom when messages change
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const text = inputValue.trim();
    if (!text) return;

    const socket = socketRef.current;
    if (!socket || socket.readyState === WebSocket.CLOSING || socket.readyState === WebSocket.CLOSED) {
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: 'Brak połączenia z serwerem — odśwież stronę.', isUser: false },
      ]);
      return;
    }

    const userMessage = { text, isUser: true };
    // Add user message to state immediately
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    // Send only on an open socket; queue while still connecting
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(userMessage));
    } else {
      pendingMessageRef.current = userMessage;
    }
    setInputValue('');
  };

  const handleCleanChat = () => {
    // Reset messages to initial state
    setMessages(initialMessages);
  };

  return (
    <div className="flex flex-col h-[500px] bg-white rounded-lg shadow-lg overflow-hidden">
      {/* Chat messages */}
      <div
        id="chat-messages"
        className="flex-1 p-4 overflow-y-auto"
        aria-live="polite"
        aria-atomic="true"
      >
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`mb-2 ${msg.isUser ? 'text-right' : 'text-left'}`}
          >
            <div
              className={`inline-block p-2 rounded-lg ${
                msg.isUser ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-800'
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Chat header with clean button */}
      <div className="border-b p-4 bg-white flex justify-between items-center">
        <h2 className="text-lg font-semibold text-gray-800">Chat</h2>
        <button
          onClick={handleCleanChat}
          className="px-3 py-1 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-500"
          aria-label="Clean chat"
          title="Clean chat"
        >
          🗑️ Clean
        </button>
      </div>

      {/* Input area */}
      <div className="border-t p-4 bg-gray-50">
        <form
          id="chat-form"
          className="flex space-x-2"
          onSubmit={handleSubmit}
        >
          <input
            type="text"
            name="message"
            placeholder="Type your message..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            aria-label="Message input"
            required
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
          />
          <button
            type="submit"
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            aria-label="Send message"
            title={connected ? 'Send message' : 'Connecting to server...'}
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatWindow;
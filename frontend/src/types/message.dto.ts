// src/types/message.dto.ts
// Data Transfer Objects (DTOs) for messages

export interface UserMessage {
  text: string;
  isUser: true;
}

export interface BotMessage {
  text: string;
  isUser: false;
}

export type Message = UserMessage | BotMessage;
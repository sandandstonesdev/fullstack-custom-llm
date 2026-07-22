// src/schemas/message.schema.ts
// Zod schemas for message validation

import { z } from 'zod';

// Schema for a user message
const UserMessageSchema = z.object({
  text: z.string().min(1, 'Message text cannot be empty'),
  isUser: z.literal(true),
});

// Schema for a bot message
const BotMessageSchema = z.object({
  text: z.string().min(1, 'Message text cannot be empty'),
  isUser: z.literal(false),
});

// Union schema for any message
export const MessageSchema = z.union([UserMessageSchema, BotMessageSchema]);

// Type for a message
export type Message = z.infer<typeof MessageSchema>;
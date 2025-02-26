
"use client"

import { useState } from 'react';
import { Paper,   Title, Stack, TextInput, Button, ScrollArea, Text, Box } from '@mantine/core';

interface Message {
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      content: "Hello! I'm your salary prediction assistant. How can I help you today?",
      sender: 'bot',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    const userMessage: Message = {
      content: input,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');

    // Simulate bot response
    setTimeout(() => {
      const botMessage: Message = {
        content: "I'm processing your request. This is a placeholder response.",
        sender: 'bot',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, botMessage]);
    }, 1000);
  };

  return (
    <Paper shadow="xs" p="md" style={{ height: '100%' }}>
      <Stack >
        <Title order={3} mb="lg">Salary Expert AI</Title>

        <ScrollArea style={{ flex: 1 }} mb="md">
          <Stack gap="md">
            {messages.map((message, index) => (
              <Box
                key={index}
                style={{
                  alignSelf: message.sender === 'user' ? 'flex-end' : 'flex-start',
                  maxWidth: '80%',
                }}
              >
                <Paper
                  p="xs"
                  bg={message.sender === 'user' ? 'blue.9' : 'gray.9'}
                  style={{
                    borderRadius: '8px',
                  }}
                >
                  <Text size="sm">{message.content}</Text>
                </Paper>
                <Text size="xs" c="dimmed" ta={message.sender === 'user' ? 'right' : 'left'}>
                  {message.timestamp.toLocaleTimeString()}
                </Text>
              </Box>
            ))}
          </Stack>
        </ScrollArea>

        <form onSubmit={handleSubmit} style={{ marginTop: 'auto' }}>
          <div style={{ display: 'flex', gap: '8px' }}>
            <TextInput
              placeholder="Type your message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              style={{ flex: 1 }}
            />
            <Button
              type="submit"
              variant="subtle"
              color="blue"
              style={{ padding: '0 8px' }}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="m3 3 3 9-3 9 19-9Z" />
              </svg>
            </Button>
          </div>
        </form>
      </Stack>
    </Paper>
  );
}

"use client"

import { ActionIcon, Box, Button, Code, Group, List, Paper, Popover, ScrollArea, Slider, Stack, Text, TextInput, Title } from '@mantine/core';
import { IconInfoCircle, IconSettings } from '@tabler/icons-react';
import { useEffect, useRef, useState } from 'react';
import Markdown from 'react-markdown';
import { streamGeminiResponse } from './gemini';

interface Message {
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

interface ModelConfig {
  propertyMarketFactors: {
    interestRate: number;
    marketGrowth: number;
    inflationRate: number;
  };
  locationFactors: {
    urbanPremium: number;
    suburbanDiscount: number;
    ruralDiscount: number;
  };
}

interface ChatProps {
  predictedPrice: number | null;
  duration: number;
  propertyDetails: {
    suburb: string;
    rooms: number | null;
    type: string;
    method: string;
    seller: string;
    distance: number | null;
    bathroom: number | null;
    car: number | null;
    landsize: number | null;
    buildingArea: number | null;
    propertyAge: number | null;
    direction: string;
    landSizeNotOwned: boolean;
  };
  triggerInitialAnalysis?: boolean;
  isBatchMode?: boolean; // Add this prop
}

export default function Chat({ predictedPrice, duration, propertyDetails, triggerInitialAnalysis, isBatchMode }: ChatProps) {
  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    propertyMarketFactors: {
      interestRate: 5.5,
      marketGrowth: 3.2,
      inflationRate: 2.8
    },
    locationFactors: {
      urbanPremium: 15,
      suburbanDiscount: 5,
      ruralDiscount: 12
    }
  });
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (scrollAreaRef.current) {
      const scrollElement = scrollAreaRef.current.querySelector('.mantine-ScrollArea-viewport');
      if (scrollElement) {
        scrollElement.scrollTop = scrollElement.scrollHeight;
      }
    }
  };

  // Modify initial message to acknowledge batch mode
  const [messages, setMessages] = useState<Message[]>([
    {
      content:
        `# 👋 Hello!

I'm your property price prediction assistant. I can help you understand the property price prediction for **${propertyDetails.suburb || 'your property'}**.
${isBatchMode ? '\n*This property is part of a batch analysis.*' : ''}

Here are some things you can ask me about:
- Property price analysis and comparisons
- Location insights and neighborhood information
- Market trends and property valuation factors
- Home buying and selling tips

Feel free to ask any questions!`,
      sender: 'bot',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');

  useEffect(() => {
    scrollToBottom();
  }, [messages]); // Scroll when messages change

  useEffect(() => {
    if (triggerInitialAnalysis && predictedPrice) {
      const initialPrompt = isBatchMode
        ? "Please analyze this property from a batch of properties. The user is analyzing multiple properties at once.\
           Provide a concise yet comprehensive assessment of this specific property.\
           Include insights about location value, property features, and how this property compares to typical properties in the area.\
           At the end, provide 2-3 brief suggestions for improving property value.\
           Start with 'Let me analyze this specific property from your batch...'"
        : "Please analyze the property price prediction and provide a comprehensive assessment.\
           Include insights about location value, property features, and current market conditions.\
           Use numbers, percentages and math to explore and illustrate your point.\
           Compare with typical property values in the area and discuss potential investment value.\
           At the end, provide 3 suggestions for improving property value, and ask 3 questions to know more about the user's goals.\
           Start with 'Hi there! I saw you just made a property price prediction! ...'";

      handleSubmit(new Event('submit') as any, initialPrompt);
    }
  }, [triggerInitialAnalysis, predictedPrice]);

  const handleSubmit = async (e: React.FormEvent, overrideInput?: string) => {
    e.preventDefault();
    const messageText = overrideInput || input;
    if (!messageText.trim()) return;

    // Create new messages array with both user and processing messages
    const newMessages = [...messages];

    // Only add user message if it's not an initial analysis
    if (!overrideInput) {
      newMessages.push({
        content: messageText,
        sender: 'user',
        timestamp: new Date(),
      });
    }

    // Add processing message
    newMessages.push({
      content: "Thinking...",
      sender: 'bot',
      timestamp: new Date(),
    });

    // Update messages state once with both new messages
    setMessages(newMessages);
    setInput(''); // Clear input after sending

    try {
      // @ts-expect-error
      const stream = streamGeminiResponse(messageText, { price: predictedPrice, duration }, propertyDetails, messages, modelConfig);

      for await (const chunk of stream) {
        setMessages(prev => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1] = {
            content: chunk,
            sender: 'bot',
            timestamp: new Date(),
          };
          return newMessages;
        });
      }
    } catch (error) {
      setMessages(prev => [
        ...prev.slice(0, -1),
        {
          content: "I apologize, but I encountered an error. Please try again.",
          sender: 'bot',
          timestamp: new Date(),
        },
      ]);
    }
  };

  return (
    <Paper
      shadow="sm"
      pl="lg"
      radius="md"
      style={{
        height: '100%',
      }}
    >
      <Stack>
        <Title order={3} mb="lg">
          <Group justify="space-between" style={{ width: '100%' }}>
            <Group>
              <span>🏠</span>
              <span>Property Expert AI</span>
            </Group>
            <Popover width={400} position="bottom-end" shadow="md">
              <Popover.Target>
                <ActionIcon variant="subtle" size="md">
                  <IconSettings style={{ width: '70%', height: '70%' }} stroke={1.5} />
                </ActionIcon>
              </Popover.Target>
              <Popover.Dropdown>
                <Stack p={16}>
                  <Title order={4}>Model Configuration</Title>

                  <Box>
                    <Group gap={4}>
                      <Text size="sm" fw={500}>Property Market Factors</Text>
                      <Popover width={250} position="bottom" shadow="md">
                        <Popover.Target>
                          <IconInfoCircle size={16} style={{ color: 'gray', cursor: 'pointer' }} />
                        </Popover.Target>
                        <Popover.Dropdown>
                          <Text size="sm">
                            Current market factors affecting property prices:
                            - Interest rates affect mortgage affordability
                            - Market growth rate indicates appreciation potential
                            - Inflation affects real property value over time
                          </Text>
                        </Popover.Dropdown>
                      </Popover>
                    </Group>
                    <Stack>
                      <Box>
                        <Text size="xs" mb={8}>Interest Rate: {modelConfig.propertyMarketFactors.interestRate.toFixed(1)}%</Text>
                        <Slider
                          value={modelConfig.propertyMarketFactors.interestRate}
                          onChange={(val) => setModelConfig(prev => ({
                            ...prev,
                            propertyMarketFactors: { ...prev.propertyMarketFactors, interestRate: val }
                          }))}
                          min={0}
                          max={10}
                          step={0.1}
                          label={(value) => `${value.toFixed(1)}%`}
                        />
                      </Box>
                      <Box>
                        <Text size="xs" mb={8}>Market Growth: {modelConfig.propertyMarketFactors.marketGrowth.toFixed(1)}%</Text>
                        <Slider
                          value={modelConfig.propertyMarketFactors.marketGrowth}
                          onChange={(val) => setModelConfig(prev => ({
                            ...prev,
                            propertyMarketFactors: { ...prev.propertyMarketFactors, marketGrowth: val }
                          }))}
                          min={-5}
                          max={10}
                          step={0.1}
                          label={(value) => `${value.toFixed(1)}%`}
                        />
                      </Box>
                      <Box>
                        <Text size="xs" mb={8}>Inflation Rate: {modelConfig.propertyMarketFactors.inflationRate.toFixed(1)}%</Text>
                        <Slider
                          value={modelConfig.propertyMarketFactors.inflationRate}
                          onChange={(val) => setModelConfig(prev => ({
                            ...prev,
                            propertyMarketFactors: { ...prev.propertyMarketFactors, inflationRate: val }
                          }))}
                          min={0}
                          max={10}
                          step={0.1}
                          label={(value) => `${value.toFixed(1)}%`}
                        />
                      </Box>
                    </Stack>
                  </Box>

                  <Box>
                    <Group gap={4}>
                      <Text size="sm" fw={500}>Location Value Factors (%)</Text>
                      <Popover width={280} position="bottom" shadow="md">
                        <Popover.Target>
                          <IconInfoCircle size={16} style={{ color: 'gray', cursor: 'pointer' }} />
                        </Popover.Target>
                        <Popover.Dropdown>
                          <Text size="sm">
                            These factors represent how much location type affects property value:
                            - Urban Premium: Added value for city properties (%)
                            - Suburban Discount: Value reduction for suburban areas (%)
                            - Rural Discount: Value reduction for rural properties (%)
                          </Text>
                        </Popover.Dropdown>
                      </Popover>
                    </Group>
                    <Stack>
                      <Box>
                        <Text size="xs" mb={8}>Urban Premium: {modelConfig.locationFactors.urbanPremium}%</Text>
                        <Slider
                          value={modelConfig.locationFactors.urbanPremium}
                          onChange={(val) => setModelConfig(prev => ({
                            ...prev,
                            locationFactors: { ...prev.locationFactors, urbanPremium: val }
                          }))}
                          min={0}
                          max={50}
                          step={1}
                          label={(value) => `${value}%`}
                        />
                      </Box>
                      <Box>
                        <Text size="xs" mb={8}>Suburban Discount: {modelConfig.locationFactors.suburbanDiscount}%</Text>
                        <Slider
                          value={modelConfig.locationFactors.suburbanDiscount}
                          onChange={(val) => setModelConfig(prev => ({
                            ...prev,
                            locationFactors: { ...prev.locationFactors, suburbanDiscount: val }
                          }))}
                          min={0}
                          max={50}
                          step={1}
                          label={(value) => `${value}%`}
                        />
                      </Box>
                      <Box>
                        <Text size="xs" mb={8}>Rural Discount: {modelConfig.locationFactors.ruralDiscount}%</Text>
                        <Slider
                          value={modelConfig.locationFactors.ruralDiscount}
                          onChange={(val) => setModelConfig(prev => ({
                            ...prev,
                            locationFactors: { ...prev.locationFactors, ruralDiscount: val }
                          }))}
                          min={0}
                          max={50}
                          step={1}
                          label={(value) => `${value}%`}
                        />
                      </Box>
                    </Stack>
                  </Box>
                </Stack>
              </Popover.Dropdown>
            </Popover>
          </Group>
        </Title>

        <ScrollArea
          ref={scrollAreaRef}
          h={600}
          mb="md"
          type="always"
          scrollbarSize={8}
          offsetScrollbars
        >
          <Stack gap="md">
            {messages.map((message: Message, index: number) => (
              <Box
                key={index}
                style={{
                  alignSelf: message.sender === 'user' ? 'flex-end' : 'flex-start',
                  maxWidth: '80%',
                  willChange: 'transform',
                  transform: 'translateZ(0)',
                }}
              >
                <Paper
                  p="sm"
                  bg={message.sender === 'user' ? 'blue.6' : 'dark.5'}
                  style={{
                    borderRadius: '12px',
                    borderBottomRightRadius: message.sender === 'user' ? '4px' : '12px',
                    borderBottomLeftRadius: message.sender === 'bot' ? '4px' : '12px',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                  }}
                >
                  <Box style={{ color: message.sender === 'user' ? 'white' : 'inherit' }}>
                    <Markdown
                      components={{
                        p: ({ children }) => <Text size="md" mb="xs" style={{ lineHeight: 1.6 }}>{children}</Text>,
                        h1: ({ children }) => <Title order={1} size="lg" mb="md" style={{ fontSize: '1.5rem' }}>{children}</Title>,
                        h2: ({ children }) => <Title order={2} size="md" mb="sm" style={{ fontSize: '1.25rem' }}>{children}</Title>,
                        h3: ({ children }) => <Title order={3} size="sm" mb="xs" style={{ fontSize: '1.1rem' }}>{children}</Title>,
                        code: ({ children }) => <Code>{children}</Code>,
                        ul: ({ children }) => <List spacing="xs" mb="sm" size="md">{children}</List>,
                        ol: ({ children }) => <List type="ordered" spacing="xs" mb="sm" size="md">{children}</List>,
                        li: ({ children }) => <List.Item style={{ color: 'inherit', display: 'list-item', fontSize: '1rem' }}>{children}</List.Item>,
                        strong: ({ children }) => <Text span fw={700} size="md" style={{ color: 'inherit' }}>{children}</Text>,
                        em: ({ children }) => <Text span fs="italic" size="md" style={{ color: 'inherit' }}>{children}</Text>,
                      }}
                    >
                      {message.content}
                    </Markdown>
                  </Box>
                </Paper>
                <Text
                  size="xs"
                  c="dimmed"
                  ta={message.sender === 'user' ? 'right' : 'left'}
                  mt={4}
                  style={{ opacity: 0.7 }}
                >
                  {message.sender === 'bot' ? '🤖 ' : '👤 '}
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </Text>
              </Box>
            ))}
          </Stack>
        </ScrollArea>

        <form onSubmit={handleSubmit} style={{ marginTop: 'auto' }}>
          <div style={{ display: 'flex', gap: '8px' }}>
            <TextInput
              placeholder="Ask me anything about the property prediction..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              style={{ flex: 1 }}
              radius="xl"
              size="md"
            />
            <Button
              type="submit"
              variant="filled"
              color="blue"
              radius="xl"
              size="md"
              style={{ padding: '0 20px' }}
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


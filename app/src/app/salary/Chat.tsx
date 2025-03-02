
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
  medianSalary: {
    US: number;
    SG: number;
    IN: number;
  };
  costOfLiving: {
    US: number;
    SG: number;
    IN: number;
  };
}

interface ChatProps {
  predictions: {
    [key: string]: {
      salary: number | null;
      duration: number;
      relativeDuration: number;
    };
  };
  triggerInitialAnalysis?: boolean;
  jobDetails: {
    job_title: string;
    job_description: string;
    contract_type: string;
    education_level: string;
    seniority: string;
    min_years_experience: string;
    countries: string[];
    location_us: string[];
    location_in: string[];
  };
}

export default function Chat({ predictions, jobDetails, triggerInitialAnalysis }: ChatProps) {
  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    medianSalary: {
      US: 63795,
      SG: 49287,
      IN: 3900
    },
    costOfLiving: {
      US: 13998,
      SG: 13576,
      IN: 4015
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


  const [messages, setMessages] = useState<Message[]>([
    {
      content:
        `# 👋 Hello!

I'm your salary prediction assistant. I can help you understand the salary predictions for **${jobDetails.job_title || 'your job search'}**.

Here are some things you can ask me about:
- Salary predictions and analysis
- Job market insights
- Negotiation tips
- Career advice

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
    if (triggerInitialAnalysis) {
      const initialPrompt = "Please analyze the salary predictions and provide a comprehensive recommendation.\
        Include insights about regional differences, market competitiveness, and any notable patterns in the data.\
        Use numbers, percentages and math to explore and illustrate your point.\
        At the end, provide a recommended location, and ask 3 questions to know more about the user to tailor your response.\
        Start with 'Hi there! I saw you just made a prediction! ...'";
      handleSubmit(new Event('submit') as any, initialPrompt);
    }
  }, [triggerInitialAnalysis, predictions]);
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
      const stream = streamGeminiResponse(messageText, predictions, jobDetails, messages, modelConfig);

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
      <Stack >
        <Title order={3} mb="lg">
          <Group justify="space-between" style={{ width: '100%' }}>
            <Group>
              <span>💰</span>
              <span>Salary Expert AI</span>
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
                      <Text size="sm" fw={500}>Median Salary (USD)</Text>
                      <Popover width={220} position="bottom" shadow="md">
                        <Popover.Target>
                          <IconInfoCircle size={16} style={{ color: 'gray', cursor: 'pointer' }} />
                        </Popover.Target>
                        <Popover.Dropdown>
                          <Text size="sm">
                            US: $63,795 (<a href="https://www.sofi.com/learn/content/average-salary-in-us/" target="_blank" rel="noopener noreferrer" style={{ color: '#228BE6' }}>SoFi 2023</a>)<br />
                            SG: $49,287 (<a href="https://stats.mom.gov.sg/Pages/Income-Summary-Table.aspx" target="_blank" rel="noopener noreferrer" style={{ color: '#228BE6' }}>MOM 2023</a>)<br />
                            IN: $3,900 (<a href="https://www.timedoctor.com/blog/average-salary-in-india/" target="_blank" rel="noopener noreferrer" style={{ color: '#228BE6' }}>TimeDoctor 2023</a>)
                          </Text>
                        </Popover.Dropdown>
                      </Popover>
                    </Group>
                    <Stack >
                      <Box>
                        <Text size="xs" mb={8}>US: ${modelConfig.medianSalary.US.toLocaleString()}</Text>
                        <Slider
                          value={modelConfig.medianSalary.US}
                          onChange={(val) => setModelConfig(prev => ({
                            ...prev,
                            medianSalary: { ...prev.medianSalary, US: val }
                          }))}
                          min={0}
                          max={100000}
                          step={1000}
                          label={(value) => `$${value.toLocaleString()}`}
                        />
                      </Box>
                      <Box>
                        <Text size="xs" mb={8}>SG: ${modelConfig.medianSalary.SG.toLocaleString()}</Text>
                        <Slider
                          value={modelConfig.medianSalary.SG}
                          onChange={(val) => setModelConfig(prev => ({
                            ...prev,
                            medianSalary: { ...prev.medianSalary, SG: val }
                          }))}
                          min={0}
                          max={100000}
                          step={1000}
                          label={(value) => `$${value.toLocaleString()}`}
                        />
                      </Box>
                      <Box>
                        <Text size="xs" mb={8}>IN: ${modelConfig.medianSalary.IN.toLocaleString()}</Text>
                        <Slider
                          value={modelConfig.medianSalary.IN}
                          onChange={(val) => setModelConfig(prev => ({
                            ...prev,
                            medianSalary: { ...prev.medianSalary, IN: val }
                          }))}
                          min={0}
                          max={100000}
                          step={1000}
                          label={(value) => `$${value.toLocaleString()}`}
                        />
                      </Box>
                    </Stack>
                  </Box>

                  <Box>
                    <Group gap={4}>
                      <Text size="sm" fw={500}>Cost of Living (USD/year)</Text>
                      <Popover width={280} position="bottom" shadow="md">
                        <Popover.Target>
                          <IconInfoCircle size={16} style={{ color: 'gray', cursor: 'pointer' }} />
                        </Popover.Target>
                        <Popover.Dropdown>
                          <Text size="sm">
                            Based on monthly living costs (Numbeo 2024):<br />
                            US: $13,998/year (<a href="https://www.numbeo.com/cost-of-living/country_result.jsp?country=United+States" target="_blank" rel="noopener noreferrer" style={{ color: '#228BE6' }}>source</a>)<br />
                            SG: $13,576/year (<a href="https://www.numbeo.com/cost-of-living/in/Singapore" target="_blank" rel="noopener noreferrer" style={{ color: '#228BE6' }}>source</a>)<br />
                            IN: $4,015/year (<a href="https://www.numbeo.com/cost-of-living/country_result.jsp?country=India&displayCurrency=USD" target="_blank" rel="noopener noreferrer" style={{ color: '#228BE6' }}>source</a>)
                          </Text>
                        </Popover.Dropdown>
                      </Popover>
                    </Group>
                    <Stack>
                      <Box>
                        <Text size="xs" mb={8}>US: ${modelConfig.costOfLiving.US.toLocaleString()}</Text>
                        <Slider
                          value={modelConfig.costOfLiving.US}
                          onChange={(val) => setModelConfig(prev => ({
                            ...prev,
                            costOfLiving: { ...prev.costOfLiving, US: val }
                          }))}
                          min={0}
                          max={20000}
                          step={100}
                          label={(value) => `$${value.toLocaleString()}`}
                        />
                      </Box>
                      <Box>
                        <Text size="xs" mb={8}>SG: ${modelConfig.costOfLiving.SG.toLocaleString()}</Text>
                        <Slider
                          value={modelConfig.costOfLiving.SG}
                          onChange={(val) => setModelConfig(prev => ({
                            ...prev,
                            costOfLiving: { ...prev.costOfLiving, SG: val }
                          }))}
                          min={0}
                          max={20000}
                          step={100}
                          label={(value) => `$${value.toLocaleString()}`}
                        />
                      </Box>
                      <Box>
                        <Text size="xs" mb={8}>IN: ${modelConfig.costOfLiving.IN.toLocaleString()}</Text>
                        <Slider
                          value={modelConfig.costOfLiving.IN}
                          onChange={(val) => setModelConfig(prev => ({
                            ...prev,
                            costOfLiving: { ...prev.costOfLiving, IN: val }
                          }))}
                          min={0}
                          max={20000}
                          step={100}
                          label={(value) => `$${value.toLocaleString()}`}
                        />
                      </Box>
                    </Stack>
                  </Box>
                </Stack>
              </Popover.Dropdown >
            </Popover >
          </Group >
        </Title >

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
              placeholder="Ask me anything about the salary prediction..."
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
      </Stack >
    </Paper >
  );
}

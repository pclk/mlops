import { Container, Title, Text, Button, Group, Stack, ThemeIcon } from '@mantine/core';
import { Briefcase, Bot, BarChart } from 'lucide-react';
import Link from 'next/link';

export default function Home() {
  return (
    <Container size="lg" py={{ base: 'xl', md: '6rem' }}>
      {/* Hero Section */}
      <Stack ta="center" gap="xl" mb={50}>
        <Title
          size="h1"
          fw={900}
        >
          WorkAdvisor
        </Title>
        <Text size="xl" maw={600} mx="auto" c="dimmed">
          Your AI-powered career companion for salary negotiations and professional growth
        </Text>
        <Group justify="center" mt="md">
          <Button
            component={Link}
            href="/salary"
            size="lg"
            variant="gradient"
            gradient={{ from: 'blue', to: 'cyan' }}
          >
            Salary AI
          </Button>
          <Button
            component={Link}
            href="/post-prediction"
            size="lg"
            variant="gradient"
            gradient={{ from: 'violet', to: 'grape', deg: 45 }}
          >
            Post Predictor AI
          </Button>
          <Button
            component={Link}
            href="/education"
            size="lg"
            variant="gradient"
            gradient={{ from: 'teal', to: 'lime', deg: 45 }}
          >
            Education AI
          </Button>
        </Group>
      </Stack>

      {/* Features Section */}
      <Container size="md" py="xl">
        <Stack gap="xl">
          <Group gap="lg" justify="center">
            <ThemeIcon size={54} radius="md" variant="light">
              <Briefcase size={30} />
            </ThemeIcon>
            <Stack gap={0} style={{ flex: 1 }}>
              <Text size="lg" fw={500} mb={4}>
                Salary Insights
              </Text>
              <Text c="dimmed">
                Get personalized salary recommendations based on your experience and market data
              </Text>
            </Stack>
          </Group>

          <Group gap="lg" justify="center">
            <ThemeIcon size={54} radius="md" variant="light">
              <Bot size={30} />
            </ThemeIcon>
            <Stack gap={0} style={{ flex: 1 }}>
              <Text size="lg" fw={500} mb={4}>
                AI-Powered Advice
              </Text>
              <Text c="dimmed">
                Receive tailored negotiation strategies and professional development guidance
              </Text>
            </Stack>
          </Group>

          <Group gap="lg" justify="center">
            <ThemeIcon size={54} radius="md" variant="light">
              <BarChart size={30} />
            </ThemeIcon>
            <Stack gap={0} style={{ flex: 1 }}>
              <Text size="lg" fw={500} mb={4}>
                Market Analysis
              </Text>
              <Text c="dimmed">
                Stay informed with real-time industry trends and compensation benchmarks
              </Text>
            </Stack>
          </Group>
        </Stack>
      </Container>
    </Container>
  );
}

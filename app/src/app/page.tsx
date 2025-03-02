'use client';

import { Container, Title, Text, Button, Group, Stack, ThemeIcon } from '@mantine/core';
import { Home, Car, DollarSign, MapPin } from 'lucide-react';
import Link from 'next/link';
import { useState, useEffect } from 'react';

export default function Main() {
  const [lastHoveredButton, setLastHoveredButton] = useState<"property" | "car">("property");
  const [isHovered, setIsHovered] = useState(false);
  const [autoTransition, setAutoTransition] = useState(true);

  const featureData = {
    property: [
      {
        icon: <Home size={30} />,
        title: 'Property Value Estimator',
        description: 'Get real-time property value estimates based on market trends and location data.',
      },
      {
        icon: <MapPin size={30} />,
        title: 'Location Insights',
        description: 'Discover optimal locations for investment based on market data and neighborhood analysis.',
      },
      {
        icon: <DollarSign size={30} />,
        title: 'Investment Potential',
        description: 'Assess the potential return on investment for different properties and neighborhoods.',
      },
    ],
    car: [
      {
        icon: <Car size={30} />,
        title: 'Car Price Prediction',
        description: 'Get accurate price predictions for used cars by providing basic details like brand, model, year, and other relevant attributes.',
      },
      {
        icon: <MapPin size={30} />,
        title: 'Location-Based Insights',
        description: 'Get localized insights on car pricing and trends based on your region or city.',
      },
      {
        icon: <DollarSign size={30} />,
        title: 'Value Estimator',
        description: 'Estimate the value of a used car based on various factors such as make, model, year, and condition.',
      },
    ],
  };

  useEffect(() => {
    if (!isHovered && autoTransition) {
      const intervalId = setInterval(() => {
        setLastHoveredButton((prev: "property" | "car") => {
          if (prev === "property") return "car";
          return "property";
        });
      }, 3750);

      return () => clearInterval(intervalId);
    }
  }, [autoTransition, isHovered]);

  const getButtonClass = (button: string) => {
    return button === lastHoveredButton ? 'button-scale' : '';
  };

  const handleMouseEnter = (section: "property" | "car") => {
    setLastHoveredButton(section);
    setIsHovered(true);
    setAutoTransition(false);
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
    setAutoTransition(true);
  };

  return (
    <Container size="lg" py={{ base: 'xl', md: '6rem' }}>
      {/* Hero Section */}
      <Stack ta="center" gap="xl" mb={50}>
        <Title size="h1" fw={900}>
          Property & Car Advisor
        </Title>
        <Text size="xl" maw={600} mx="auto" c="dimmed">
          Your AI-powered guide to property valuation, car price prediction, and more.
        </Text>
        <Group justify="center" mt="md">
          <Button
            component={Link}
            href="/property"
            size="lg"
            variant="gradient"
            gradient={{ from: 'blue', to: 'cyan' }}
            onMouseEnter={() => handleMouseEnter('property')}
            onMouseLeave={handleMouseLeave}
            className={getButtonClass('property')}
          >
            Property Predictor
          </Button>
          <Button
            component={Link}
            href="/used_car_prediction"
            size="lg"
            variant="gradient"
            gradient={{ from: 'violet', to: 'grape', deg: 45 }}
            onMouseEnter={() => handleMouseEnter('car')}
            onMouseLeave={handleMouseLeave}
            className={getButtonClass('car')}
          >
            Used Car Predictor
          </Button>
        </Group>
      </Stack>

      {/* Features Section */}
      <Container size="md" py="xl">
        <Stack gap="xl">
          {/* Map the features based on the last hovered button */}
          {featureData[lastHoveredButton].map((feature, index) => (
            <Group key={index} gap="lg" justify="center">
              <ThemeIcon size={54} radius="md" variant="light">
                {feature.icon}
              </ThemeIcon>
              <Stack gap={0} style={{ flex: 1 }}>
                <Text size="lg" fw={500} mb={4}>
                  {feature.title}
                </Text>
                <Text c="dimmed">
                  {feature.description}
                </Text>
              </Stack>
            </Group>
          ))}
        </Stack>
      </Container>
    </Container>
  );
}

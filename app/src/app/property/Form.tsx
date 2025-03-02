
"use client"

import {
  Autocomplete,
  Box,
  Button,
  Card,
  Collapse,
  Grid,
  Group,
  NumberInput,
  Paper,
  Progress,
  SegmentedControl,
  Select,
  Slider,
  Stack,
  Switch,
  Text,
  TextInput,
  Title,
  Tooltip
} from '@mantine/core';
import { useForm } from '@mantine/form';
import { notifications } from '@mantine/notifications';
import { driver } from "driver.js";
import "driver.js/dist/driver.css";
import { CheckCircle2, HelpCircle, XCircle } from 'lucide-react';
import React, { useEffect, useState } from 'react';
import "./driver.css";
import { predictPropertyPrice } from './actions';
import Chat from './Chat'; // Import the Chat component

import propertyTypes from './ts-data/propertyTypes';
import sellingMethods from './ts-data/sellingMethods';
import suburbs from './ts-data/suburbs';
import sellers from './ts-data/sellers';
import BatchForm from './BatchForm';

interface FormLabelProps {
  label: string;
  tooltip: string;
  style?: React.CSSProperties;
}

function FormLabel({ label, tooltip, style }: FormLabelProps) {
  const tooltipContent = tooltip.split('\n').map((line, index) => (
    <span key={index}>
      {line}
      {index < tooltip.split('\n').length - 1 && <br />}
    </span>
  ));

  return (
    <Text component="div" style={{ display: 'inline-flex', alignItems: 'center', gap: '5px', ...style }}>
      {label}
      <Tooltip
        label={tooltipContent}
        position="top-start"
        multiline
      >
        <HelpCircle size={16} style={{ cursor: 'help' }} />
      </Tooltip>
    </Text>
  );
}

// Property form tooltips
const propertyTooltips = {
  suburb: "The name of the suburb where the property is located.",
  rooms: "Number of rooms in the property.",
  type: "Type of property (House, Unit, Townhouse, etc.).",
  method: "Method of sale (S = property sold; SP = property sold prior; PI = property passed in; PN = sold prior not disclosed; SN = sold not disclosed; NB = no bid; VB = vendor bid; W = withdrawn prior to auction; SA = sold after auction; SS = sold after auction price not disclosed).",
  seller: "Real estate company handling the sale.",
  distance: "Distance from CBD in kilometers.",
  bathroom: "Number of bathrooms in the property.",
  car: "Number of car spaces.",
  landsize: "Land size in square meters.",
  buildingArea: "Building area in square meters.",
  propertyAge: "Age of the property in years.",
  direction: "Direction the property faces (N, S, E, W, NE, NW, SE, SW).",
  landSizeNotOwned: "Whether part of the land size is not owned by the property (e.g., shared driveways)."
};

// Property presets
const propertyPresets = {
  suburban_house: {
    suburb: "Reservoir",
    rooms: 3,
    type: "h",
    method: "S",
    seller: "Ray",
    distance: 11.2,
    bathroom: 1.0,
    car: 2,
    landsize: 556.0,
    buildingArea: 120.0,
    propertyAge: 50,
    direction: "N",
    landSizeNotOwned: false
  },
  inner_city_apartment: {
    suburb: "Melbourne",
    rooms: 2,
    type: "u",
    method: "SP",
    seller: "Nelson",
    distance: 5.8,
    bathroom: 1.0,
    car: 1,
    landsize: 1,
    buildingArea: 75.0,
    propertyAge: 15,
    direction: "E",
    landSizeNotOwned: true
  },
  luxury_house: {
    suburb: "Toorak",
    rooms: 5,
    type: "h",
    method: "S",
    seller: "RT Edgar",
    distance: 7.6,
    bathroom: 3.5,
    car: 4,
    landsize: 850.0,
    buildingArea: 320.0,
    propertyAge: 25,
    direction: "N",
    landSizeNotOwned: false
  }
};

export default function PropertyPriceForm() {
  const driverObj = React.useRef(
    driver({
      showProgress: true,
      animate: true,
      smoothScroll: true,
      stagePadding: 5,
      popoverClass: 'custom-popover',
      steps: [
        {
          popover: {
            title: 'Welcome to Property Price Predictor!',
            description: 'This tool will help you predict property prices based on various features of the property. \
            \n\nIf you don\'t want to go through this tour, click the "x" button on the top-right. \
            \n\nOtherwise, let\'s quickly make a prediction to show you around. \
            ',
            side: "bottom",
            align: 'start',
            onCloseClick: () => {
              localStorage.setItem('tourSkipped', 'true');
              setShowTour(false);
              driverObj.current.destroy();
            }
          }
        },
        {
          element: '#preset-selector',
          popover: {
            title: 'Property Presets',
            description: 'Start by selecting a property preset. \
            \n\nThis can make it easy for you to try out our model. \
            \n\n You can create your own custom property profile, but let\'s proceed with any of the presets. \
            \n\nClick "Next" to continue.',
            side: "top",
            align: 'start'
          }
        },
        {
          element: '#form-section',
          popover: {
            title: 'Property Details',
            description: 'The purpose of this tool is to accurately and quickly predict property prices based on property features. \
            \n\nThe fields here are what the model needs to make an informed prediction. \
            \n\nOur model generally makes predictions within a reasonable margin of error based on historical sales data.',
            side: "left",
            align: 'start'
          }
        },
        {
          element: '#submit-button',
          popover: {
            title: 'Submit your predictions',
            description: 'Go ahead and submit your prediction. \
            \n\nLet\'s see what the prediction made by our model!.',
            side: "top",
            align: 'start',
            disableButtons: ['next']
          }
        },
      ]
    })
  );

  interface SavedPrediction {
    name: string;
    timestamp: number;
    predictedPrice: number | null;
    duration: number;
    formValues: FormValues;
  }

  interface FormValues {
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
  }

  const form = useForm<FormValues>({
    initialValues: {
      suburb: '',
      rooms: null,
      type: '',
      method: '',
      seller: '',
      distance: null,
      bathroom: null,
      car: null,
      landsize: null,
      buildingArea: null,
      propertyAge: null,
      direction: '',
      landSizeNotOwned: false,
    },
    validate: {
      suburb: (value) => !value ? 'Suburb is required' : null,
      rooms: (value) => value === null ? 'Number of rooms is required' : null,
    },
    validateInputOnChange: true,
  });

  const [isLoading, setIsLoading] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState('suburban_house');
  const [formExpanded, setFormExpanded] = useState(true);
  const [showTour, setShowTour] = useState(true);

  const [predictedPrice, setPredictedPrice] = useState<number | null>(null);
  const [duration, setDuration] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const [savedPredictions, setSavedPredictions] = useState<SavedPrediction[]>([]);
  const [savePredictionName, setSavePredictionName] = useState('');
  const [progress, setProgress] = useState({ completed: 0, total: 1 });
  const [timer, setTimer] = useState(0);
  const [showProgress, setShowProgress] = useState(false);
  const [triggerAnalysis, setTriggerAnalysis] = useState(false); // Added state for triggering AI analysis
  const timerIntervalRef = React.useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = React.useRef<number>(0);
  const fadeTimeoutRef = React.useRef<NodeJS.Timeout | null>(null);
  const [predictionMode, setPredictionMode] = useState<'single' | 'batch'>('single');

  useEffect(() => {
    const tourSkipped = localStorage.getItem('tourSkipped');
    if (tourSkipped === 'true') {
      setShowTour(false);
      return; // Do not proceed to show tour if skipped
    }

    if (showTour) {
      driverObj.current.drive();
    }
  }, [showTour]);

  // Cleanup function to ensure interval is properly cleared
  const clearCurrentInterval = () => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    setTimer(0);
  };

  // Start timer function
  const startTimer = () => {
    clearCurrentInterval();
    startTimeRef.current = performance.now();
    timerIntervalRef.current = setInterval(() => {
      setTimer(performance.now() - startTimeRef.current);
    }, 10);
  };

  const handleSubmit = async (values: typeof form.values) => {
    // Destroy the tour if it's active
    driverObj.current.destroy();
    setShowTour(false);
    localStorage.setItem('tourSkipped', 'true');

    setIsLoading(true);
    setError(null);
    setPredictedPrice(null);
    setFormExpanded(false);
    setProgress({ completed: 0, total: 1 });
    setTimer(0);
    clearCurrentInterval();
    setTriggerAnalysis(false); // Reset trigger analysis

    setProgress({ completed: 0, total: 1 });
    setShowProgress(true);

    // Start timer for the prediction
    startTimer();

    try {
      const startTime = performance.now();
      const result = await predictPropertyPrice(values);
      const endTime = performance.now();
      const durationTime = endTime - startTime;
      console.log('API Response:', result);
      setDuration(durationTime);
      clearCurrentInterval();
      setProgress({ completed: 1, total: 1 });

      if (result.success && result.data) {
        setPredictedPrice(result.data.prediction);
        setTriggerAnalysis(true); // Trigger AI analysis when prediction succeeds
        notifications.show({
          title: 'Prediction Complete',
          message: 'Successfully predicted property price',
          color: 'green',
          icon: <CheckCircle2 size={18} />,
        });
      } else {
        setError(result.error || 'Failed to predict property price');
        notifications.show({
          title: 'Prediction Failed',
          message: result.error || 'An error occurred',
          color: 'red',
          icon: <XCircle size={18} />,
        });
      }
    } catch (error) {
      console.error('Error predicting property price:', error);
      setError('An unexpected error occurred');
    }

    clearCurrentInterval();
    setIsLoading(false);

    // Keep progress visible then fade out
    if (fadeTimeoutRef.current) {
      clearTimeout(fadeTimeoutRef.current);
    }
    fadeTimeoutRef.current = setTimeout(() => {
      setShowProgress(false);
    }, 1000);
  };

  // Set initial preset values when component mounts
  // Load saved predictions from localStorage
  React.useEffect(() => {
    const saved = localStorage.getItem('savedPropertyPredictions');
    if (saved) {
      setSavedPredictions(JSON.parse(saved));
    }
    if (propertyPresets.suburban_house) {
      form.setValues(propertyPresets.suburban_house);
    }
  }, []);

  // Cleanup intervals and timeouts when component unmounts
  React.useEffect(() => {
    return () => {
      clearCurrentInterval();
      if (fadeTimeoutRef.current) {
        clearTimeout(fadeTimeoutRef.current);
      }
    };
  }, []);

  return (
    <Paper shadow="xs" p="md" style={{ height: '100%' }}>
      <SegmentedControl
        data={[
          { value: 'single', label: 'Single Property' },
          { value: 'batch', label: 'Batch Prediction' }
        ]}
        value={predictionMode}
        onChange={(value) => setPredictionMode(value as 'single' | 'batch')}
        mb="md"
      />

      {predictionMode === 'single' ? (

        <Grid>
          <Grid.Col span={predictedPrice ? { base: 12, md: 6 } : 12} style={{ minHeight: 0 }}>
            <Group justify="space-between" mb="md">
              <Title order={3} id="welcome-message">Property Price Prediction Form</Title>
              <Button
                variant="subtle"
                size="xs"
                onClick={() => driverObj.current.drive()}
              >
                Show Tour
              </Button>
            </Group>


            <Collapse in={!isLoading && formExpanded} transitionDuration={200} id="preset-selector">
              <Stack gap="xs">
                <Group mb="md">
                  <Tooltip
                    label={
                      <img
                        src="/suburban.jpg"
                        alt="Suburban Preview"
                        style={{
                          maxWidth: '500px',
                          height: 'auto'
                        }}
                      />
                    }
                    position="bottom"
                    transitionProps={{ transition: 'pop' }}
                  >
                    <Button
                      variant={selectedPreset === 'suburban_house' ? 'filled' : 'light'}
                      onClick={() => {
                        setSelectedPreset('suburban_house');
                        form.setValues(propertyPresets.suburban_house);
                      }}
                      size="sm"
                      leftSection={<span style={{ fontSize: '18px' }}>🏡</span>}
                      className="preset-button"
                    >
                      Suburban House
                    </Button>
                  </Tooltip>

                  {/* Inner City Apartment Button */}
                  <Tooltip
                    label={
                      <img
                        src="/apartment.jpg"
                        alt="Apartment Preview"
                        style={{
                          maxWidth: '500px',
                          height: 'auto'
                        }}
                      />
                    }
                    position="bottom"
                    transitionProps={{ transition: 'pop' }}
                  >
                    <Button
                      variant={selectedPreset === 'inner_city_apartment' ? 'filled' : 'light'}
                      onClick={() => {
                        setSelectedPreset('inner_city_apartment');
                        form.setValues(propertyPresets.inner_city_apartment);
                      }}
                      size="sm"
                      leftSection={<span style={{ fontSize: '18px' }}>🏢</span>}
                      className="preset-button"
                    >
                      Inner City Apartment
                    </Button>
                  </Tooltip>

                  {/* Luxury House Button */}
                  <Tooltip
                    label={
                      <img
                        src="/luxury.jpg"
                        alt="Luxury Preview"
                        style={{
                          maxWidth: '500px',
                          height: 'auto'
                        }}
                      />
                    }
                    position="bottom"
                    transitionProps={{ transition: 'pop' }}
                  >
                    <Button
                      variant={selectedPreset === 'luxury_house' ? 'filled' : 'light'}
                      onClick={() => {
                        setSelectedPreset('luxury_house');
                        form.setValues(propertyPresets.luxury_house);
                      }}
                      size="sm"
                      leftSection={<span style={{ fontSize: '18px' }}>🏰</span>}
                      className="preset-button"
                    >
                      Luxury House
                    </Button>
                  </Tooltip>

                  {/* Clear Button */}
                  <Button
                    variant="subtle"
                    color="gray"
                    onClick={() => {
                      setSelectedPreset('');
                      form.reset();
                    }}
                    size="sm"
                    leftSection={<span style={{ fontSize: '18px' }}>🧹</span>}
                  >
                    Clear
                  </Button>
                </Group>
              </Stack>
            </Collapse>


            <form onSubmit={form.onSubmit(handleSubmit)} style={{ marginTop: formExpanded ? 0 : '1rem' }}>
              <Card withBorder shadow="sm" style={{ overflow: 'hidden' }}>
                <Stack gap="md">
                  <Collapse in={formExpanded} transitionDuration={200} id="form-section">
                    <Grid style={{ minHeight: 0 }}>
                      <Grid.Col span={12}>
                        <Select
                          label={<FormLabel label="🏙️ Suburb" tooltip={propertyTooltips.suburb} />}
                          placeholder="Select suburb"
                          data={suburbs.map(suburb => ({ value: suburb, label: suburb }))}
                          searchable
                          maxDropdownHeight={280}
                          {...form.getInputProps('suburb')}
                        />
                      </Grid.Col>

                      <Grid.Col span={6} style={{ minHeight: '90px' }}>
                        <Select
                          label={<FormLabel label="🧭 Direction" tooltip={propertyTooltips.direction} />}
                          placeholder="Select direction"
                          data={[
                            { value: 'N', label: 'North' },
                            { value: 'S', label: 'South' },
                            { value: 'E', label: 'East' },
                            { value: 'W', label: 'West' },
                          ]}
                          {...form.getInputProps('direction')}
                        />
                      </Grid.Col>


                      <Grid.Col span={6} style={{ minHeight: '90px' }}>
                        <Select
                          label={<FormLabel label="🏠 Type" tooltip={propertyTooltips.type} />}
                          placeholder="Select property type"
                          data={propertyTypes.map(method => {
                            const labels = {
                              'h': 'House',
                              't': 'Town',
                              'u': 'Unit/Appartment',
                            };
                            return {
                              value: method,
                              label: labels.hasOwnProperty(method) ? labels[method as keyof typeof labels] : method
                            };
                          })}
                          {...form.getInputProps('type')}
                        />
                      </Grid.Col>

                      <Grid.Col span={6} style={{ minHeight: '90px' }}>
                        <Select
                          label={<FormLabel label="🔨 Sale Method" tooltip={propertyTooltips.method} />}
                          placeholder="Select sale method"
                          data={sellingMethods.map(method => {
                            const labels = {
                              'S': 'Sold',
                              'SP': 'Sold Prior',
                              'PI': 'Passed In',
                              "SA": "Sold after auction",
                              "VB": "Vendor bid",
                            };
                            return {
                              value: method,
                              label: labels.hasOwnProperty(method) ? labels[method as keyof typeof labels] : method
                            };
                          })}
                          {...form.getInputProps('method')}
                        />
                      </Grid.Col>

                      <Grid.Col span={6} style={{ minHeight: '90px' }}>
                        <Autocomplete
                          label={<FormLabel label="🏢 Seller" tooltip={propertyTooltips.seller} />}
                          placeholder="Real estate agency"
                          data={sellers}
                          {...form.getInputProps('seller')}
                        />
                      </Grid.Col>


                      <Grid.Col span={6} style={{ minHeight: '90px' }}>
                        <FormLabel
                          label="🛏️ Rooms"
                          tooltip={propertyTooltips.rooms}
                          style={{ color: 'var(--foreground)', marginBottom: '10px' }}
                        />
                        <Slider
                          min={1}
                          max={10}
                          step={1}
                          label={(value) => `${value} rooms`}
                          marks={[
                            { value: 1, label: '1' },
                            { value: 3, label: '3' },
                            { value: 5, label: '5' },
                            { value: 7, label: '7' },
                            { value: 10, label: '10' }
                          ]}
                          value={form.values.rooms || 1}
                          onChange={(val) => form.setFieldValue('rooms', val)}
                        />
                      </Grid.Col>

                      <Grid.Col span={6} style={{ minHeight: '90px' }}>
                        <FormLabel
                          label="🚗 Distance from CBD (km)"
                          tooltip={propertyTooltips.distance}
                          style={{ color: 'var(--foreground)', marginBottom: '10px' }}
                        />
                        <Slider
                          min={0}
                          max={50}
                          step={1}
                          label={(value) => `${value} km`}
                          marks={[
                            { value: 0, label: '0' },
                            { value: 10, label: '10' },
                            { value: 20, label: '20' },
                            { value: 30, label: '30' },
                            { value: 40, label: '40' },
                            { value: 50, label: '50' },
                          ]}
                          value={form.values.distance || 0}
                          onChange={(val) => form.setFieldValue('distance', val)}
                        />
                      </Grid.Col>

                      <Grid.Col span={6} style={{ minHeight: '90px' }}>
                        <FormLabel
                          label="🚿 Bathrooms"
                          tooltip={propertyTooltips.bathroom}
                          style={{ color: 'var(--foreground)', marginBottom: '10px' }}
                        />
                        <Slider
                          min={1}
                          max={5}
                          step={1}
                          label={(value) => `${value} ${value === 1 ? 'bathroom' : 'bathrooms'}`}
                          marks={[
                            { value: 1, label: '1' },
                            { value: 2, label: '2' },
                            { value: 3, label: '3' },
                            { value: 4, label: '4' },
                            { value: 5, label: '5' },
                          ]}
                          value={form.values.bathroom || 0}
                          onChange={(val) => form.setFieldValue('bathroom', val)}
                        />
                      </Grid.Col>

                      <Grid.Col span={6} style={{ minHeight: '90px' }}>
                        <FormLabel
                          label="🅿️ Car Spaces"
                          tooltip={propertyTooltips.car}
                          style={{ color: 'var(--foreground)', marginBottom: '10px' }}
                        />
                        <Slider
                          min={0}
                          max={5}
                          step={1}
                          label={(value) => `${value} ${value === 1 ? 'space' : 'spaces'}`}
                          marks={[
                            { value: 0, label: '0' },
                            { value: 1, label: '1' },
                            { value: 2, label: '2' },
                            { value: 3, label: '3' },
                            { value: 4, label: '4' },
                            { value: 5, label: '5' },
                          ]}
                          value={form.values.car || 0}
                          onChange={(val) => form.setFieldValue('car', val)}
                        />
                      </Grid.Col>

                      <Grid.Col span={6} style={{ minHeight: '90px' }}>
                        <FormLabel
                          label="📏 Land Size (m²)"
                          tooltip={propertyTooltips.landsize}
                          style={{ color: 'var(--foreground)', marginBottom: '10px' }}
                        />
                        <Slider
                          min={10}
                          max={2000}
                          step={10}
                          label={(value) => `${value} m²`}
                          marks={[
                            { value: 10, label: '10' },
                            { value: 500, label: '500' },
                            { value: 1000, label: '1000' },
                            { value: 1500, label: '1500' },
                            { value: 2000, label: '2000' },
                          ]}
                          value={form.values.landsize || 0}
                          onChange={(val) => form.setFieldValue('landsize', val)}
                        />
                      </Grid.Col>

                      <Grid.Col span={6} style={{ minHeight: '90px' }}>
                        <FormLabel
                          label="🏗️ Building Area (m²)"
                          tooltip={propertyTooltips.buildingArea}
                          style={{ color: 'var(--foreground)', marginBottom: '10px' }}
                        />
                        <Slider
                          min={0}
                          max={500}
                          step={10}
                          label={(value) => `${value} m²`}
                          marks={[
                            { value: 0, label: '0' },
                            { value: 100, label: '100' },
                            { value: 200, label: '200' },
                            { value: 300, label: '300' },
                            { value: 400, label: '400' },
                            { value: 500, label: '500' },
                          ]}
                          value={form.values.buildingArea || 0}
                          onChange={(val) => form.setFieldValue('buildingArea', val)}
                        />
                      </Grid.Col>

                      <Grid.Col span={6} style={{ minHeight: '90px' }}>
                        <FormLabel
                          label="📅 Property Age (years)"
                          tooltip={propertyTooltips.propertyAge}
                          style={{ color: 'var(--foreground)', marginBottom: '10px' }}
                        />
                        <Slider
                          id="property-age"
                          min={0}
                          max={100}
                          step={1}
                          label={(value) => `${value} years`}
                          marks={[
                            { value: 0, label: 'New' },
                            { value: 25, label: '25y' },
                            { value: 50, label: '50y' },
                            { value: 75, label: '75y' },
                            { value: 100, label: '100y+' },
                          ]}
                          value={form.values.propertyAge || 0}
                          onChange={(val) => form.setFieldValue('propertyAge', val)}
                        />
                      </Grid.Col>


                      <Grid.Col span={12}>
                        <Switch
                          label={<FormLabel label="⚠️ Land Size Not Fully Owned" tooltip={propertyTooltips.landSizeNotOwned} />}
                          {...form.getInputProps('landSizeNotOwned', { type: 'checkbox' })}
                        />
                      </Grid.Col>
                    </Grid>
                  </Collapse>


                  <Collapse in={showProgress} transitionDuration={200}>
                    <Stack gap="xs" style={{ transition: 'opacity 0.5s ease-out' }}>
                      <Progress
                        value={(progress.completed / progress.total) * 100}
                        animated={isLoading}
                        size="xl"
                        radius="xl"
                      />
                      <Text size="sm" ta="center">
                        {isLoading ?
                          `Processing prediction...` :
                          `Prediction completed`
                        }
                      </Text>
                      <Text size="sm" ta="center" c="dimmed">
                        {(timer / 1000).toFixed(2)}s
                      </Text>
                    </Stack>
                  </Collapse>

                  {predictedPrice ? (
                    <Button
                      onClick={(e) => {
                        e.preventDefault();
                        setPredictedPrice(null);
                        setError(null);
                        setFormExpanded(true);
                        setProgress({ completed: 0, total: 1 });
                        setTimer(0);
                        setShowProgress(false);
                        setTriggerAnalysis(false); // Reset trigger analysis
                        clearCurrentInterval();
                      }}
                      type="button"
                      color="gray"
                    >
                      Start New Prediction
                    </Button>
                  ) : (
                    <Button
                      id='submit-button'
                      type="submit"
                      mt="md"
                      loading={isLoading}
                    >
                      {isLoading ? 'Calculating...' : 'Predict Property Price'}
                    </Button>
                  )}

                  {predictedPrice && (
                    <Paper p="xl" withBorder radius="md">
                      <Stack spacing="lg">
                        <Box>
                          <Text size="sm" weight={500} color="dimmed" transform="uppercase" mb={4}>
                            Prediction Result
                          </Text>
                          <Title order={3} mb="xs">Predicted Property Price</Title>
                        </Box>

                        <Card withBorder shadow="sm" radius="md" p="xl">
                          <Stack spacing="xl">
                            {/* Main price with visual emphasis */}
                            <Box ta="center" py="md">
                              <Text size="sm" c="dimmed" mb={5}>Estimated Market Value</Text>
                              <Text fw={800} size="30px" c="blue.7">
                                ${predictedPrice.toLocaleString()}
                              </Text>
                            </Box>

                            <Box>
                              <Text size="sm" weight={500} c="dimmed" mb={8}>Property Details</Text>
                              <Group position="apart" spacing="xl">
                                <Stack spacing={4}>
                                  <Text fw={600}>{form.values.suburb}</Text>
                                  <Text size="sm">
                                    {form.values.type === 'h' ? 'House' :
                                      form.values.type === 'u' ? 'Unit/Apartment' :
                                        form.values.type === 't' ? 'Townhouse' : form.values.type}
                                  </Text>
                                </Stack>

                                <Stack spacing={4} align="flex-end">
                                  <Text fw={600}>{form.values.rooms} {form.values.rooms === 1 ? 'room' : 'rooms'}</Text>
                                  <Text size="sm">{form.values.bathroom} {form.values.bathroom === 1 ? 'bathroom' : 'bathrooms'}</Text>
                                </Stack>
                              </Group>
                            </Box>

                            <Group position="right">
                              <Text size="xs" c="dimmed" style={{ fontStyle: 'italic' }}>
                                Calculation time: {(duration / 1000).toFixed(2)}s
                              </Text>
                            </Group>
                          </Stack>
                        </Card>
                      </Stack>
                    </Paper>
                  )}
                  {predictedPrice && (
                    <Card withBorder>
                      <Stack gap="md">
                        <Title order={4}>Save Prediction</Title>
                        <Group>
                          <TextInput
                            placeholder="Enter a name for this prediction"
                            value={savePredictionName}
                            onChange={(event) => setSavePredictionName(event.currentTarget.value)}
                            style={{ flex: 1 }}
                          />
                          <Button
                            onClick={() => {
                              if (!savePredictionName.trim()) {
                                notifications.show({
                                  title: 'Error',
                                  message: 'Please enter a name for the prediction',
                                  color: 'red',
                                });
                                return;
                              }

                              const newSavedPrediction: SavedPrediction = {
                                name: savePredictionName,
                                timestamp: Date.now(),
                                predictedPrice,
                                duration,
                                formValues: form.values,
                              };

                              const updatedSavedPredictions = [...savedPredictions, newSavedPrediction];
                              setSavedPredictions(updatedSavedPredictions);
                              localStorage.setItem('savedPropertyPredictions', JSON.stringify(updatedSavedPredictions));

                              setSavePredictionName('');
                              notifications.show({
                                title: 'Success',
                                message: 'Prediction saved successfully',
                                color: 'green',
                              });
                            }}
                          >
                            Save
                          </Button>
                        </Group>
                      </Stack>
                    </Card>
                  )}

                  {error && (
                    <Text c="red" size="sm">
                      {error}
                    </Text>
                  )}
                </Stack>
              </Card>
            </form>

            {savedPredictions.length > 0 && (
              <Card withBorder mt="md" id="saved-predictions">
                <Stack gap="md">
                  <Group justify="space-between">
                    <Title order={4}>Saved Predictions</Title>
                    <Text size="sm" c="dimmed">{savedPredictions.length} saved</Text>
                  </Group>
                  <Grid>
                    {savedPredictions
                      .sort((a, b) => b.timestamp - a.timestamp)
                      .map((saved, index) => (
                        <Grid.Col key={index} span={{ base: 12, sm: 6, lg: 4 }}>
                          <Card withBorder shadow="sm">
                            <Stack gap="md">
                              <Group justify="space-between" wrap="nowrap">
                                <Stack gap={2}>
                                  <Text fw={700}>
                                    {saved.name}
                                  </Text>
                                  <Text size="xs" c="dimmed">
                                    {new Date(saved.timestamp).toLocaleString()}
                                  </Text>
                                </Stack>
                              </Group>

                              <Stack gap="xs">
                                <Text size="sm">
                                  {saved.formValues.suburb} {saved.formValues.type}
                                </Text>
                                <Text fw={700} size="md" c="blue">
                                  ${saved.predictedPrice?.toLocaleString() ?? 'N/A'}
                                </Text>
                              </Stack>

                              <Group grow>
                                <Button
                                  variant="light"
                                  size="xs"
                                  onClick={() => {
                                    setPredictedPrice(saved.predictedPrice);
                                    setDuration(saved.duration);
                                    form.setValues(saved.formValues);
                                    setFormExpanded(false);
                                    setTriggerAnalysis(true); // Trigger analysis when loading a saved prediction
                                    notifications.show({
                                      title: 'Prediction Loaded',
                                      message: `Loaded "${saved.name}"`,
                                      color: 'blue',
                                    });
                                  }}
                                >
                                  Load
                                </Button>
                                <Button
                                  variant="light"
                                  color="red"
                                  size="xs"
                                  onClick={() => {
                                    const updatedSavedPredictions = savedPredictions.filter((_, i) => i !== index);
                                    setSavedPredictions(updatedSavedPredictions);
                                    localStorage.setItem('savedPropertyPredictions', JSON.stringify(updatedSavedPredictions));
                                    notifications.show({
                                      title: 'Deleted',
                                      message: `Deleted "${saved.name}"`,
                                      color: 'red',
                                    });
                                  }}
                                >
                                  Delete
                                </Button>
                              </Group>
                            </Stack>
                          </Card>
                        </Grid.Col>
                      ))}
                  </Grid>
                </Stack>
              </Card>
            )}
          </Grid.Col>

          {/* Chat component integration */}
          {predictedPrice && (
            <Grid.Col
              span={{ base: 12, md: 6 }}
              style={{
                transition: 'all 0.3s ease-out',
                opacity: predictedPrice ? 1 : 0,
                transform: predictedPrice ? 'translateX(0)' : 'translateX(20px)',
              }}
            >
              <Chat
                predictedPrice={predictedPrice}
                duration={duration}
                propertyDetails={form.values}
                triggerInitialAnalysis={triggerAnalysis}
              />
            </Grid.Col>
          )}
        </Grid>
      ) : (
        <BatchForm />
      )}
    </Paper>
  );
}


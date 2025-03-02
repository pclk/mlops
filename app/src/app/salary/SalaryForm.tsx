"use client"

import {
  Box,
  Button,
  Card,
  Collapse,
  Grid,
  Group,
  MultiSelect,
  Paper,
  Progress,
  Select,
  Stack,
  Text,
  TextInput,
  Textarea,
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

interface FormLabelProps {
  label: string;
  tooltip: string;
}

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

import { predictSalary } from './actions';
import Chat from './Chat';
import { formTooltips, jobPresets } from './presets';

export default function SalaryForm() {
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
            title: 'Welcome to Salary Predictor!',
            description: 'Hello, I\'m Wei Heng, the creator of this tool. Let me show you around our salary prediction tool. \
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
            title: 'Job Presets',
            description: 'Start by selecting a job preset. \
            \n\nThis can make it easy for you to try out our model. \
            \n\n You can create your own custom profile, but let\'s proceed with any of the presets. \
            \n\nClick "Next" to continue.',
            side: "top",
            align: 'start'
          }
        },
        {
          element: '#form-section',
          popover: {
            title: 'Job Details',
            description: 'The purpose of this tool is to accurately and quickly predict salaries across countries and locations. \
            \n\nThe fields over here are what the model needs to make an informed prediction. \
            \n\nOur model generally makes predictions within 20KUSD/year of error, so it\'s advisable that this is the tolerance you provide.',
            side: "left",
            align: 'start'
          }
        },
        {
          element: '#countries-select',
          popover: {
            title: 'Select Countries',
            description: 'The fields here are special. \
            \n\nThese fields are not only understood by our model, but also are looped as individual predictions.\
            \n\nThis means that you will receive 3 predictions if you chose all these countries, where each prediction is the respective country.\
            \n\nSame applies to the locations, where you will receive multiple predictions for each of your chosen multiple locations.\ ',
            side: "top",
            align: 'start'
          }
        },
        {
          element: '#submit-button',
          popover: {
            title: 'Submit your predictions',
            description: 'Go ahead and submit your prediction. \
            \n\nLet\'s see what the predictions made by our model!.',
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
    predictions: {
      [key: string]: {
        salary: number | null;
        duration: number;
        relativeDuration: number;
      };
    };
    formValues: FormValues;
  }

  interface FormValues {
    job_title: string;
    job_description: string;
    contract_type: string;
    education_level: string;
    seniority: string;
    min_years_experience: string;
    countries: string[];
    location_us: string[];
    location_in: string[];
  }


  const form = useForm<FormValues>({
    initialValues: {
      job_title: '',
      job_description: '',
      contract_type: '',
      education_level: '',
      seniority: '',
      min_years_experience: '',
      countries: ['US', 'SG', 'IN'],
      location_us: [],
      location_in: [],
    },
    validate: {
      countries: (value) => value.length === 0 ? 'Please select at least one country' : null,
    },
    validateInputOnChange: true,
  });

  const [isLoading, setIsLoading] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState('software_engineer');
  const [formExpanded, setFormExpanded] = useState(true);
  const [showTour, setShowTour] = useState(true);


  const [predictions, setPredictions] = useState<{ [key: string]: { salary: number | null; duration: number; relativeDuration: number } }>({});
  const [lastPredictionTime, setLastPredictionTime] = useState<number | null>(null);
  const [lastEndTime, setLastEndTime] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [savedPredictions, setSavedPredictions] = useState<SavedPrediction[]>([]);
  const [savePredictionName, setSavePredictionName] = useState('');
  const [progress, setProgress] = useState({ completed: 0, total: 0 });
  const [timer, setTimer] = useState(0);
  const [showProgress, setShowProgress] = useState(false);
  const [triggerAnalysis, setTriggerAnalysis] = useState(false);
  const timerIntervalRef = React.useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = React.useRef<number>(0);
  const fadeTimeoutRef = React.useRef<NodeJS.Timeout | null>(null);


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
    setPredictions({});
    setFormExpanded(false);
    setLastPredictionTime(null);
    setLastEndTime(null);
    setProgress({ completed: 0, total: 0 });
    setTimer(0);
    clearCurrentInterval();

    console.log("handleSubmit started");

    // Create a queue of all predictions we need to make
    const predictionQueue: { country: string; location: string }[] = [];

    for (const country_code of values.countries) {
      const locationKey = `location_${country_code.toLowerCase()}` as keyof typeof values;
      // Ensure locations is an array, default to empty array if undefined
      const locations = (values[locationKey] as string[]) || [];

      if (locations.length === 0) {
        predictionQueue.push({ country: country_code, location: '' });
      } else {
        for (const location of locations) {
          predictionQueue.push({ country: country_code, location });
        }
      }
    }

    setProgress({ completed: 0, total: predictionQueue.length });
    setShowProgress(true);
    let completedPredictions = 0;
    const totalPredictions = predictionQueue.length;

    // Process predictions sequentially
    for (const { country, location } of predictionQueue) {
      // Reset and start timer for each prediction
      startTimer();

      const payload = {
        job_title: values.job_title,
        job_description: values.job_description,
        contract_type: values.contract_type,
        education_level: values.education_level,
        seniority: values.seniority,
        min_years_experience: values.min_years_experience,
        location_us: values.location_us,
        location_in: values.location_in,
      };

      const predictionKey = location ? `${country}-${location}` : country;
      console.log(`Fetching prediction for ${predictionKey}`);

      try {
        const startTime = performance.now();
        const result = await predictSalary(payload, country, location);
        const endTime = performance.now();
        const duration = endTime - startTime;
        const relativeDuration = lastEndTime ? startTime - lastEndTime : duration;
        setLastEndTime(endTime);
        clearCurrentInterval();
        completedPredictions++;
        setProgress(prev => ({ ...prev, completed: completedPredictions }));

        if (result.success && result.data) {
          setPredictions(prev => ({
            ...prev,
            [predictionKey]: {
              salary: result.data.predicted_salary,
              duration: duration,
              relativeDuration: relativeDuration
            }
          }));
        } else {
          setError(prev => prev || result.error || 'Failed to predict salary');
          notifications.show({
            title: 'Prediction Failed',
            message: `Failed for ${predictionKey}: ${result.error}`,
            color: 'red',
            icon: <XCircle size={18} />,
          });
        }
      } catch (error) {
        console.error(`Error predicting for ${predictionKey}:`, error);
      }
    }

    if (completedPredictions === totalPredictions) {
      notifications.show({
        title: 'All Predictions Complete',
        message: `Successfully processed ${completedPredictions} predictions`,
        color: 'green',
        icon: <CheckCircle2 size={18} />,
      });
      setTriggerAnalysis(true);
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

    console.log("handleSubmit finished");
  };


  // Set initial preset values when component mounts
  // Load saved predictions from localStorage
  React.useEffect(() => {
    const saved = localStorage.getItem('savedPredictions');
    if (saved) {
      setSavedPredictions(JSON.parse(saved));
    }
    if (jobPresets.software_engineer) {
      form.setValues(jobPresets.software_engineer);
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
      <Grid>
        <Grid.Col span={Object.keys(predictions).length > 0 ? { base: 12, md: 6 } : 12} style={{ minHeight: 0 }}>
          <Group justify="space-between" mb="md">
            <Title order={3} id="welcome-message">Salary Prediction Form</Title>
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
                <Button
                  variant={selectedPreset === 'software_engineer' ? 'filled' : 'light'}
                  onClick={() => {
                    setSelectedPreset('software_engineer');
                    form.setValues(jobPresets.software_engineer);
                  }}
                  size="sm"
                >
                  Software Engineer
                </Button>
                <Button
                  variant={selectedPreset === 'data_scientist' ? 'filled' : 'light'}
                  onClick={() => {
                    setSelectedPreset('data_scientist');
                    form.setValues(jobPresets.data_scientist);
                  }}
                  size="sm"
                >
                  Data Scientist
                </Button>
                <Button
                  variant={selectedPreset === 'product_manager' ? 'filled' : 'light'}
                  onClick={() => {
                    setSelectedPreset('product_manager');
                    form.setValues(jobPresets.product_manager);
                  }}
                  size="sm"
                >
                  Product Manager
                </Button>
                <Box style={{}}>
                  <Tooltip
                    label={
                      <img
                        src="/Senior Lecturer.png"
                        alt="Senior Lecturer Preview"
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
                      variant={selectedPreset === 'senior_lecturer' ? 'filled' : 'light'}
                      onClick={() => {
                        setSelectedPreset('senior_lecturer');
                        form.setValues(jobPresets.senior_lecturer);
                      }}
                      size="sm"
                    >
                      Senior Lecturer
                      <Box
                        style={{
                          background: 'red',
                          color: 'white',
                          padding: '2px 6px',
                          borderRadius: '10px',
                          fontSize: '10px',
                          position: 'absolute',
                          top: '0',
                          right: '0'
                        }}
                      >
                        New!
                      </Box>
                    </Button>
                  </Tooltip>
                </Box>
                <Button
                  variant="subtle"
                  color="gray"
                  onClick={() => {
                    setSelectedPreset('');
                    form.reset();
                  }}
                  size="sm"
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
                      <TextInput
                        label={
                          <FormLabel label="Job Title" tooltip={formTooltips.query} />
                        }
                        placeholder="Enter job title"
                        {...form.getInputProps('job_title')}
                      />
                    </Grid.Col>

                    <Grid.Col span={12}>
                      <Textarea
                        label={<FormLabel label="Job Description" tooltip={formTooltips.job_description} />}
                        placeholder="Enter job description"
                        minRows={2}
                        maxRows={10}
                        autosize
                        {...form.getInputProps('job_description')}
                      />
                    </Grid.Col>



                    <Grid.Col span={6}>
                      <Select
                        label={<FormLabel label="Contract Type" tooltip={formTooltips.contract_type} />}
                        placeholder="Select contract type"
                        data={[
                          { value: 'Full-time', label: 'Full Time' },
                          { value: 'Part-time', label: 'Part Time' },
                          { value: 'Contract', label: 'Contract' },
                        ]}
                        {...form.getInputProps('contract_type')}
                      />
                    </Grid.Col>

                    <Grid.Col span={6}>
                      <Select
                        label={<FormLabel label="Education Level" tooltip={formTooltips.education_level} />}
                        placeholder="Select education level"
                        data={[
                          { value: "Bachelor's", label: "Bachelor's Degree" },
                          { value: "Master's", label: "Master's Degree" },
                          { value: 'PhD', label: 'PhD' },
                        ]}
                        {...form.getInputProps('education_level')}
                      />
                    </Grid.Col>

                    <Grid.Col span={6}>
                      <Select
                        label={<FormLabel label="Seniority" tooltip={formTooltips.seniority} />}
                        placeholder="Select seniority level"
                        data={[
                          { value: 'Entry', label: 'Entry Level' },
                          { value: 'Mid', label: 'Mid Level' },
                          { value: 'Senior', label: 'Senior Level' },
                          { value: 'Lead', label: 'Lead' },
                        ]}
                        {...form.getInputProps('seniority')}
                      />
                    </Grid.Col>

                    <Grid.Col span={6}>
                      <TextInput
                        label={<FormLabel label="Minimum Years of Experience" tooltip={formTooltips.min_years_experience} />}
                        placeholder="Enter years of experience"
                        {...form.getInputProps('min_years_experience')}
                      />
                    </Grid.Col>
                    <Box w="100%" id="countries-select">
                      <Grid.Col span={12} >
                        <FormLabel
                          label="Countries"
                          tooltip={formTooltips.countries}
                          style={{
                            fontSize: 'var(--mantine-font-size-sm)',
                            fontWeight: 500,
                            marginBottom: '0.25rem'
                          }}
                        />
                        <Group>
                          <Button
                            variant={form.values.countries.includes('US') ? 'filled' : 'light'}
                            onClick={() => {
                              const newCountries = form.values.countries.includes('US')
                                ? form.values.countries.filter(c => c !== 'US')
                                : [...form.values.countries, 'US'];
                              form.setFieldValue('countries', newCountries);
                            }}
                            size="sm"
                          >
                            <Text style={{
                              textDecoration: form.values.countries.includes('US') ? 'none' : 'line-through'
                            }}>
                              United States {form.values.countries.includes('US') && '✓'}
                            </Text>
                          </Button>
                          <Button
                            variant={form.values.countries.includes('SG') ? 'filled' : 'light'}
                            onClick={() => {
                              const newCountries = form.values.countries.includes('SG')
                                ? form.values.countries.filter(c => c !== 'SG')
                                : [...form.values.countries, 'SG'];
                              form.setFieldValue('countries', newCountries);
                            }}
                            size="sm"
                          >
                            <Text style={{
                              textDecoration: form.values.countries.includes('SG') ? 'none' : 'line-through'
                            }}>
                              Singapore {form.values.countries.includes('SG') && '✓'}
                            </Text>
                          </Button>
                          <Button
                            variant={form.values.countries.includes('IN') ? 'filled' : 'light'}
                            onClick={() => {
                              const newCountries = form.values.countries.includes('IN')
                                ? form.values.countries.filter(c => c !== 'IN')
                                : [...form.values.countries, 'IN'];
                              form.setFieldValue('countries', newCountries);
                            }}
                            size="sm"
                          >
                            <Text style={{
                              textDecoration: form.values.countries.includes('IN') ? 'none' : 'line-through'
                            }}>
                              India {form.values.countries.includes('IN') && '✓'}
                            </Text>
                          </Button>
                        </Group>
                        {form.errors.countries && (
                          <Text c="red" size="sm">
                            {form.errors.countries}
                          </Text>
                        )}
                      </Grid.Col>

                      {form.values.countries.includes('US') && (
                        <Grid.Col span={12}>
                          <MultiSelect
                            label={<FormLabel label="US Locations" tooltip={formTooltips.location_us} />}
                            placeholder="Select locations in United States"
                            searchable
                            clearable
                            data={[
                              'New York',
                              'San Francisco',
                              'Seattle',
                              'Austin',
                              'Boston',
                              'Los Angeles',
                              'Chicago',
                              'Denver',
                            ]}
                            {...form.getInputProps('location_us')}
                          />
                        </Grid.Col>
                      )}

                      {form.values.countries.includes('IN') && (
                        <Grid.Col span={12}>
                          <MultiSelect
                            label={<FormLabel label="India Locations" tooltip={formTooltips.location_in} />}
                            placeholder="Select locations in India"
                            searchable
                            clearable
                            data={[
                              'Bangalore',
                              'Mumbai',
                              'Delhi',
                              'Hyderabad',
                              'Chennai',
                              'Pune',
                              'Noida',
                              'Gurgaon',
                            ]}
                            {...form.getInputProps('location_in')}
                          />
                        </Grid.Col>
                      )}
                    </Box>
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
                        `Processing ${progress.completed} of ${progress.total} predictions` :
                        `Completed ${progress.total} predictions`
                      }
                    </Text>
                    <Text size="sm" ta="center" c="dimmed">
                      {(timer / 1000).toFixed(2)}s
                    </Text>
                  </Stack>
                </Collapse>
                {Object.keys(predictions).length > 0 ? (
                  <Button
                    onClick={(e) => {
                      e.preventDefault();
                      setPredictions({});
                      setError(null);
                      setFormExpanded(true);
                      setLastPredictionTime(null);
                      setLastEndTime(null);
                      setProgress({ completed: 0, total: 0 });
                      setTimer(0);
                      setShowProgress(false);
                      setTriggerAnalysis(false);
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
                    {isLoading ? 'Calculating...' : 'Submit'}
                  </Button>
                )}

                {Object.keys(predictions).length > 0 && (
                  <Paper p="md" withBorder>
                    <Title order={4} mb="md">Predicted Salaries</Title>
                    <Grid>
                      {['US', 'SG', 'IN'].filter(country =>
                        Object.keys(predictions).some(key => key.startsWith(country))
                      ).map(country => (
                        <Grid.Col key={country} span={{ base: 12, sm: 6, lg: 4 }}>
                          <Card withBorder shadow="sm">
                            <Stack gap="xs">
                              <Group justify="space-between">
                                <Stack gap="xs">
                                  <Title order={5}>
                                    {country === 'US' ? 'United States' :
                                      country === 'SG' ? 'Singapore' :
                                        'India'}
                                  </Title>
                                  {predictions[country]?.salary && (
                                    <Stack gap={0}>
                                      <Text fw={700} size="lg" c="blue" ta="right">
                                        ${predictions[country].salary.toLocaleString()}
                                      </Text>
                                      <Text size="xs" c="dimmed" ta="left">
                                        {(predictions[country].relativeDuration / 1000).toFixed(2)}s
                                      </Text>
                                    </Stack>
                                  )}
                                </Stack>
                              </Group>

                              {/* Location-specific predictions */}
                              {Object.entries(predictions)
                                .filter(([key]) => key.startsWith(country) && key !== country)
                                .map(([key, data]) => {
                                  const location = key.split('-')[1];
                                  return (
                                    <Card key={key} withBorder p="xs">
                                      <Stack gap="xs">
                                        <Text fw={500} size="sm" lineClamp={1}>
                                          {location}
                                        </Text>
                                        <Stack gap={0}>
                                          <Text fw={700} size="sm">
                                            ${data.salary?.toLocaleString() ?? 'N/A'}
                                          </Text>
                                          <Text size="xs" c="dimmed">
                                            {(data.relativeDuration / 1000).toFixed(2)}s
                                          </Text>
                                        </Stack>
                                      </Stack>
                                    </Card>
                                  );
                                })}
                            </Stack>
                          </Card>
                        </Grid.Col>
                      ))}
                    </Grid>
                  </Paper>
                )}

                {Object.keys(predictions).length > 0 && (
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
                              predictions,
                              formValues: form.values,
                            };

                            const updatedSavedPredictions = [...savedPredictions, newSavedPrediction];
                            setSavedPredictions(updatedSavedPredictions);
                            localStorage.setItem('savedPredictions', JSON.stringify(updatedSavedPredictions));

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
                                <Text fw={700} >
                                  {saved.name}
                                </Text>
                                <Text size="xs" c="dimmed">
                                  {new Date(saved.timestamp).toLocaleString()}
                                </Text>
                              </Stack>
                            </Group>

                            <Stack gap="xs">
                              <Text size="sm">
                                {saved.formValues.job_title}
                              </Text>
                              <Text size="xs" c="dimmed">
                                {Object.keys(saved.predictions).length} predictions
                              </Text>
                            </Stack>

                            <Group grow>
                              <Button
                                variant="light"
                                size="xs"
                                onClick={() => {
                                  setPredictions(saved.predictions);
                                  form.setValues(saved.formValues);
                                  setFormExpanded(false);
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
                                  localStorage.setItem('savedPredictions', JSON.stringify(updatedSavedPredictions));
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

        {Object.keys(predictions).length > 0 && (
          <Grid.Col
            span={{ base: 12, md: 6 }}
            style={{
              transition: 'all 0.3s ease-out',
              opacity: Object.keys(predictions).length > 0 ? 1 : 0,
              transform: Object.keys(predictions).length > 0 ? 'translateX(0)' : 'translateX(20px)',
            }}
          >
            <Chat
              predictions={predictions}
              jobDetails={form.values}
              triggerInitialAnalysis={triggerAnalysis}
            />
          </Grid.Col>
        )}
      </Grid>
    </Paper >
  );
}

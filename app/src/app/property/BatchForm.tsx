

import { useState, useRef, useEffect } from 'react';
import {
  Box, Card, Group, Stack, Title, Text, Button,
  FileInput, Progress, Tabs, Table, ActionIcon,
  LoadingOverlay, Tooltip, Switch, Select,
  Grid
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { useDisclosure } from '@mantine/hooks';
import { FileSpreadsheet, Upload, Database, AlertCircle, Download, Edit, Trash, CheckCircle, BrainCircuit } from 'lucide-react';
import Papa from 'papaparse';
import { predictPropertyPrice } from './actions'; // Changed to use single prediction function
import Chat from './Chat'; // Import the Chat component
// Define missing interfaces
interface ValidationResult {
  hasErrors: boolean;
  errors: string[];
}

interface BatchPredictionResult {
  properties: any[];
  predictions: number[];
  completedAt: Date;
  processingTime: number;
}

interface ProgressState {
  current: number;
  total: number;
}

interface DataPreviewProps {
  data: any[];
  validationResults: ValidationResult[];
  onEdit: (index: number) => void;
  onDelete: (index: number) => void;
}

const propertyTypeLabels: Record<string, string> = {
  'h': 'House',
  't': 'Townhouse',
  'u': 'Unit/Apartment'
};


import { DataTable } from 'mantine-datatable';


export default function BatchPredictionForm() {
  const [files, setFiles] = useState<File | null>(null);
  const [parsedData, setParsedData] = useState<any[]>([]);
  const [validationResults, setValidationResults] = useState<ValidationResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [batchResults, setBatchResults] = useState<BatchPredictionResult | null>(null);
  const [viewMode, setViewMode] = useState<'input' | 'results'>('input');
  const [progress, setProgress] = useState<ProgressState>({ current: 0, total: 1 });
  const csvReaderRef = useRef<any>(null);
  const startTimeRef = useRef<number>(0);
  const [showBatchAnalysis, setShowBatchAnalysis] = useState(false);
  const [analysisProperty, setAnalysisProperty] = useState<any | null>(null);

  // Property type labels for display
  const propertyTypeLabels: Record<string, string> = {
    'h': 'House',
    't': 'Townhouse',
    'u': 'Unit/Apartment'
  };

  // File upload handler
  const handleFileUpload = (file: File | null) => {
    setFiles(file);
    if (!file) {
      setParsedData([]);
      return;
    }

    // Parse CSV/Excel
    if (file.type === 'text/csv' || file.type === 'application/vnd.ms-excel') {
      parseCSV(file);
    } else if (file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') {
      parseExcel(file);
    } else {
      notifications.show({
        title: 'Unsupported File Type',
        message: 'Please upload a CSV or Excel file.',
        color: 'red',
      });
    }
  };

  // CSV parsing function
  const parseCSV = async (file: File) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      if (!event.target?.result) return;

      Papa.parse(event.target.result.toString(), {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => {
          const formattedData = formatParsedData(results.data);
          setParsedData(formattedData);
          validateBatchData(formattedData);
        },
        error: (error) => {
          notifications.show({
            title: 'Error Parsing CSV',
            message: error.message,
            color: 'red',
          });
        }
      });
    };
    reader.readAsText(file);
  };

  // Excel parsing function (placeholder - would require xlsx library)
  const parseExcel = (file: File) => {
    notifications.show({
      title: 'Feature Not Available',
      message: 'Excel parsing is not implemented yet. Please use CSV format.',
      color: 'yellow',
    });
  };

  const handleAnalyzeProperty = (property: any, prediction: number) => {
    setAnalysisProperty({
      property,
      predictedPrice: prediction
    });
    setShowBatchAnalysis(true);
  };



  // Format data to match expected structure
  const formatParsedData = (data: any[]) => {
    return data.map(item => ({
      suburb: item.suburb || '',
      rooms: typeof item.rooms === 'number' ? item.rooms : null,
      type: item.type || '',
      method: item.method || '',
      seller: item.seller || '',
      distance: typeof item.distance === 'number' ? item.distance : null,
      bathroom: typeof item.bathroom === 'number' ? item.bathroom : null,
      car: typeof item.car === 'number' ? item.car : null,
      landsize: typeof item.landsize === 'number' ? item.landsize : null,
      buildingArea: typeof item.buildingArea === 'number' ? item.buildingArea : null,
      propertyAge: typeof item.propertyAge === 'number' ? item.propertyAge : null,
      direction: item.direction || '',
      landSizeNotOwned: Boolean(item.landSizeNotOwned),
    }));
  };

  // Validate batch data
  const validateBatchData = (data: any[]) => {
    const results: ValidationResult[] = data.map(item => {
      const errors: string[] = [];

      if (!item.suburb) {
        errors.push('Suburb is required');
      }

      if (item.rooms === null || item.rooms === undefined) {
        errors.push('Number of rooms is required');
      }

      if (!item.type) {
        errors.push('Property type is required');
      }

      return {
        hasErrors: errors.length > 0,
        errors
      };
    });

    setValidationResults(results);
  };

  const DataPreviewTable = ({ data, validationResults, onEdit, onDelete }: DataPreviewProps) => {
    return (
      <DataTable
        withTableBorder
        borderRadius="sm"
        striped
        highlightOnHover
        columns={[
          { accessor: 'index', title: '#', render: (_, index) => index + 1, width: 60 },
          { accessor: 'suburb', title: 'Suburb' },
          { accessor: 'rooms', title: 'Rooms', sortable: true },
          {
            accessor: 'type',
            title: 'Property Type',
            render: (row) => propertyTypeLabels[row.type as keyof typeof propertyTypeLabels] || row.type
          },
          { accessor: 'landsize', title: 'Land Size (m²)', sortable: true },
          {
            accessor: 'status',
            title: 'Status',
            render: (row, index) => validationResults[index]?.hasErrors ? (
              <Tooltip label={validationResults[index].errors.join('\n')}>
                <Group>
                  <AlertCircle size={16} color="red" />
                  <Text size="xs">Has issues</Text>
                </Group>
              </Tooltip>
            ) : (
              <Group>
                <CheckCircle size={16} color="green" />
                <Text size="xs">Valid</Text>
              </Group>
            )
          },
          {
            accessor: 'actions',
            title: 'Actions',
            render: (_, index) => (
              <Group>
                <ActionIcon onClick={() => onEdit(index)}>
                  <Edit size={16} />
                </ActionIcon>
                <ActionIcon color="red" onClick={() => onDelete(index)}>
                  <Trash size={16} />
                </ActionIcon>
              </Group>
            )
          }
        ]}
        records={data}
        rowStyle={(row, index) => ({
          backgroundColor: validationResults[index]?.hasErrors ? 'rgba(255, 0, 0, 0.05)' : undefined
        })}
        emptyState={<Text >No properties loaded. Upload a CSV file to get started.</Text>}
      />
    );
  };

  // Updated batch prediction handler that processes properties one by one
  const handleBatchPrediction = async () => {
    if (parsedData.length === 0) {
      notifications.show({
        title: 'No Data',
        message: 'Please upload or enter property data before processing.',
        color: 'red',
      });
      return;
    }

    const hasValidationErrors = validationResults.some(result => result.hasErrors);
    if (hasValidationErrors) {
      const confirmed = window.confirm(
        'Some properties have validation errors. Would you like to proceed with only the valid properties?'
      );
      if (!confirmed) return;
    }

    // Set both states at the start of processing
    setIsProcessing(true);
    setIsLoading(true);
    setProgress({ current: 0, total: parsedData.length });
    startTimeRef.current = performance.now();

    try {
      // Get only valid properties
      const validProperties = parsedData.filter((_, index) =>
        !validationResults[index]?.hasErrors
      );

      // Process properties one by one
      const results = [];
      const errors = [];

      for (let i = 0; i < validProperties.length; i++) {
        const property = validProperties[i];

        try {
          // Update progress
          setProgress({
            current: i,
            total: validProperties.length
          });

          // Process this property
          const result = await predictPropertyPrice(property);

          if (result.success && result.data) {
            results.push({
              property: i,
              prediction: result.data.prediction,
              propertyData: property
            });
          } else {
            errors.push({
              property: i,
              error: result.error || 'Unknown error',
              propertyData: property
            });
          }
        } catch (error) {
          console.error(`Error processing property ${i}:`, error);
          errors.push({
            property: i,
            error: error instanceof Error ? error.message : 'Unknown error',
            propertyData: property
          });
        }

        // Small delay to prevent UI freezing
        await new Promise(resolve => setTimeout(resolve, 50));
      }

      // Final progress update
      setProgress({
        current: validProperties.length,
        total: validProperties.length
      });

      setBatchResults({
        properties: validProperties,
        predictions: results.map(r => r.prediction),
        completedAt: new Date(),
        processingTime: performance.now() - startTimeRef.current
      });

      setViewMode('results');

      // Display appropriate notification
      if (errors.length === 0) {
        notifications.show({
          title: 'Batch Processing Complete',
          message: `Successfully processed all ${results.length} properties`,
          color: 'green',
        });
      } else {
        notifications.show({
          title: 'Batch Processing Completed with Errors',
          message: `Processed ${results.length} properties, ${errors.length} failed`,
          color: 'yellow',
        });
      }
    } catch (error: any) {
      console.error('Batch processing error:', error);
      notifications.show({
        title: 'Processing Error',
        message: error.message || 'An unexpected error occurred during batch processing',
        color: 'red',
      });
    } finally {
      // Update both states at the end of processing
      setIsProcessing(false);
      setIsLoading(false);
    }
  };

  // Handle row editing
  const handleEditRow = (index: number) => {
    // In a real implementation, you might open a modal or form to edit the property
    alert(`Edit functionality would open a form for property ${index + 1}`);
  };

  // Handle row deletion
  const handleDeleteRow = (index: number) => {
    const newData = [...parsedData];
    newData.splice(index, 1);
    setParsedData(newData);

    const newValidation = [...validationResults];
    newValidation.splice(index, 1);
    setValidationResults(newValidation);
  };

  // Component JSX
  return (
    <Box>
      <Stack >
        {viewMode === 'input' ? (
          <>
            <Card withBorder p="md">
              <Stack >
                <Title order={3}>Batch Property Price Prediction</Title>
                <Text size="sm">
                  Upload a CSV file with multiple properties to predict prices in batch.
                </Text>

                <FileInput
                  label="Upload property data"
                  placeholder="Upload CSV or Excel file"
                  accept=".csv,.xlsx,.xls"
                  leftSection={<Upload size={14} />}
                  value={files}
                  onChange={handleFileUpload}
                />

                <Group >
                  <Button
                    variant="outline"
                    leftSection={<FileSpreadsheet size={16} />}
                    component="a"
                    href="/template.csv"
                    download
                  >
                    Download Template
                  </Button>

                  <Button
                    disabled={parsedData.length === 0}
                    loading={isProcessing}
                    onClick={handleBatchPrediction}
                  >
                    {isProcessing ? 'Processing...' : 'Process Batch'}
                  </Button>
                </Group>
              </Stack>
            </Card>

            {parsedData.length > 0 && (
              <Card withBorder p="md">
                <Stack >
                  <Group >
                    <Title order={4}>Data Preview</Title>
                    <Text size="sm">{parsedData.length} properties loaded</Text>
                  </Group>

                  <DataPreviewTable
                    data={parsedData}
                    validationResults={validationResults}
                    onEdit={handleEditRow}
                    onDelete={handleDeleteRow}
                  />
                </Stack>
              </Card>
            )}
          </>
        ) : (
          // Results view
          batchResults && (
            <>
              <Grid gutter="md">
                <Grid.Col span={showBatchAnalysis ? 7 : 12}>
                  <Card withBorder p="md">
                    <Stack >
                      <Group >
                        <Title order={3}>Batch Results</Title>
                        <Text size="sm">Processing time: {(batchResults.processingTime / 1000).toFixed(2)} seconds</Text>
                      </Group>

                      <DataTable
                        withTableBorder
                        borderRadius="sm"
                        striped
                        highlightOnHover
                        columns={[
                          { accessor: 'index', title: '#', render: (_, index) => index + 1, width: 60 },
                          { accessor: 'suburb', title: 'Suburb', sortable: true },
                          {
                            accessor: 'type',
                            title: 'Property Type',
                            sortable: true,
                            render: (row) => propertyTypeLabels[row.type as keyof typeof propertyTypeLabels] || row.type
                          },
                          { accessor: 'rooms', title: 'Rooms', sortable: true },
                          {
                            accessor: 'landsize',
                            title: 'Land Size',
                            render: (row) => `${row.landsize} m²`,
                            sortable: true
                          },
                          {
                            accessor: 'prediction',
                            title: 'Predicted Price',
                            render: (_, index) => `$${batchResults.predictions[index]?.toLocaleString() || 'N/A'}`,
                            sortable: true
                          },
                          {
                            accessor: 'actions',
                            title: 'Actions',
                            width: 80,
                            render: (row, index) => (
                              <ActionIcon
                                color={analysisProperty?.property === row ? 'yellow' : 'blue'}
                                onClick={() => handleAnalyzeProperty(row, batchResults.predictions[index])}
                                title="Analyze with AI"
                              >
                                <BrainCircuit size={16} />
                              </ActionIcon>
                            )
                          }
                        ]}
                        records={batchResults.properties}
                        rowStyle={(row) => ({
                          backgroundColor: analysisProperty?.property === row ? 'rgba(0, 100, 250, 0.05)' : undefined
                        })}
                        minHeight={300}
                        noRecordsText="No prediction results available"
                      />

                      <Group >
                        <Text>Successfully predicted prices for {batchResults.properties.length} properties</Text>
                        <Button onClick={() => setViewMode('input')}>
                          Start New Batch
                        </Button>
                      </Group>
                    </Stack>
                  </Card>
                </Grid.Col>

                {/* Chat component column that shows when analysis is requested */}
                {showBatchAnalysis && (
                  <Grid.Col
                    span={{ base: 12, md: 5 }}
                    style={{
                      transition: 'all 0.3s ease-out',
                      opacity: analysisProperty ? 1 : 0,
                      transform: analysisProperty ? 'translateX(0)' : 'translateX(20px)',
                      height: '100%'
                    }}
                  >
                    {analysisProperty && (
                      <Chat
                        predictedPrice={analysisProperty.predictedPrice}
                        duration={batchResults.processingTime}
                        propertyDetails={analysisProperty.property}
                        triggerInitialAnalysis={true}
                        isBatchMode={true}
                      />
                    )}
                  </Grid.Col>
                )}
              </Grid>

              {!showBatchAnalysis && batchResults && (
                <BatchSummaryCard
                  properties={batchResults.properties}
                  predictions={batchResults.predictions}
                />
              )}


              {/* Button to toggle analysis panel if no property is selected */}
              {!showBatchAnalysis && (
                <Button
                  variant="light"
                  mt="md"
                  leftSection={<BrainCircuit size={16} />}
                  onClick={() => {
                    if (batchResults.properties.length > 0 && batchResults.predictions.length > 0) {
                      handleAnalyzeProperty(
                        batchResults.properties[0],
                        batchResults.predictions[0]
                      );
                    }
                  }}
                >
                  Show AI Analysis for First Property
                </Button>
              )}
            </>
          )
        )}

        {isProcessing && (
          <Card withBorder p="md">
            <Stack >
              <Text>Processing batch prediction...</Text>
              <Progress
                value={(progress.current / progress.total) * 100}
                animated={isLoading}
                size="lg"
              />
              <Text size="xs" >
                {isLoading
                  ? `Processing property ${progress.current + 1} of ${progress.total}...`
                  : "Processing complete!"}
              </Text>
            </Stack>
          </Card>
        )}
      </Stack>
    </Box>
  );
}


function BatchSummaryCard({ properties, predictions }: { properties: any[], predictions: number[] }) {
  // Calculate batch statistics
  const avgPrice = predictions.reduce((sum, price) => sum + price, 0) / predictions.length;
  const minPrice = Math.min(...predictions);
  const maxPrice = Math.max(...predictions);
  const priceRange = maxPrice - minPrice;
  const propertyTypeLabels: Record<string, string> = {
    'h': 'House',
    't': 'Townhouse',
    'u': 'Unit/Apartment'
  };


  // Group properties by type
  const typeGroups = properties.reduce((groups: any, property, index) => {
    const type = property.type;
    if (!groups[type]) {
      groups[type] = [];
    }
    groups[type].push({
      ...property,
      prediction: predictions[index]
    });
    return groups;
  }, {});

  return (
    <Card withBorder p="md" mt="md">
      <Stack >
        <Title order={4}>Batch Analysis Summary</Title>

        <Group >
          <Text size="sm">Average Price: <Text span fw={600}>${avgPrice.toLocaleString()}</Text></Text>
          <Text size="sm">Price Range: <Text span fw={600}>${minPrice.toLocaleString()} - ${maxPrice.toLocaleString()}</Text></Text>
          <Text size="sm">Properties: <Text span fw={600}>{properties.length}</Text></Text>
        </Group>

        <Title order={5} mt="xs">Property Types</Title>
        <DataTable
          withTableBorder
          borderRadius="sm"
          striped
          highlightOnHover
          columns={[
            {
              accessor: 'type',
              title: 'Type',
              render: (row) => propertyTypeLabels[row.type as keyof typeof propertyTypeLabels] || row.type
            },
            { accessor: 'count', title: 'Count', sortable: true },
            {
              accessor: 'avgPrice',
              title: 'Avg. Price',
              render: (row) => `$${row.avgPrice.toLocaleString()}`,
              sortable: true
            },
            {
              accessor: 'priceRange',
              title: 'Price Range',
              render: (row) => `$${row.minPrice.toLocaleString()} - $${row.maxPrice.toLocaleString()}`
            }
          ]}
          records={Object.entries(typeGroups).map(([type, props]: [string, any[]]) => {
            const typePrices = props.map(p => p.prediction);
            const typeAvg = typePrices.reduce((sum, price) => sum + price, 0) / typePrices.length;
            const typeMin = Math.min(...typePrices);
            const typeMax = Math.max(...typePrices);

            return {
              type,
              count: props.length,
              avgPrice: typeAvg,
              minPrice: typeMin,
              maxPrice: typeMax
            };
          })}
          minHeight={150}
          noRecordsText="No property type data available"
        />
      </Stack>
    </Card>
  );
}


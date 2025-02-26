'use client';

import { useState } from 'react';
import { 
  Container, 
  Title, 
  Text, 
  Button, 
  Group, 
  Stack, 
  TextInput, 
  NumberInput, 
  Select, 
  SimpleGrid, 
  Paper, 
  Divider 
} from '@mantine/core';
import axios from 'axios';
import brandModels from './brand_models';

export default function CarPricePredictor() {
  const [brandModel, setBrandModel] = useState('');
  const [location, setLocation] = useState('');
  const [year, setYear] = useState<number | undefined>(undefined);
  const [kilometersDriven, setKilometersDriven] = useState<number | undefined>(undefined);
  const [fuelType, setFuelType] = useState('');
  const [transmission, setTransmission] = useState('');
  const [ownerType, setOwnerType] = useState('');
  const [mileage, setMileage] = useState<number | undefined>(undefined);
  const [power, setPower] = useState<number | undefined>(undefined);
  const [seats, setSeats] = useState<number | undefined>(undefined);
  const [predictedPrice, setPredictedPrice] = useState<string | null>(null);

  const locations = ["Ahmedabad", "Bangalore", "Chennai", "Coimbatore", "Delhi", "Hyderabad", "Jaipur", "Kochi", "Kolkata", "Mumbai", "Pune"];
  const fuelTypes = ["Petrol", "Diesel", "Other"];
  const transmissions = ["Manual", "Automatic"];
  const ownerTypes = ["First", "Second", "Third & Above"];

  const loadExample = () => {
    setBrandModel("Maruti Swift VDI");
    setLocation("Mumbai");
    setYear(2018);
    setKilometersDriven(30000);
    setFuelType("Petrol");
    setTransmission("Manual");
    setOwnerType("First");
    setMileage(18.5);
    setPower(82.0);
    setSeats(5);
    setPredictedPrice(null); // Reset output when loading example
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
  
    const formData = {
      input: [ // Wrap data inside an "input" array
        {
          brand_model: brandModel,
          location,
          year,
          kilometers_driven: kilometersDriven,
          fuel_type: fuelType,
          transmission,
          owner_type: ownerType,
          mileage,
          power,
          seats
        }
      ]
    };
  
    try {
      const response = await axios.post('https://car-price-predictor-m9l0-1531a252.mt-guc1.bentoml.ai/predict', formData);
  
      // Update this to handle multiple predictions properly
      if (response.data.length > 0 && response.data[0].predicted_price) {
        setPredictedPrice(`₹ ${response.data[0].predicted_price.toFixed(2)}`);
      } else {
        setPredictedPrice("Error in prediction.");
      }
    } catch (error) {
      console.error("Prediction error", error);
      setPredictedPrice("Error in prediction.");
    }
  };
  

  return (
    <Container size="lg" py="xl">
      <Paper shadow="md" p="xl" radius="md" withBorder>
        <Title order={1} mb="md">
          Used Car Price Predictor
        </Title>
        <Text c="dimmed" mb="lg">
          Enter your car details to get an accurate price prediction.
        </Text>

        <form onSubmit={handleSubmit}>
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="md">
            <Select
              label="Brand & Model"
              placeholder="Select brand & model"
              data={brandModels}
              value={brandModel}
              onChange={(value) => setBrandModel(value || '')}
              searchable
              required
            />
            <Select
              label="Location"
              placeholder="Select location"
              data={locations}
              value={location}
              onChange={(value) => setLocation(value || '')}
              required
            />
            <NumberInput
              label="Year"
              placeholder="e.g., 2015"
              min={2000}
              max={2019}
              value={year}
              onChange={(value) => setYear(typeof value === "number" ? value : undefined)}
              required
            />

            <NumberInput
              label="Kilometers Driven"
              placeholder="e.g., 50000"
              min={0}
              value={kilometersDriven}
              onChange={(value) => setKilometersDriven(typeof value === "number" ? value : undefined)}
              required
            />
            <Select
              label="Fuel Type"
              placeholder="Select fuel type"
              data={fuelTypes}
              value={fuelType}
              onChange={(value) => setFuelType(value || '')}
              required
            />
            <Select
              label="Transmission"
              placeholder="Select transmission"
              data={transmissions}
              value={transmission}
              onChange={(value) => setTransmission(value || '')}
              required
            />
            <Select
              label="Owner Type"
              placeholder="Select owner type"
              data={ownerTypes}
              value={ownerType}
              onChange={(value) => setOwnerType(value || '')}
              required
            />
            <NumberInput
              label="Mileage (km/l)"
              placeholder="e.g., 18"
              min={0}
              value={mileage}
              onChange={(value) => setMileage(typeof value === "number" ? value : undefined)}
              required
            />

            <NumberInput
              label="Power (bhp)"
              placeholder="e.g., 120"
              min={0}
              value={power}
              onChange={(value) => setPower(typeof value === "number" ? value : undefined)}
              required
            />

            <NumberInput
              label="Seats"
              placeholder="e.g., 5"
              min={1}
              max={10}
              value={seats}
              onChange={(value) => setSeats(typeof value === "number" ? value : undefined)}
              required
            />
          </SimpleGrid>

          <Group mt="xl">
            <Button 
              type="submit" 
              size="lg" 
              variant="gradient" 
              gradient={{ from: 'blue', to: 'cyan' }}
            >
              Predict Price
            </Button>
            <Button 
              size="lg" 
              variant="outline" 
              color="gray" 
              onClick={loadExample}
            >
              Load Example
            </Button>
          </Group>
        </form>

        {predictedPrice !== null && (
          <>
            <Divider my="xl" />
            <Paper shadow="xs" p="md" withBorder style={{ textAlign: 'center', backgroundColor: '#f8f9fa' }}>
              <Text size="lg" c={predictedPrice.includes("Error") ? "red" : "green"}>
                {predictedPrice.includes("Error") ? "⚠️ Prediction Failed" : `Predicted Price: ${predictedPrice} Lakhs`}
              </Text>
            </Paper>
          </>
        )}
      </Paper>
    </Container>
  );
}

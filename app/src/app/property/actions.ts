

'use server'


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

// Interface for property data in batch predictions
export interface PropertyData {
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

// Interface for the prediction result of a single property
export interface PredictionResult {
  prediction: number;
  [key: string]: any; // For any additional fields the API might return
}

// Interface for the batch prediction response
export interface BatchPredictionResponse {
  success: boolean;
  data?: PredictionResult[]; // Optional, present when success is true
  error?: string; // Optional, present when success is false
  processingTime: number; // Processing time in milliseconds
}

export async function predictPropertyPrice(payload: FormValues) {
  try {
    const response = await fetch('https://housing-price-prediction-csdc-f774f0f9.mt-guc1.bentoml.ai/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        housing_features: {
          Suburb: payload.suburb,
          Rooms: payload.rooms || 0,
          Type: payload.type,
          Method: payload.method,
          Seller: payload.seller,
          Distance: payload.distance || 0,
          Bathroom: payload.bathroom || 0,
          Car: payload.car || 0,
          Landsize: payload.landsize || 0,
          BuildingArea: payload.buildingArea || undefined,
          PropertyAge: payload.propertyAge || undefined,
          Direction: payload.direction,
          LandSizeNotOwned: payload.landSizeNotOwned,
        },
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    console.error('Prediction error:', error);
    return { success: false, error: 'Failed to predict property price' };
  }
}



export async function predictPropertiesBatch(properties: PropertyData[]): Promise<BatchPredictionResponse> {
  // Start measuring response time
  const startTime = Date.now();

  try {
    // Transform property data to match the expected BentoML format
    const transformedProperties = properties.map(property => ({
      Suburb: property.suburb,
      Rooms: property.rooms || 0,
      Type: property.type,
      Method: property.method,
      Seller: property.seller,
      Distance: property.distance || 0,
      Bathroom: property.bathroom || 0,
      Car: property.car || 0,
      Landsize: property.landsize || 0,
      BuildingArea: property.buildingArea || undefined,
      PropertyAge: property.propertyAge || undefined,
      Direction: property.direction,
      LandSizeNotOwned: property.landSizeNotOwned,
    }));

    // The correct structure based on the BentoML service pattern
    const requestBody = {
      batch_features: {
        features: transformedProperties
      }
    };

    console.log('Sending batch request:', JSON.stringify(requestBody, null, 2));

    const response = await fetch('https://housing-price-prediction-csdc-f774f0f9.mt-guc1.bentoml.ai/batch_predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      let errorData;
      try {
        errorData = JSON.parse(errorText);
      } catch (e) {
        errorData = { error: errorText };
      }

      console.error('Error response:', errorData);
      return {
        success: false,
        error: errorData.error || `Failed to predict property prices (Status: ${response.status})`,
        processingTime: Date.now() - startTime,
      };
    }

    const data = await response.json();
    console.log('Batch prediction success:', data);

    return {
      success: true,
      data: data.predictions.map((price: number) => ({ prediction: price })),
      processingTime: Date.now() - startTime,
    };
  } catch (error) {
    console.error('Error in batch prediction:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'An unexpected error occurred during batch prediction',
      processingTime: Date.now() - startTime,
    };
  }
}





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



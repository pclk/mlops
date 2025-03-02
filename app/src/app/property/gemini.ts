
import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.NEXT_PUBLIC_GEMINI_API_KEY || '');

interface PropertyDetails {
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

interface PredictionResult {
  price: number | null;
  duration: number;
}

export async function* streamGeminiResponse(
  userInput: string,
  prediction: PredictionResult,
  propertyDetails: PropertyDetails,
  messageHistory: { content: string; sender: 'user' | 'bot' }[],
  modelConfig: {
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
) {
  const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

  // Convert message history to a chat-like format
  const conversationHistory = messageHistory
    .slice(1) // Skip the initial greeting
    .map(msg => `${msg.sender === 'user' ? 'User' : 'Assistant'}: ${msg.content}`)
    .join('\n\n');

  // Format property type label
  const propertyTypeLabel = {
    'h': 'House',
    't': 'Townhouse',
    'u': 'Unit/Apartment'
  }[propertyDetails.type] || propertyDetails.type;

  // Format selling method label
  const sellingMethodLabel = {
    'S': 'Sold',
    'SP': 'Sold Prior',
    'PI': 'Passed In',
    'SA': 'Sold After Auction',
    'VB': 'Vendor Bid'
  }[propertyDetails.method] || propertyDetails.method;

  const prompt = `
    You are a property price prediction assistant. You provide helpful, detailed answers about property valuations, market trends, and investment potential.
    When asked about buying, selling, or investment advice, you provide practical, actionable suggestions.
    
    Current property details:
    - Location: ${propertyDetails.suburb}
    - Property Type: ${propertyTypeLabel}
    - Rooms: ${propertyDetails.rooms || 'Not specified'}
    - Bathrooms: ${propertyDetails.bathroom || 'Not specified'}
    - Car Spaces: ${propertyDetails.car || 'Not specified'}
    - Distance from CBD: ${propertyDetails.distance ? `${propertyDetails.distance} km` : 'Not specified'}
    - Land Size: ${propertyDetails.landsize ? `${propertyDetails.landsize} m²` : 'Not specified'}
    - Building Area: ${propertyDetails.buildingArea ? `${propertyDetails.buildingArea} m²` : 'Not specified'}
    - Property Age: ${propertyDetails.propertyAge ? `${propertyDetails.propertyAge} years` : 'Not specified'}
    - Direction Faced: ${propertyDetails.direction || 'Not specified'}
    - Land Fully Owned: ${propertyDetails.landSizeNotOwned ? 'No' : 'Yes'}
    - Sale Method: ${sellingMethodLabel}
    - Real Estate Agency: ${propertyDetails.seller || 'Not specified'}

    Market Configuration:
    Property Market Factors:
    - Current Interest Rate: ${modelConfig.propertyMarketFactors.interestRate}%
    - Market Growth Rate: ${modelConfig.propertyMarketFactors.marketGrowth}%
    - Inflation Rate: ${modelConfig.propertyMarketFactors.inflationRate}%

    Location Value Factors:
    - Urban Premium: ${modelConfig.locationFactors.urbanPremium}%
    - Suburban Discount: ${modelConfig.locationFactors.suburbanDiscount}%
    - Rural Discount: ${modelConfig.locationFactors.ruralDiscount}%

    The property price prediction is: $${prediction.price?.toLocaleString() || 'Not available'}
    Prediction calculated in: ${(prediction.duration / 1000).toFixed(2)} seconds

    Information related to this project:
    - The purpose of this project is to accurately and quickly predict property prices based on property features.
    - The predictive model is trained on historical property sales data from Melbourne, Australia.
    - The model considers multiple factors including location, property type, size, and amenities.
    - The model has an average error margin of approximately 5-10%.
    - The user has the option to press "Start new prediction" to create a new property profile.
    - The user can save their predictions and load them anytime for comparison.
    - Property values are expressed in Australian Dollars (AUD).

    Please provide helpful, detailed answers about this property price prediction.
    If asked about buying, selling, or investment advice, provide practical, actionable suggestions.
    When providing comparisons, use statistics and percentages to arrive at a final recommendation based on multiple factors.

    You're encouraged to use markdown, however, do not go beyond lists, as the markdown renderer is primitive. If you use markdown tables, the user will be distraught, as it is not rendered at all.

    Previous conversation:
    ${conversationHistory}

    User: ${userInput}
  `;

  try {
    const result = await model.generateContentStream([prompt]);

    let accumulatedText = '';
    for await (const chunk of result.stream) {
      const chunkText = chunk.text();
      accumulatedText += chunkText;
      yield accumulatedText;
    }
  } catch (error) {
    console.error('Gemini API Error:', error);
    throw error;
  }
}


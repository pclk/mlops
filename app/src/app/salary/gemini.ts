import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.NEXT_PUBLIC_GEMINI_API_KEY || '');

type CountryCode = 'US' | 'SG' | 'IN';

interface JobDetails {
  job_title: string;
  contract_type: string;
  education_level: string;
  seniority: string;
  min_years_experience: number;
  countries: CountryCode[];
  location_us: string[];
  location_in: string[];
}

interface PredictionData {
  salary: number;
  [key: string]: any;
}

interface Predictions {
  [model: string]: PredictionData;
}

export async function* streamGeminiResponse(
  userInput: string,
  predictions: Predictions,
  jobDetails: JobDetails,
  messageHistory: { content: string; sender: 'user' | 'bot' }[],
  modelConfig: {
    medianSalary: { US: number; SG: number; IN: number };
    costOfLiving: { US: number; SG: number; IN: number };
  }
) {
  const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

  // Convert message history to a chat-like format
  const conversationHistory = messageHistory
    .slice(1) // Skip the initial greeting
    .map(msg => `${msg.sender === 'user' ? 'User' : 'Assistant'}: ${msg.content}`)
    .join('\n\n');

  const prompt = `
    You are a salary prediction assistant. You provide helpful, concise answers about salary predictions and job details.
    When asked about negotiation or career advice, you provide practical, actionable suggestions.
    
    Current job details:
    - Job Title: ${jobDetails.job_title}
    - Contract Type: ${jobDetails.contract_type}
    - Education Level: ${jobDetails.education_level}
    - Seniority: ${jobDetails.seniority}
    - Experience Required: ${jobDetails.min_years_experience} years

    Location Details:
    ${jobDetails.countries.map((country: CountryCode) => {
    const locations = country === 'US' ? jobDetails.location_us
      : country === 'IN' ? jobDetails.location_in
        : [];
    return `${country === 'US' ? 'United States'
      : country === 'SG' ? 'Singapore'
        : 'India'}${locations.length ? `: ${locations.join(', ')}` : ''}`
  }).join('\n    ')}
    
    Model Configuration:
    Median Salaries per year (USD):
    ${jobDetails.countries.map((country: CountryCode) => {
    return `${country === 'US' ? '- United States'
      : country === 'SG' ? '- Singapore'
        : '- India'}: $${modelConfig.medianSalary[country].toLocaleString()}`
  }).join('\n    ')}

    Cost of Living expenses per year (USD):
    ${jobDetails.countries.map((country: CountryCode) => {
    return `${country === 'US' ? '- United States'
      : country === 'SG' ? '- Singapore'
        : '- India'}: $${modelConfig.costOfLiving[country].toLocaleString()}`
  }).join('\n    ')}

    The salary predictions are:
    ${Object.entries(predictions)
      .map(([model, data]: [string, PredictionData]) => `${model}: $${data.salary}`)
      .join('\n')}

    Information related to this project:
    - The purpose of this project is to accurately and quickly predict salaries across countries and locations.
    - The predictive model is trained with DistilBERT as the feature extractor, on glassdoor job posting data.
    - The data is collected in Jan 2025.
    - The model has an average error margin of approximately 20K USD/year.
    - Singapore only has 1 location because that's how it is in job postings. They just list as Singapore.
    - The user has the option to press "Start new prediction" at the top left corner, and you can provide some details as to how to fill them out.
    - The user has the option to save their prediction and load it anytime.
    - The user very likely currently resides in Singapore, though you're not 100% sure.

    Please provide helpful, concise answers about these salary predictions and job details.
    If asked about negotiation or career advice, provide practical, actionable suggestions.
    When asked to "compare these two opportunities" or similar, read the conversation history to find any two job opportunities and compare them with statistics and percentages to arrive at a final recommendation at which opportunities to consider based on some factors.

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

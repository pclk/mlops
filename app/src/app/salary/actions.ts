'use server'

interface SalaryPredictionPayload {
  job_description: string;
  job_title: string;
  query: string;
  contract_type: string;
  education_level: string;
  seniority: string;
  min_years_experience: string;
  location_us: string[];
  location_sg: string[];
  location_in: string[];
}

export async function predictSalary(payload: SalaryPredictionPayload, country_code: string, location: string) {
  try {
    const response = await fetch('https://salary-prediction-491899619233.asia-southeast1.run.app/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        job_description: payload.job_description,
        job_title: payload.job_title,
        query: payload.query,
        contract_type: payload.contract_type,
        education_level: payload.education_level,
        seniority: payload.seniority,
        min_years_experience: payload.min_years_experience,
        country: country_code,
        location: location
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    console.error('Prediction error:', error);
    return { success: false, error: 'Failed to predict salary' };
  }
}

export const formTooltips = {
  job_title: "The specific title of the job position you're analyzing",
  query: "A broader search term that helps match similar job titles",
  job_description: "Key responsibilities and requirements of the role",
  contract_type: "The type of employment arrangement",
  education_level: "Minimum education qualification required",
  seniority: "Level of experience and responsibility in the role",
  min_years_experience: "Minimum years of relevant work experience required",
  countries: "Select countries for salary prediction. \nThe countries listed here will be looped in the prediction.",
  location_us: "Specific cities or regions within the United States. \nIf there are no locations selected, the country will be predicted in general.",
  location_sg: "Specific regions within Singapore. \nIf there are no locations selected, the country will be predicted in general.",
  location_in: "Specific cities or regions within in India. \nIf there are no locations selected, the country will be predicted in general."
};

export const jobPresets = {
  'software_engineer': {
    job_title: "Software Engineer",
    query: "Python Developer",
    job_description: "Looking for an experienced Python developer with ML knowledge",
    contract_type: "Full-time",
    education_level: "Bachelor's",
    seniority: "Senior",
    min_years_experience: "5"
  },
  'data_scientist': {
    job_title: "Data Scientist",
    query: "ML Engineer",
    job_description: "Seeking a data scientist with expertise in machine learning and statistical analysis",
    contract_type: "Full-time",
    education_level: "Master's",
    seniority: "Senior",
    min_years_experience: "5"
  },
  'product_manager': {
    job_title: "Product Manager",
    query: "Technical Product Manager",
    job_description: "Looking for an experienced product manager to lead technical product initiatives",
    contract_type: "Full-time",
    education_level: "Bachelor's",
    seniority: "Senior",
    min_years_experience: "7"
  }
};

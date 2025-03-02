export const formTooltips = {
  job_title: "The specific title of the job position you're analyzing",
  query: "A broader search term that helps match similar job titles",
  job_description: "Key responsibilities and requirements of the role",
  contract_type: "The type of employment arrangement",
  education_level: "Minimum education qualification required",
  seniority: "Level of experience and responsibility in the role",
  min_years_experience: "Minimum years of relevant work experience required",
  countries: "Select countries for salary prediction. \nThe countries listed here will be looped in the prediction. \n Singapore is a small island, so there will be no location selection",
  location_us: "Specific cities or regions within the United States. \nIf there are no locations selected, the country will be predicted in general.",
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
  },
  'senior_lecturer': {
    job_title: "Lecturer / Senior Lecturer (Software Engineering) - School of Information Technology",
    query: "Senior Lecturer",
    job_description: "About the job: You will be a subject matter expert in Software Engineering and will play an active role in the academic, professional and personal development of your learners to prepare them for work & life.\n\nKey Responsibilities:\n• Develop courses including curriculum development, course delivery and project supervision\n• Spearhead and identify new opportunities, initiatives and collaborations with industry\n• Plan, lead, manage and undertake project development with industry\n\nRequirements:\n• At least 3 years of relevant experience in related domain\n• Relevant qualification in Computer Science, Engineering, Information Technology or related fields\n• Proficient in programming languages such as Python and Java/C#\n• Experience with databases (Oracle, MS SQL, MySQL, NoSQL) including SQL scripting and modeling\n• Extensive experience in managing software engineering projects\n• Strong knowledge of software architecture practices, design patterns, and OOP\n• Experience with agile methodologies (SCRUM)\n• Proficient in DevOps practices\n• Knowledge in software testing and security testing\n• Professional certifications (PMP or CITPM) preferred",
    contract_type: "Full-time",
    education_level: "Bachelor",
    seniority: "Senior",
    min_years_experience: "3"
  }
};

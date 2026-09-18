import os

from dotenv import load_dotenv
from openai import OpenAI

from utils.schemas import JobMatchAnalysis


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY is not set. Please check your .env file."
    )

client = OpenAI(api_key=api_key)


def match_resume_to_job(
    resume_text: str,
    job_description: str
) -> JobMatchAnalysis:

    prompt = f"""
You are an expert technical recruiter and ATS resume reviewer.

Compare the candidate's resume against the provided job description.

Rules:

- Give an overall match score from 0 to 100.
- Base the analysis ONLY on information explicitly present in the resume.
- Do not invent candidate experience.
- Do not assume that a skill is present just because it is related to another skill.
- Distinguish between demonstrated experience and skills merely listed.
- Identify important missing skills from the job description.
- Identify experience that directly matches the job requirements.
- Identify meaningful experience gaps.
- Recommend realistic resume improvements.
- Do not recommend removing genuine experience.
- The assessment must be based on the evidence provided.

RESUME
-------------------------
{resume_text}
-------------------------

JOB DESCRIPTION
-------------------------
{job_description}
-------------------------

Return a structured job-match analysis.
"""

    response = client.responses.parse(
        model="gpt-5.6-luna",
        input=prompt,
        text_format=JobMatchAnalysis,
    )

    return response.output_parsed
import os

from dotenv import load_dotenv
from openai import OpenAI

from utils.schemas import ResumeAnalysis


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY is not set. Please check your .env file."
    )

client = OpenAI(api_key=api_key)


def analyze_resume(resume_text: str) -> ResumeAnalysis:

    prompt = f"""
You are an expert technical recruiter and ATS resume reviewer.

Analyze the resume below.

Rules:

- Give an ATS compatibility score from 0 to 100.
- Estimate years of experience only from information explicitly supported
  by the resume.
- Base your analysis only on the resume.
- Do not invent experience, skills, certifications, or projects.
- Identify realistic technical skill gaps.
- Recommend practical skills that would improve employability.
- Recommend job roles based on demonstrated experience.

Resume:
-------------------------
{resume_text}
-------------------------
"""

    response = client.responses.parse(
        model="gpt-5.6-luna",
        input=prompt,
        text_format=ResumeAnalysis,
    )

    return response.output_parsed
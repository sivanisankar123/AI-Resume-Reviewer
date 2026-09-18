from pydantic import BaseModel, Field


class ResumeAnalysis(BaseModel):

    ats_score: int = Field(
        ge=0,
        le=100,
        description="ATS compatibility score from 0 to 100"
    )

    years_of_experience: float = Field(
        ge=0,
        le=60,
        description="Years of professional experience explicitly supported by the resume"
    )

    strengths: list[str] = Field(
        description="Key strengths identified in the resume"
    )

    weaknesses: list[str] = Field(
        description="Important weaknesses or gaps in the resume"
    )

    missing_skills: list[str] = Field(
        description="Technical skills missing or insufficiently demonstrated"
    )

    recommended_skills: list[str] = Field(
        description="Technical skills the candidate should learn or strengthen"
    )

    suitable_roles: list[str] = Field(
        description="Job roles suitable for the candidate based on the resume"
    )

    improvement_suggestions: list[str] = Field(
        description="Specific recommendations to improve the resume"
    )

    overall_assessment: str = Field(
        description="Overall assessment of the resume"
    )

    top_recommendation: str = Field(
        description="Give the candidate the single most important recommendation for improving their career/resume."
    )


class JobMatchAnalysis(BaseModel):

    match_score: int = Field(
        ge=0,
        le=100,
        description="Overall match score between the resume and job description"
    )

    matching_skills: list[str] = Field(
        description="Skills and technologies present in both the resume and job description"
    )

    missing_skills: list[str] = Field(
        description="Skills required by the job description that are missing or insufficiently demonstrated in the resume"
    )

    matching_experience: list[str] = Field(
        description="Relevant experience from the resume that matches the job description"
    )

    experience_gaps: list[str] = Field(
        description="Experience requirements from the job description that are not sufficiently demonstrated"
    )

    recommendations: list[str] = Field(
        description="Specific recommendations to improve the candidate's match for this job"
    )

    overall_assessment: str = Field(
        description="Overall assessment of how well the resume matches the job description"
    )
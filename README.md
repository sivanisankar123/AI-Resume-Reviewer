# 📄 AI Resume Reviewer

An AI-powered resume analysis application built with Python, Streamlit, OpenAI, RAG, and an agentic workflow.

The application allows users to upload a resume, analyze it with AI, compare it against a job description, and interact with a conversational AI Resume Agent that can select appropriate tools to answer questions using the resume.

---

## 🚀 Features

### 1. Resume Upload

- Upload a resume in PDF format.
- Extract resume text automatically.
- Display the extracted resume content.

### 2. AI Resume Analysis

The application analyzes the resume and provides:

- ATS compatibility score
- Years of professional experience
- Strengths
- Weaknesses
- Missing skills
- Recommended skills
- Suitable roles
- Resume improvement suggestions
- Overall assessment
- Top recommendation

Structured outputs are validated using Pydantic models.

### 3. Job Description Matching

Paste a job description and compare it with the uploaded resume.

The application provides:

- Match score
- Matching skills
- Missing skills
- Matching experience
- Experience gaps
- Recommendations
- Overall assessment

### 4. Resume Knowledge Base

The application converts the resume into a searchable knowledge base.

Pipeline:

```text
PDF Resume
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
OpenAI Embeddings
    ↓
Vector Store

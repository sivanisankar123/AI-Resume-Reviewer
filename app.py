import streamlit as st
from utils.pdf_reader import extract_text

# Configure the page
st.set_page_config(
    page_title="AI Resume Reviewer",
    page_icon="📄",
    layout="wide"
)

# Title
st.title("📄 AI Resume Reviewer")
st.write("Upload your resume in PDF format and extract its text.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a PDF Resume",
    type=["pdf"]
)

# If user uploads a file
if uploaded_file is not None:

    # Save uploaded file temporarily
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text from PDF
    text = extract_text("temp_resume.pdf")

    st.success("✅ Resume uploaded successfully!")

    st.subheader("Extracted Resume Text")

    st.text_area(
        "Resume Content",
        text,
        height=400
    )
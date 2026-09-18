import streamlit as st

from utils.pdf_reader import extract_text
from utils.ai import analyze_resume
from utils.jd_matcher import match_resume_to_job
from utils.resume_store import build_resume_store
from agent import run_agent


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Reviewer",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("📄 AI Resume Reviewer")

st.write(
    "Upload your resume, analyze it with AI, compare it "
    "against a job description, and ask questions using "
    "an agent-powered resume assistant."
)


# ============================================================
# RESUME UPLOAD
# ============================================================

st.header("1️⃣ Upload Resume")

uploaded_file = st.file_uploader(
    "Choose your resume",
    type=["pdf"]
)


if uploaded_file is not None:

    # --------------------------------------------------------
    # SAVE UPLOADED PDF
    # --------------------------------------------------------

    with open("temp_resume.pdf", "wb") as file:
        file.write(uploaded_file.getbuffer())

    # --------------------------------------------------------
    # EXTRACT RESUME TEXT
    # --------------------------------------------------------

    resume_text = extract_text(
        "temp_resume.pdf"
    )

    st.success(
        "✅ Resume uploaded successfully!"
    )

    # --------------------------------------------------------
    # VIEW EXTRACTED RESUME
    # --------------------------------------------------------

    with st.expander(
        "📄 View Extracted Resume Text"
    ):

        st.text_area(
            "Resume",
            resume_text,
            height=400,
            key="resume_text_display"
        )


    # ========================================================
    # RESUME ANALYSIS
    # ========================================================

    st.header("2️⃣ Resume Analysis")

    if st.button(
        "🤖 Analyze Resume",
        key="analyze_resume_button"
    ):

        with st.spinner(
            "Analyzing resume with AI..."
        ):

            try:

                analysis = analyze_resume(
                    resume_text
                )

                st.session_state.analysis = analysis

            except Exception as e:

                st.error(
                    f"❌ Resume analysis failed: {e}"
                )


    # --------------------------------------------------------
    # DISPLAY RESUME ANALYSIS
    # --------------------------------------------------------

    if "analysis" in st.session_state:

        analysis = st.session_state.analysis

        st.subheader("📊 ATS Score")

        st.metric(
            "ATS Compatibility Score",
            f"{analysis.ats_score}/100"
        )

        st.subheader(
            "💼 Years of Experience"
        )

        st.write(
            analysis.years_of_experience
        )

        st.subheader(
            "💪 Strengths"
        )

        for item in analysis.strengths:

            st.write(
                f"• {item}"
            )

        st.subheader(
            "⚠️ Weaknesses"
        )

        for item in analysis.weaknesses:

            st.write(
                f"• {item}"
            )

        st.subheader(
            "❌ Missing Skills"
        )

        for item in analysis.missing_skills:

            st.write(
                f"• {item}"
            )

        st.subheader(
            "📚 Recommended Skills"
        )

        for item in analysis.recommended_skills:

            st.write(
                f"• {item}"
            )

        st.subheader(
            "🎯 Suitable Roles"
        )

        for item in analysis.suitable_roles:

            st.write(
                f"• {item}"
            )

        st.subheader(
            "💡 Improvement Suggestions"
        )

        for item in analysis.improvement_suggestions:

            st.write(
                f"• {item}"
            )

        st.subheader(
            "📝 Overall Assessment"
        )

        st.write(
            analysis.overall_assessment
        )

        st.subheader(
            "⭐ Top Recommendation"
        )

        st.write(
            analysis.top_recommendation
        )


    # ========================================================
    # JOB DESCRIPTION MATCHING
    # ========================================================

    st.header(
        "3️⃣ Job Description Matching"
    )

    job_description = st.text_area(
        "Paste the job description here",
        height=250,
        key="job_description"
    )


    if st.button(
        "🎯 Match Resume to Job",
        key="match_job_button"
    ):

        if not job_description.strip():

            st.warning(
                "Please paste a job description first."
            )

        else:

            with st.spinner(
                "Comparing resume with job description..."
            ):

                try:

                    match_result = match_resume_to_job(
                        resume_text,
                        job_description
                    )

                    st.session_state.match_result = (
                        match_result
                    )

                except Exception as e:

                    st.error(
                        f"❌ Job matching failed: {e}"
                    )


    # --------------------------------------------------------
    # DISPLAY JOB MATCH RESULT
    # --------------------------------------------------------

    if "match_result" in st.session_state:

        match_result = st.session_state.match_result

        st.subheader(
            "🎯 Job Match Result"
        )

        st.metric(
            "Match Score",
            f"{match_result.match_score}/100"
        )

        st.subheader(
            "✅ Matching Skills"
        )

        for item in match_result.matching_skills:

            st.write(
                f"• {item}"
            )

        st.subheader(
            "❌ Missing Skills"
        )

        for item in match_result.missing_skills:

            st.write(
                f"• {item}"
            )

        st.subheader(
            "💼 Matching Experience"
        )

        for item in match_result.matching_experience:

            st.write(
                f"• {item}"
            )

        st.subheader(
            "⚠️ Experience Gaps"
        )

        for item in match_result.experience_gaps:

            st.write(
                f"• {item}"
            )

        st.subheader(
            "💡 Recommendations"
        )

        for item in match_result.recommendations:

            st.write(
                f"• {item}"
            )

        st.subheader(
            "📝 Overall Assessment"
        )

        st.write(
            match_result.overall_assessment
        )


    # ========================================================
    # RESUME KNOWLEDGE BASE
    # ========================================================

    st.header(
        "4️⃣ Resume Knowledge Base"
    )

    st.write(
        "Build the vector store used by the "
        "Resume Agent for semantic search."
    )


    if st.button(
        "🔄 Build Resume Knowledge Base",
        key="build_resume_store_button"
    ):

        with st.spinner(
            "Creating resume embeddings..."
        ):

            try:

                total_records = build_resume_store(
                    "temp_resume.pdf",
                    "sivani_resume"
                )

                st.success(
                    "✅ Resume knowledge base created "
                    f"successfully with {total_records} chunks."
                )

            except Exception as e:

                st.error(
                    f"❌ Failed to build resume knowledge base: {e}"
                )


    # ========================================================
    # RESUME AGENT
    # ========================================================

    st.header(
        "5️⃣ Resume Agent"
    )

    st.write(
        "Ask questions about the uploaded resume. "
        "The AI agent decides when to use the resume "
        "search tool and remembers the conversation."
    )


    # --------------------------------------------------------
    # INITIALIZE AGENT CONVERSATION
    # --------------------------------------------------------

    if "agent_conversation" not in st.session_state:

        st.session_state.agent_conversation = []


    # --------------------------------------------------------
    # CLEAR CONVERSATION
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Agent Conversation",
        key="clear_agent_conversation"
    ):

        st.session_state.agent_conversation = []

        st.rerun()


    # --------------------------------------------------------
    # DISPLAY PREVIOUS CONVERSATION
    # --------------------------------------------------------

    for message in st.session_state.agent_conversation:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )


    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask the agent about the resume..."
    )


    if question:

        # ----------------------------------------------------
        # DISPLAY USER QUESTION
        # ----------------------------------------------------

        with st.chat_message(
            "user"
        ):

            st.write(
                question
            )


        # ----------------------------------------------------
        # RUN AGENT
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "🤖 Agent is thinking..."
            ):

                try:

                    answer = run_agent(
                        question,
                        st.session_state.agent_conversation
                    )

                except Exception as e:

                    answer = (
                        f"❌ Agent error: {e}"
                    )


            st.write(
                answer
            )


        # ----------------------------------------------------
        # SAVE CONVERSATION
        # ----------------------------------------------------

        st.session_state.agent_conversation.append(
            {
                "role": "user",
                "content": question
            }
        )

        st.session_state.agent_conversation.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
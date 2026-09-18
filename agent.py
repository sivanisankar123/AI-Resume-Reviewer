import json

from openai import OpenAI

from rag import answer_question
from utils.pdf_reader import extract_text
from utils.ai import analyze_resume
from utils.jd_matcher import match_resume_to_job


client = OpenAI()

RESUME_ID = "sivani_resume"
RESUME_PATH = "temp_resume.pdf"


# ============================================================
# TOOL 1 — RESUME SEARCH
# ============================================================

def resume_search_tool(
    question: str,
    conversation_history: list[dict] | None = None
) -> str:

    if conversation_history is None:
        conversation_history = []

    answer, results = answer_question(
        question,
        conversation_history=conversation_history,
        resume_id=RESUME_ID
    )

    if not results:
        return answer

    evidence = "\n\n".join(
        result["text"]
        for result in results
    )

    return (
        f"ANSWER:\n{answer}\n\n"
        f"RESUME EVIDENCE:\n{evidence}"
    )


# ============================================================
# TOOL 2 — RESUME ANALYSIS
# ============================================================

def resume_analysis_tool() -> str:

    resume_text = extract_text(
        RESUME_PATH
    )

    if not resume_text.strip():
        return (
            "Resume text could not be extracted."
        )

    analysis = analyze_resume(
        resume_text
    )

    return json.dumps(
        analysis.model_dump(),
        indent=2
    )


# ============================================================
# TOOL 3 — JOB MATCHING
# ============================================================

def job_match_tool(
    job_description: str
) -> str:

    resume_text = extract_text(
        RESUME_PATH
    )

    if not resume_text.strip():
        return (
            "Resume text could not be extracted."
        )

    if not job_description.strip():
        return (
            "Job description is empty."
        )

    result = match_resume_to_job(
        resume_text,
        job_description
    )

    return json.dumps(
        result.model_dump(),
        indent=2
    )


# ============================================================
# AGENT TOOLS
# ============================================================

TOOLS = [

    {
        "type": "function",
        "name": "resume_search",
        "description": (
            "Search the candidate's resume using semantic "
            "search. Use this for questions asking about "
            "specific resume facts, experience, skills, "
            "projects, technologies, responsibilities, "
            "companies, education, certifications, or "
            "career history."
        ),
        "parameters": {

            "type": "object",

            "properties": {

                "question": {
                    "type": "string",
                    "description": (
                        "A standalone question about the "
                        "candidate's resume."
                    )
                }

            },

            "required": [
                "question"
            ]
        }
    },

    {
        "type": "function",
        "name": "resume_analysis",
        "description": (
            "Analyze the uploaded resume and return "
            "structured resume analysis including ATS "
            "score, years of experience, strengths, "
            "weaknesses, missing skills, recommended "
            "skills, suitable roles, improvement "
            "suggestions, overall assessment, and "
            "top recommendation."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },

    {
        "type": "function",
        "name": "job_match",
        "description": (
            "Compare the uploaded resume against a "
            "job description. Return match score, "
            "matching skills, missing skills, matching "
            "experience, experience gaps, recommendations, "
            "and overall assessment."
        ),
        "parameters": {

            "type": "object",

            "properties": {

                "job_description": {
                    "type": "string",
                    "description": (
                        "The complete job description "
                        "to compare against the resume."
                    )
                }

            },

            "required": [
                "job_description"
            ]
        }
    }
]


# ============================================================
# AGENT
# ============================================================

def run_agent(
    question: str,
    conversation_history: list[dict] | None = None
) -> str:

    if conversation_history is None:
        conversation_history = []

    # --------------------------------------------------------
    # Conversation context
    # --------------------------------------------------------

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in conversation_history[-8:]
    )

    # --------------------------------------------------------
    # Agent instructions
    # --------------------------------------------------------

    agent_prompt = f"""
You are an AI Resume Review Agent.

Your job is to help the user understand and evaluate
the candidate's uploaded resume.

You have three tools:

1. resume_search
   Use this for factual questions about information
   contained in the resume.

2. resume_analysis
   Use this when the user asks for resume analysis,
   ATS score, strengths, weaknesses, missing skills,
   recommended skills, suitable roles, or improvement
   recommendations.

3. job_match
   Use this when the user asks to compare the resume
   against a job description.

IMPORTANT RULES:

- Use the appropriate tool instead of guessing.
- Do not invent information.
- Keep answers grounded in the resume.
- If information is not available, clearly say so.
- Follow-up questions may refer to previous messages.
- Resolve references such as "it", "that", "this",
  "the above", or "how was it used?" using the
  conversation history.
- When calling resume_search, rewrite follow-up
  questions into standalone questions.
- When the user asks about a job description, use
  job_match rather than resume_search.
- When the user asks for ATS/resume analysis,
  use resume_analysis.
- Present the final answer clearly and concisely.

CONVERSATION HISTORY:

{history_text}

CURRENT USER QUESTION:

{question}
"""

    # --------------------------------------------------------
    # First agent call
    # --------------------------------------------------------

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=agent_prompt,
        tools=TOOLS
    )

    tool_outputs = []

    # --------------------------------------------------------
    # Process tool calls
    # --------------------------------------------------------

    for item in response.output:

        if item.type != "function_call":
            continue

        print(
            f"\n🔧 Agent selected tool: {item.name}"
        )

        arguments = json.loads(
            item.arguments
        )

        # ----------------------------------------------------
        # Resume Search
        # ----------------------------------------------------

        if item.name == "resume_search":

            tool_result = resume_search_tool(
                arguments["question"],
                conversation_history
            )

        # ----------------------------------------------------
        # Resume Analysis
        # ----------------------------------------------------

        elif item.name == "resume_analysis":

            tool_result = resume_analysis_tool()

        # ----------------------------------------------------
        # Job Match
        # ----------------------------------------------------

        elif item.name == "job_match":

            tool_result = job_match_tool(
                arguments["job_description"]
            )

        else:

            tool_result = (
                f"Unknown tool: {item.name}"
            )

        tool_outputs.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": tool_result
            }
        )

    # --------------------------------------------------------
    # Agent answered without a tool
    # --------------------------------------------------------

    if not tool_outputs:

        return response.output_text.strip()

    # --------------------------------------------------------
    # Send tool results back to agent
    # --------------------------------------------------------

    final_response = client.responses.create(
        model="gpt-5.6-luna",
        previous_response_id=response.id,
        input=tool_outputs
    )

    return final_response.output_text.strip()


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    conversation_history = []

    print("\n" + "=" * 60)
    print("AI RESUME AGENT — MULTI TOOL")
    print("=" * 60)

    print(
        "\nAvailable tools:"
    )

    print(
        "• resume_search"
    )

    print(
        "• resume_analysis"
    )

    print(
        "• job_match"
    )

    while True:

        question = input(
            "\nAsk the agent: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit"
        }:

            print(
                "\nGoodbye!"
            )

            break

        if not question:
            continue

        try:

            answer = run_agent(
                question,
                conversation_history
            )

            print(
                "\n🤖 AGENT ANSWER"
            )

            print(
                "-" * 60
            )

            print(
                answer
            )

            # ------------------------------------------------
            # Save conversation
            # ------------------------------------------------

            conversation_history.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            conversation_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as e:

            print(
                f"\n❌ Agent error: {e}"
            )
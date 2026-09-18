import json

from openai import OpenAI

from rag import answer_question


client = OpenAI()


# ============================================================
# RESUME SEARCH TOOL
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
        resume_id="sivani_resume"
    )

    if results:

        evidence = "\n\n".join(
            result["text"]
            for result in results
        )

        return (
            f"ANSWER:\n{answer}\n\n"
            f"RESUME EVIDENCE:\n{evidence}"
        )

    return answer


# ============================================================
# TOOLS
# ============================================================

TOOLS = [
    {
        "type": "function",
        "name": "resume_search",
        "description": (
            "Search the candidate's resume using semantic search. "
            "Use this whenever the user asks about information "
            "contained in the resume."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "A standalone question about the candidate's resume."
                    )
                }
            },
            "required": ["question"]
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
    # Give the agent conversation context
    # --------------------------------------------------------

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in conversation_history[-6:]
    )

    agent_prompt = f"""
You are an AI Resume Agent.

Answer questions using the candidate's resume.

The user may ask follow-up questions such as:
- "How was it used?"
- "What about that?"
- "How many years?"
- "Which companies?"
- "Tell me more about it."

Use the conversation history to understand what
the user is referring to.

If the question requires information from the resume,
use the resume_search tool.

Always rewrite follow-up questions into a standalone
question when calling the tool.

Do not invent resume information.

CONVERSATION HISTORY:
{history_text}

CURRENT QUESTION:
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
    # Execute requested tools
    # --------------------------------------------------------

    for item in response.output:

        if item.type == "function_call":

            print(
                f"\n🔧 Agent selected tool: {item.name}"
            )

            arguments = json.loads(
                item.arguments
            )

            if item.name == "resume_search":

                tool_result = resume_search_tool(
                    arguments["question"],
                    conversation_history
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
    # No tool required
    # --------------------------------------------------------

    if not tool_outputs:

        return response.output_text.strip()

    # --------------------------------------------------------
    # Give tool result back to agent
    # --------------------------------------------------------

    final_response = client.responses.create(
        model="gpt-5.6-luna",
        previous_response_id=response.id,
        input=tool_outputs
    )

    return final_response.output_text.strip()


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    conversation_history = []

    print("\n" + "=" * 60)
    print("AI RESUME AGENT")
    print("=" * 60)

    while True:

        question = input(
            "\nAsk the agent: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit"
        }:
            print("\nGoodbye!")
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

            print("-" * 60)

            print(answer)

            # Save conversation
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
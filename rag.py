from openai import OpenAI

from utils.embeddings import create_embedding
from utils.vector_store import load_vectors
from utils.retriever import retrieve


# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI()


# ============================================================
# DEFAULT RESUME
# ============================================================

DEFAULT_RESUME_ID = "sivani_resume"


# ============================================================
# ANSWER RESUME QUESTION
# ============================================================

def answer_question(
    question: str,
    conversation_history: list[dict] | None = None,
    resume_id: str = DEFAULT_RESUME_ID
):

    # --------------------------------------------------------
    # Initialize conversation history
    # --------------------------------------------------------

    if conversation_history is None:
        conversation_history = []


    # --------------------------------------------------------
    # Load vectors for the selected resume
    # --------------------------------------------------------

    records = load_vectors(resume_id)


    if not records:

        return (
            "Resume vector store is empty. "
            "Please build the resume vector store first.",
            []
        )


    # --------------------------------------------------------
    # Rewrite follow-up questions
    # --------------------------------------------------------

    search_question = question


    if conversation_history:

        previous_context = "\n".join(

            f"{message['role']}: {message['content']}"

            for message in conversation_history[-6:]

        )


        rewrite_prompt = f"""
Rewrite the user's question into a standalone search question
using the conversation history.

Do not answer the question.

Return only the rewritten search question.

CONVERSATION HISTORY:
{previous_context}

CURRENT QUESTION:
{question}
"""


        rewrite_response = client.responses.create(

            model="gpt-5.6-luna",

            input=rewrite_prompt

        )


        search_question = (
            rewrite_response.output_text.strip()
        )


    # --------------------------------------------------------
    # Display search question
    # --------------------------------------------------------

    print("\nSEARCH QUESTION")

    print("-" * 60)

    print(search_question)

    print("-" * 60)


    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = create_embedding(
        search_question
    )


    # --------------------------------------------------------
    # Retrieve relevant resume chunks
    # --------------------------------------------------------

    results = retrieve(

        query_embedding,

        records,

        top_k=3

    )


    # --------------------------------------------------------
    # No relevant information
    # --------------------------------------------------------

    if not results:

        return (

            "No relevant information was found in the resume.",

            []

        )


    # --------------------------------------------------------
    # Build resume context
    # --------------------------------------------------------

    context = "\n\n".join(

        result["text"]

        for result in results

    )


    # --------------------------------------------------------
    # Build conversation context
    # --------------------------------------------------------

    conversation_context = "\n".join(

        f"{message['role']}: {message['content']}"

        for message in conversation_history[-6:]

    )


    # --------------------------------------------------------
    # Grounded RAG prompt
    # --------------------------------------------------------

    prompt = f"""
You are an AI resume reviewer.

Answer the user's question using ONLY the resume
information provided below.

Rules:

- Do not invent information.
- Do not make assumptions.
- Use only information supported by the resume.
- If the answer is not available in the resume,
  clearly say that it is not mentioned in the resume.
- Keep the answer concise and relevant.
- If the question is a follow-up question, use the
  conversation history to understand what the user means.
- Do not introduce skills, technologies, responsibilities,
  or experience that are not supported by the resume.

RESUME CONTEXT:
-------------------------
{context}
-------------------------

CONVERSATION HISTORY:
-------------------------
{conversation_context}
-------------------------

CURRENT USER QUESTION:
{question}

Answer:
"""


    # --------------------------------------------------------
    # Ask the LLM
    # --------------------------------------------------------

    response = client.responses.create(

        model="gpt-5.6-luna",

        input=prompt

    )


    answer = response.output_text.strip()


    # --------------------------------------------------------
    # Save conversation
    # --------------------------------------------------------

    conversation_history.append({

        "role": "user",

        "content": question

    })


    conversation_history.append({

        "role": "assistant",

        "content": answer

    })


    # --------------------------------------------------------
    # Return answer + retrieved evidence
    # --------------------------------------------------------

    return answer, results


# ============================================================
# COMMAND-LINE CHAT
# ============================================================

if __name__ == "__main__":

    conversation_history = []

    resume_id = DEFAULT_RESUME_ID


    print("\n" + "=" * 60)

    print("AI RESUME ASSISTANT")

    print("=" * 60)

    print("Ask questions about the resume.")

    print("Type 'exit' to quit.")

    print("=" * 60)


    while True:

        question = input(
            "\nAsk a question about the resume: "
        )


        if question.lower().strip() == "exit":

            print("\nGoodbye!")

            break


        answer, results = answer_question(

            question,

            conversation_history,

            resume_id

        )


        print("\n" + "=" * 60)

        print("AI ANSWER")

        print("=" * 60)

        print(answer)


        # ----------------------------------------------------
        # Display retrieved evidence
        # ----------------------------------------------------

        if results:

            print("\n" + "=" * 60)

            print("RETRIEVED RESUME EVIDENCE")

            print("=" * 60)


            for result in results:

                print(

                    f"\nChunk {result['chunk_id']}"

                    f" | Similarity: "

                    f"{result['score']:.4f}"

                )

                print("-" * 60)

                print(result["text"])
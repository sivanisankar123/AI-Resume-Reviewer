from utils.pdf_reader import extract_text
from utils.chunker import chunk_by_words
from utils.embeddings import create_embedding
from utils.vector_store import save_vectors


def build_resume_store(
    pdf_path: str,
    resume_id: str,
    chunk_size: int = 100
) -> int:

    # 1. Extract resume text
    text = extract_text(pdf_path)

    if not text or not text.strip():
        raise ValueError("Could not extract text from the resume.")

    # 2. Split resume into chunks
    chunks = chunk_by_words(
        text,
        chunk_size=chunk_size
    )

    if not chunks:
        raise ValueError("No chunks were created from the resume.")

    # 3. Create embeddings
    records = []

    for index, chunk in enumerate(
        chunks,
        start=1
    ):
        print(
            f"Creating embedding for chunk "
            f"{index}/{len(chunks)}..."
        )

        embedding = create_embedding(chunk)

        records.append({
            "chunk_id": index,
            "text": chunk,
            "embedding": embedding
        })

    # 4. Save vector store
    save_vectors(
        records,
        resume_id
    )

    print(
        "\nResume vector store created successfully!"
    )

    print(
        f"Resume ID: {resume_id}"
    )

    print(
        f"Total records: {len(records)}"
    )

    return len(records)

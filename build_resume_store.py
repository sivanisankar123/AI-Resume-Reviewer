from utils.pdf_reader import extract_text
from utils.chunker import chunk_by_words
from utils.embeddings import create_embedding
from utils.vector_store import save_vectors


PDF_PATH = "uploads/Sivani Sankar Mohapatra-Automation .pdf"
RESUME_ID = "sivani_resume"


# 1. Extract resume text
text = extract_text(PDF_PATH)


# 2. Split into chunks
chunks = chunk_by_words(
    text,
    chunk_size=100
)


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


# 4. Save vectors for this resume
save_vectors(
    records,
    RESUME_ID
)


print(
    "\nResume vector store created successfully!"
)

print(
    f"Resume ID: {RESUME_ID}"
)

print(
    f"Total records: {len(records)}"
)

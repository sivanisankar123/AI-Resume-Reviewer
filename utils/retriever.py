import math


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )


def retrieve(
    query_embedding,
    records,
    top_k=5,
    similarity_threshold=0.30
):
    """
    Retrieve the most relevant resume chunks.

    We use both:
    1. A lower similarity threshold so valid factual
       resume information is not discarded.
    2. top_k=5 to provide enough context to the LLM.
    """

    results = []

    for record in records:

        score = cosine_similarity(
            query_embedding,
            record["embedding"]
        )

        if score >= similarity_threshold:

            results.append({
                "chunk_id": record["chunk_id"],
                "text": record["text"],
                "score": score
            })

    # Highest similarity first
    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # Debug output
    print("\nRETRIEVAL RESULTS")
    print("-" * 60)

    for result in results[:top_k]:

        print(
            f"Chunk {result['chunk_id']} "
            f"| Similarity: {result['score']:.4f}"
        )

    print("-" * 60)

    return results[:top_k]
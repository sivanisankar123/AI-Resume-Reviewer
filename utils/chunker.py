def chunk_by_words(text, chunk_size=100):
    """
    Split text into chunks based on word count.
    """

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks
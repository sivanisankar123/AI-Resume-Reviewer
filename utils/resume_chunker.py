import re


SECTION_NAMES = [
    "PROFESSIONAL SUMMARY",
    "CORE COMPETENCIES",
    "CAREER HIGHLIGHTS",
    "PROFESSIONAL EXPERIENCE",
    "FEATURED PROJECTS",
    "TECHNICAL SKILLS",
    "CERTIFICATIONS",
    "LANGUAGES",
]


def split_into_sections(text):
    """
    Split resume text into logical sections.
    """

    pattern = "|".join(
        re.escape(section)
        for section in SECTION_NAMES
    )

    matches = list(
        re.finditer(
            rf"\b({pattern})\b",
            text,
            re.IGNORECASE
        )
    )

    sections = []

    for index, match in enumerate(matches):

        section_name = match.group(1).upper()

        start = match.end()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        section_text = text[start:end].strip()

        sections.append({
            "section": section_name,
            "text": section_text
        })

    return sections


def chunk_text(text, chunk_size=100):
    """
    Split text into word-based chunks.
    """

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        chunks.append(chunk)

    return chunks


def chunk_resume(text, chunk_size=100):
    """
    Split resume into sections and then chunks.
    """

    sections = split_into_sections(text)

    chunks = []

    chunk_id = 1

    for section in sections:

        section_chunks = chunk_text(
            section["text"],
            chunk_size
        )

        for chunk in section_chunks:

            chunks.append({
                "chunk_id": chunk_id,
                "section": section["section"],
                "text": chunk
            })

            chunk_id += 1

    return chunks
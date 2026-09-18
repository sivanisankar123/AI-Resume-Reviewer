import json
from pathlib import Path


VECTOR_STORE_DIR = Path("vector_stores")


def save_vectors(
    records: list[dict],
    resume_id: str
) -> None:

    VECTOR_STORE_DIR.mkdir(
        exist_ok=True
    )

    vector_store_file = (
        VECTOR_STORE_DIR / f"{resume_id}.json"
    )

    with open(
        vector_store_file,
        "w"
    ) as file:

        json.dump(
            records,
            file
        )


def load_vectors(
    resume_id: str
) -> list[dict]:

    vector_store_file = (
        VECTOR_STORE_DIR / f"{resume_id}.json"
    )

    if not vector_store_file.exists():

        return []

    with open(
        vector_store_file,
        "r"
    ) as file:

        return json.load(file)

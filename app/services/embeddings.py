from functools import lru_cache


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=512)
def create_embedding(text: str) -> list[float]:
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def create_embeddings(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return [embedding.tolist() for embedding in embeddings]


def warm_embedding_model() -> None:
    create_embedding("warm up memvora embeddings")

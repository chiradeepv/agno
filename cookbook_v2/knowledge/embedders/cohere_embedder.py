import asyncio

from agno.knowledge.embedder.cohere import CohereEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector

embeddings = CohereEmbedder().get_embedding(
    "The quick brown fox jumps over the lazy dog."
)
# Print the embeddings and their dimensions
print(f"Embeddings: {embeddings[:5]}")
print(f"Dimensions: {len(embeddings)}")

# Example usage:
knowledge = Knowledge(
    vector_db=PgVector(
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        table_name="cohere_embeddings",
        embedder=CohereEmbedder(
            dimensions=1024,
        ),
    ),
    max_results=2,
)

asyncio.run(
    knowledge.add_content(
        path="cookbook_v2/knowledge/data/filters/cv_1.pdf",
    )
)

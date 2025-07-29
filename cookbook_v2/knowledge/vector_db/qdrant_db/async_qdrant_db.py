import asyncio

from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant

COLLECTION_NAME = "thai-recipes"

vector_db = Qdrant(collection=COLLECTION_NAME, url="http://localhost:6333")

# Create knowledge base
knowledge = Knowledge(
    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=vector_db,
)

agent = Agent(knowledge=knowledge, show_tool_calls=True)

if __name__ == "__main__":
    # Comment out after first run
    asyncio.run(knowledge.async_add_content(url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"))

    # Create and use the agent
    asyncio.run(agent.aprint_response("How to make Tom Kha Gai", markdown=True))

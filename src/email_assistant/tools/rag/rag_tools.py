from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from pydantic import BaseModel, Field
import os

# Use absolute path to vector store
_vector_db_path = os.path.join(os.path.dirname(__file__), "dno_guidance_db")
_qdrant_client = QdrantClient(path=_vector_db_path)
_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
_vector_store = QdrantVectorStore(
    client=_qdrant_client,
    collection_name="dno_guidance",
    embedding=_embeddings,
)

class SearchDNOGuidanceInput(BaseModel):
    query: str = Field(description="Query string to search DNO guidance")
    max_results: int = Field(default=2, description="Maximum number of results to return")


@tool(args_schema=SearchDNOGuidanceInput)
def search_dno_guidance_tool(query: str, max_results: int = 2) -> str:
    """Search DNO operational knowledge base and return concatenated sources and content."""
    results = _vector_store.similarity_search(query, k=max_results)
    print(results[0])
    return "\n\n".join(
        [
            f"**{doc.metadata.get('source', 'Unknown')}**\n{doc.page_content}"
            for doc in results
        ]
    )
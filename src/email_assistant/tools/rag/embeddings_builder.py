from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os

# Configuration
EMBEDDING_MODEL = "text-embedding-3-large"
COLLECTION_NAME = "dno_guidance"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(SCRIPT_DIR, "dno_guidance_db")
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "..", "data")  # Relative to script location

def build_vector_store():
    """Build and persist the Qdrant vector store from PDF documents."""

    # Load documents
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    all_docs = []

    for pdf_file in pdf_files:
        file_path = os.path.join(DATA_DIR, pdf_file)
        loader = PDFPlumberLoader(file_path)
        all_docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True
    )
    splits = text_splitter.split_documents(all_docs)

    # Initialise embeddings
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Create persistent Qdrant client
    client = QdrantClient(path=VECTOR_DB_PATH)

    # Create collection if it doesn't exist
    vector_size = len(embeddings.embed_query("sample text"))

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

    # Create vector store and add documents
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    ids = vector_store.add_documents(documents=splits)
    print(f"Added {len(ids)} document chunks to vector store")

    return vector_store

if __name__ == "__main__":
    build_vector_store()
    print(f"Vector store persisted to {VECTOR_DB_PATH}")

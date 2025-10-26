from langchain_core.tools import tool
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

file_path = "/Users/aidan.kelly/nesta/discovery/agentic_prototype/agents-from-scratch/src/email_assistant/data/ena_connect_direct_guidance.pdf"
loader = PDFPlumberLoader(file_path)

docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)



embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


client = QdrantClient(":memory:")

vector_size = len(embeddings.embed_query("sample text"))

if not client.collection_exists("test"):
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
vector_store = QdrantVectorStore(
    client=client,
    collection_name="test",
    embedding=embeddings,
)

ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search(
    "What is Connect Direct?"
)



retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)



print(retriever.batch(
    [
        "What is Connect Direct?",
        "Which DNOs have access to Connect Direct?",
    ],
))
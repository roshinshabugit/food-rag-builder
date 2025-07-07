import requests
import os
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Environment variables
QDRANT_URL = os.getenv("QDRANT_HOST", "https://qdrant-5qn9.onrender.com")
COLLECTION_NAME = "food_safety"

# Set up remote Qdrant client
qdrant = QdrantClient(
    url=QDRANT_URL,
    prefer_grpc=False  # Required for Render (HTTP only)
)

# Ensure the collection exists (optional but recommended)
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Scrape the source
url = "https://inspection.canada.ca/en/food-safety-industry/food-safety-standards-guidelines"

def extract_text(url):
    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return ""

# Extract and chunk
full_text = extract_text(url)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.create_documents([full_text])

# Embed & store
embedding_model = OpenAIEmbeddings()
Qdrant.from_documents(
    documents=chunks,
    embedding=embedding_model,
    client=qdrant,
    collection_name=COLLECTION_NAME
)

print(f"✅ Embedded and stored {len(chunks)} chunks in Qdrant")

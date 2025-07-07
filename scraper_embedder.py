import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
import qdrant_client
import os

QDRANT_URL = "https://qdrant-5qn9.onrender.com"
COLLECTION_NAME = "food_safety"

# Connect to Qdrant
qdrant = qdrant_client.QdrantClient(
    url=QDRANT_URL,
    prefer_grpc=False,
    timeout=30.0
)

# Scrape
url = "https://inspection.canada.ca/en/food-safety-industry/food-safety-standards-guidelines"

def extract_text(url):
    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return ""

full_text = extract_text(url)

# Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.create_documents([full_text])

# Embed
embedding_model = OpenAIEmbeddings()

# Store in Qdrant
Qdrant.from_documents(
    documents=chunks,
    embedding=embedding_model,
    client=qdrant,
    collection_name=COLLECTION_NAME
)

print(f"✅ Embedded and stored {len(chunks)} chunks in Qdrant")

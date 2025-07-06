import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
import qdrant_client
import os

# ENV variables
QDRANT_URL = os.getenv("QDRANT_HOST", "http://localhost:6333")
COLLECTION_NAME = "food_safety"

# Set up Qdrant client
qdrant = qdrant_client.QdrantClient(url=QDRANT_URL)

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
Qdrant.from_documents(documents=chunks, embedding=embedding_model, client=qdrant, collection_name=COLLECTION_NAME)

print(f"✅ Embedded and stored {len(chunks)} chunks in Qdrant")

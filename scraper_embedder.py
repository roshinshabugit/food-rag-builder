import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
import os

# 🔧 Config
QDRANT_URL = os.getenv("QDRANT_HOST", "https://qdrant-5qn9.onrender.com")
COLLECTION_NAME = "food_safety"

# 🌐 URL to scrape
url = "https://inspection.canada.ca/en/food-safety-industry/food-safety-standards-guidelines"

# 📄 Extract text content
def extract_text(url):
    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return ""

# 🧹 Scrape and split into chunks
full_text = extract_text(url)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.create_documents([full_text])

# 🔑 Embedding model
embedding_model = OpenAIEmbeddings()

# ✅ Embed and store in Qdrant (pass just the URL!)
qdrant = Qdrant.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
)

print(f"✅ Embedded and stored {len(chunks)} chunks in Qdrant")

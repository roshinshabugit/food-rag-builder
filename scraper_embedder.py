import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import time

# ----------------------------------
# CONFIG
# ----------------------------------
QDRANT_URL = "https://bd5b2f9b-a9eb-4b6f-b001-5c6a97acef10.us-east-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9LqZiWGtid-bHHlNf2_LoeRef8mjPKSayymPrE7QoB4")  # replace or use .env
COLLECTION_NAME = "food_safety"
TARGET_URL = "https://inspection.canada.ca/en/food-safety-industry/food-safety-standards-guidelines"

# ----------------------------------
# Functions
# ----------------------------------

def extract_text(url):
    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return ""

def connect_qdrant_with_retry(url, api_key, retries=3, delay=5):
    for i in range(retries):
        try:
            client = QdrantClient(url=url, api_key=api_key)
            client.get_collections()
            print("✅ Connected to Qdrant Cloud")
            return client
        except Exception as e:
            print(f"⏳ Qdrant connection failed (attempt {i+1}): {e}")
            time.sleep(delay)
    raise RuntimeError("❌ Qdrant is unreachable after multiple attempts.")

# ----------------------------------
# MAIN
# ----------------------------------

def main():
    # Step 1: Scrape the text
    print(f"🌐 Scraping content from: {TARGET_URL}")
    raw_text = extract_text(TARGET_URL)

    # Step 2: Chunk the text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([raw_text])
    print(f"✂️ Split into {len(docs)} chunks")

    # Step 3: Embed and store in Qdrant
    qdrant = connect_qdrant_with_retry(QDRANT_URL, QDRANT_API_KEY)
    embeddings = OpenAIEmbeddings()

    Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        client=qdrant,
        collection_name=COLLECTION_NAME
    )

    print(f"✅ Stored {len(docs)} chunks into Qdrant collection '{COLLECTION_NAME}'")

# ----------------------------------
# RUN
# ----------------------------------
if __name__ == "__main__":
    main()

import requests
import time
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
import qdrant_client
import os

# === CONFIGURATION ===
QDRANT_URL = "https://qdrant-5qn9.onrender.com"  # no port needed, Render routes it
COLLECTION_NAME = "food_safety"
MAX_RETRIES = 5
RETRY_INTERVAL = 10  # seconds

# === RETRYABLE QDRANT CLIENT CONNECTION ===
def connect_qdrant_with_retry(url):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"🔄 Attempt {attempt}: Connecting to Qdrant at {url}")
            client = qdrant_client.QdrantClient(url=url)
            client.get_collections()  # test connection
            print("✅ Connected to Qdrant!")
            return client
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_INTERVAL)
            else:
                raise RuntimeError("❌ Qdrant is unreachable after multiple attempts.")

# === TEXT SCRAPING FUNCTION ===
def extract_text(url):
    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return ""

# === MAIN PROCESS ===
def main():
    url = "https://inspection.canada.ca/en/food-safety-industry/food-safety-standards-guidelines"
    full_text = extract_text(url)

    if not full_text.strip():
        print("⚠️ No text extracted. Exiting.")
        return

    # Chunk text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.create_documents([full_text])

    # Embed & store in Qdrant
    embedding_model = OpenAIEmbeddings()
    qdrant = connect_qdrant_with_retry(QDRANT_URL)

    Qdrant.from_documents(
        documents=chunks,
        embedding=embedding_model,
        client=qdrant,
        collection_name=COLLECTION_NAME
    )

    print(f"✅ Successfully embedded and stored {len(chunks)} chunks to Qdrant!")

# === ENTRY POINT ===
if __name__ == "__main__":
    main()

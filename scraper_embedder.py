import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

# ----------------------------------
# CONFIG
# ----------------------------------
QDRANT_URL = "https://bd5b2f9b-a9eb-4b6f-b001-5c6a97acef10.us-east-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9LqZiWGtid-bHHlNf2_LoeRef8mjPKSayymPrE7QoB4"
COLLECTION_NAME = "food_safety"
TARGET_URL = "https://inspection.canada.ca/en/food-safety-industry/food-safety-standards-guidelines"

# ----------------------------------
# FUNCTIONS
# ----------------------------------

def extract_text(url):
    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return ""

# ----------------------------------
# MAIN
# ----------------------------------

def main():
    print(f"🌐 Scraping content from: {TARGET_URL}")
    raw_text = extract_text(TARGET_URL)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([raw_text])
    print(f"✂️ Split into {len(docs)} chunks")

    embeddings = OpenAIEmbeddings()

    Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME
    )

    print(f"✅ Stored {len(docs)} chunks into Qdrant collection '{COLLECTION_NAME}'")

# ----------------------------------
# RUN
# ----------------------------------

if __name__ == "__main__":
    main()

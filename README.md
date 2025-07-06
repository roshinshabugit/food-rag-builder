# Food RAG Builder (Qdrant Version)

This cron job scrapes the CFIA website once per month and stores the text in a Qdrant vector database hosted on Render.

## Setup

1. Deploy this repo as a Cron Job on [Render.com](https://render.com)
2. Set environment variables:
   - `OPENAI_API_KEY` (your OpenAI key)
   - `QDRANT_HOST` (your Qdrant Render URL, e.g., https://qdrant-xxxx.onrender.com)

## Notes

- Vector store: Qdrant
- Embeddings: OpenAI
- Target page: https://inspection.canada.ca/en/food-safety-industry/food-safety-standards-guidelines

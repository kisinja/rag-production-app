from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

response = client.embeddings.create(
    model="nomic-embed-text:latest",
    input="Hello, this is a RAG test.",
)

embedding = response.data[0].embedding

print("Embedding dimensions:", len(embedding))
print("First 5 values:", embedding[:5])

# uv run uvicorn main:app
# inngest-cli dev -u http://127.0.0.1:8000/api/inngest --no-discovery
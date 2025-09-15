import os
import json
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# =========================
# 1. ENV SETUP
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "atn"
DIMENSION = 3072  # text-embedding-3-large output size

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# =========================
# 2. CREATE INDEX IF NEEDED
# =========================
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait for index to be ready
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)

index = pc.Index(INDEX_NAME)

# =========================
# 3. HELPER: Build text for embeddings
# =========================
def build_text(record: dict) -> str:
    """Convert JSON fields into one concatenated string for embeddings."""
    fields = [
        record.get("Customer Name", ""),
        record.get("Email", ""),
        record.get("Phone Number", ""),
        record.get("country", ""),
        record.get("status", ""),
        record.get("Ad Lead Qualification Reason", ""),
        record.get("Decision and liquidity info", ""),
        record.get("Pain points Objections Outcomes", ""),
        record.get("Next Steps", ""),
        record.get("Supporting Justification or Description", ""),
    ]
    return " | ".join([f for f in fields if f])

# =========================
# 4. LOAD DATA
# =========================
with open("data (6).json", "r", encoding="utf-8") as f:
    data = json.load(f)

# =========================
# 5. UPSERT TO PINECONE
# =========================
batch_size = 50
to_upsert = []

for i, record in enumerate(data):
    record_id = record.get("Customer ID", f"cust-{i}")
    text_to_embed = build_text(record)

    # Generate embedding
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text_to_embed
    )
    embedding = response.data[0].embedding

    # ✅ Structured metadata
    metadata = {}
    for k, v in record.items():
        if isinstance(v, (str, int, float, bool)):
            metadata[k] = v

    # ✅ Add free-text field (so Pinecone stores it too)
    metadata["text"] = text_to_embed

    to_upsert.append({
        "id": record_id,
        "values": embedding,
        "metadata": metadata
    })

    # Batch upload
    if len(to_upsert) >= batch_size:
        index.upsert(vectors=to_upsert)
        print(f"Upserted {len(to_upsert)} records")
        to_upsert = []

# Final flush
if to_upsert:
    index.upsert(vectors=to_upsert)
    print(f"Upserted final {len(to_upsert)} records")

print("✅ Data successfully stored in Pinecone with structured metadata + free text!")

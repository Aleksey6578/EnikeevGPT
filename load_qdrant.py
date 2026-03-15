import json
import time
import requests
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# =====================
# CONFIG
# =====================
COLLECTION = "lectures_rag"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
QDRANT_URL = "http://localhost:6333"

BATCH_EMBED = 8
UPSERT_BATCH = 128
RETRIES = 5
RETRY_SLEEP = 1.5

# =====================
# LOAD CHUNKS
# =====================
chunks = []
with open("chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))
print(f"📦 Чанков: {len(chunks)}")

# =====================
# EMBEDDING
# =====================
def normalize_for_embedding(text: str) -> str:
    return " ".join(text.lower().split())

def embed_text(text: str) -> List[float]:
    text = normalize_for_embedding(text)
    for attempt in range(RETRIES):
        try:
            r = requests.post(
                OLLAMA_URL,
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=120
            )
            r.raise_for_status()
            data = r.json()
            return data.get("embedding") or data["data"][0]["embedding"]
        except Exception:
            time.sleep(RETRY_SLEEP)
    return []

# =====================
# PARALLEL EMBEDDING
# =====================
results: List[Tuple[List[float], dict]] = []

with ThreadPoolExecutor(max_workers=BATCH_EMBED) as executor:
    futures = {executor.submit(embed_text, ch["text"]): ch for ch in chunks}
    for future in as_completed(futures):
        ch = futures[future]
        emb = future.result()
        if emb:
            results.append((emb, ch))
print(f"✅ Получено embeddings: {len(results)}")

# =====================
# QDRANT
# =====================
client = QdrantClient(QDRANT_URL, check_compatibility=False)
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)

if results:
    emb_size = len(results[0][0])
else:
    emb_size = 1536  # fallback

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=emb_size, distance=Distance.COSINE)
)

# =====================
# UPSERT
# =====================
points = []
for i, (emb, ch) in enumerate(results, 1):
    points.append(PointStruct(id=ch["id"], vector=emb, payload=ch))
    if len(points) == UPSERT_BATCH or i == len(results):
        client.upsert(collection_name=COLLECTION, points=points)
        points = []

print("✅ Qdrant полностью заполнен")

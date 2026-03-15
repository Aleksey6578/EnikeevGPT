import requests
import re
import math
from collections import Counter
from qdrant_client import QdrantClient
import pymorphy2

# =====================
# CONFIG
# =====================
OLLAMA_LLM = "mistral:latest"
EMBED_MODEL = "nomic-embed-text"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GEN_URL = "http://localhost:11434/api/generate"

COLLECTION = "lectures_rag"
QDRANT_URL = "http://localhost:6333"

QUESTION = "Что будет, если номера на слайдах презентации мелкие ?"

RETRIEVE_K = 80
FINAL_K = 3
MAX_CONTEXT_CHARS = 7000

VECTOR_WEIGHT = 0.3
LEXICAL_WEIGHT = 0.2
BM25_WEIGHT = 0.5

# =====================
# MORPHOLOGY
# =====================
morph = pymorphy2.MorphAnalyzer()

def tokenize(text):
    tokens = re.findall(r'[а-яa-z0-9\-]+', text.lower())
    return [morph.parse(t)[0].normal_form for t in tokens]

# =====================
# EMBEDDING
# =====================
def embed_query(text):
    r = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60
    )
    data = r.json()
    return data.get("embedding") or data["data"][0]["embedding"]

# =====================
# BM25 (локальный по retrieved)
# =====================
def bm25(q_tokens, docs_tokens):
    k1, b = 1.5, 0.75
    N = len(docs_tokens)
    avgdl = sum(len(d) for d in docs_tokens) / N
    df = Counter(t for d in docs_tokens for t in set(d))
    scores = []
    for doc in docs_tokens:
        tf = Counter(doc)
        s = 0.0
        for t in q_tokens:
            if t in tf:
                idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(doc) / avgdl))
        scores.append(s)
    m = max(scores) or 1.0
    return [x / m for x in scores]

# =====================
# MAIN
# =====================
client = QdrantClient(QDRANT_URL, check_compatibility=False)
query_emb = embed_query(QUESTION)

hits = client.query_points(
    COLLECTION,
    query=query_emb,
    limit=RETRIEVE_K,
    with_payload=True
).points

q_tokens = tokenize(QUESTION)
docs_tokens = [tokenize(h.payload["text"]) for h in hits]
bm25_scores = bm25(q_tokens, docs_tokens)

scored = []
for h, b, dt in zip(hits, bm25_scores, docs_tokens):
    overlap = len(set(q_tokens) & set(dt)) / max(len(q_tokens), 1)
    score = VECTOR_WEIGHT * h.score + LEXICAL_WEIGHT * overlap + BM25_WEIGHT * b
    scored.append((score, h))

scored.sort(key=lambda x: x[0], reverse=True)
top = [h for _, h in scored[:FINAL_K]]

context = ""
for h in top:
    block = f"\n=== {h.payload['title']} ===\n{h.payload['text']}\n"
    if len(context) + len(block) > MAX_CONTEXT_CHARS:
        break
    context += block

PROMPT = f"""
Ответь на основе приведённого контекста.

КОНТЕКСТ:
{context}

ВОПРОС:
{QUESTION}

ОТВЕТ:
"""

r = requests.post(
    OLLAMA_GEN_URL,
    json={"model": OLLAMA_LLM, "prompt": PROMPT, "stream": False, "options": {"temperature": 0.0}},
    timeout=300
)
print(r.json().get("response", "Ошибка генерации ответа"))

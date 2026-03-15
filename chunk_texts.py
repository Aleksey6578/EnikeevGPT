import json
import re
from typing import List

# =====================
# CONFIG
# =====================
MAX_CHARS = 300         # размер чанка
OVERLAP_CHARS = 50      # overlap
HARD_LIMIT = 350         # защита от гигантских блоков

# =====================
# METADATA
# =====================
def detect_section(text: str) -> str:
    t = text.lower()
    if "сесс" in t or "зачет" in t or "экзам" in t:
        return "расписание"
    if "вкр" in t:
        return "вкр"
    return "other"

def detect_year(text: str):
    m = re.search(r'20\d{2}', text)
    return int(m.group()) if m else None

# =====================
# NORMALIZATION
# =====================
def normalize_text(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# =====================
# SPLIT INTO SEMANTIC UNITS
# =====================
def split_units(text: str) -> List[str]:
    blocks = re.split(r'\n{2,}', text)
    units = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # короткий блок без точек — единица
        if len(block) < 150 and not re.search(r'[.!?…]', block):
            units.append(block)
            continue
        # разбивка на предложения
        sentences = re.split(r'(?<=[.!?…])\s+', block)
        units.extend([s.strip() for s in sentences if s.strip()])
    return units

# =====================
# BUILD OVERLAP
# =====================
def build_overlap(buffer: List[str], overlap_chars: int) -> List[str]:
    overlap = []
    total = 0
    for u in reversed(buffer):
        if total + len(u) > overlap_chars:
            break
        overlap.insert(0, u)
        total += len(u)
    return overlap

# =====================
# MAIN CHUNKING
# =====================
def chunk_text(text: str) -> List[str]:
    units = split_units(text)
    chunks = []

    buffer = []
    buffer_len = 0

    for unit in units:
        unit_len = len(unit)

        # Сверхбольшой блок
        if unit_len > HARD_LIMIT:
            if buffer:
                chunks.append(" ".join(buffer))
                buffer = []
                buffer_len = 0
            for i in range(0, unit_len, MAX_CHARS):
                chunks.append(unit[i:i + MAX_CHARS])
            continue

        buffer.append(unit)
        buffer_len += unit_len

        # Если встретили "Ответ:", завершаем chunk
        if re.search(r"Ответ[:\n]", unit):
            chunks.append(" ".join(buffer))
            buffer = []
            buffer_len = 0
            continue

        # Если превышен MAX_CHARS без "Ответ:", делаем обычный разрез
        if buffer_len >= MAX_CHARS:
            chunks.append(" ".join(buffer))
            overlap = build_overlap(buffer, OVERLAP_CHARS)
            buffer = overlap
            buffer_len = sum(len(x) for x in buffer)

    # Финальный остаток
    if buffer:
        chunks.append(" ".join(buffer))

    return chunks

# =====================
# FILE PROCESSING
# =====================
def process_jsonl(input_path, output_path):
    out = []
    global_id = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for lecture_id, line in enumerate(f, start=1):
            record = json.loads(line)
            text = normalize_text(record.get("text", ""))
            source = record.get("source", "unknown")
            chunks = chunk_text(text)

            for i, ch in enumerate(chunks):
                global_id += 1
                out.append({
                    "id": global_id,
                    "lecture_id": lecture_id,
                    "chunk_index": i,
                    "source": source,
                    "title": record.get("title"),
                    "text": ch,
                    "section": detect_section(ch),
                    "year": detect_year(ch)
                })

    with open(output_path, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Готово. Всего чанков: {len(out)}")

if __name__ == "__main__":
    process_jsonl("data_clean.jsonl", "chunks.jsonl")

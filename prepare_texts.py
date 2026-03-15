import os
import re
import json
import unicodedata

DATA_DIR = "rag_data"
OUTPUT_FILE = "data_clean.jsonl"

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    text = re.sub(
        r'[^0-9A-Za-zА-Яа-яёЁ\s.,!?():;"\'«»\-–—\n№]',
        '',
        text
    )

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines).strip()


def process_file(path, out_file):
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("{") and "\n{" in content:
        for line in content.splitlines():
            process_record(json.loads(line), out_file)
    else:
        data = json.loads(content)
        if isinstance(data, list):
            for r in data:
                process_record(r, out_file)
        else:
            process_record(data, out_file)


def process_record(record, out_file):
    if "text" not in record:
        return

    cleaned = clean_text(record["text"])
    if len(cleaned) < 200:
        return

    out_file.write(json.dumps({
        "lecture": record.get("lecture"),
        "title": record.get("title", ""),
        "text": cleaned
    }, ensure_ascii=False) + "\n")


def main():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for fn in sorted(os.listdir(DATA_DIR)):
            if fn.endswith((".json", ".jsonl")):
                process_file(os.path.join(DATA_DIR, fn), out)

    print(f"✅ Записано в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

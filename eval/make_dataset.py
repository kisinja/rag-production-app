import asyncio
import json
import random
import sys
from pathlib import Path

from llama_index.readers.file import PDFReader

from rag_production_app.config import settings
from rag_production_app.data_loader import client

QUESTIONS_PER_PAGE = 3

PROMPT = """Read the passage and write ONE question a user could ask that is \
answerable only from this passage. Do not mention "the passage" or "the text". \
Return only the question.

Passage:
{passage}"""


async def make_question(passage: str) -> str:
    resp = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": PROMPT.format(passage=passage)}],
        temperature=0.5,
        max_tokens=120,
        extra_body={
            "reasoning_effort": "none"
        },  # delete if you use a non-reasoning model
    )
    return (resp.choices[0].message.content or "").strip()


async def main(pdfs: list[str], out: str = "eval/dataset.jsonl") -> None:
    rows = []
    for pdf in pdfs:
        source = Path(pdf).name
        for doc in PDFReader().load_data(file=pdf):
            text = (doc.text or "").strip()
            if len(text) < 300:
                continue
            for _ in range(QUESTIONS_PER_PAGE):
                start = random.randint(0, max(0, len(text) - 800))
                q = await make_question(text[start : start + 800])
                if len(q) > 10:
                    rows.append(
                        {
                            "question": q,
                            "source": source,
                            "page": doc.metadata.get("page_label"),
                        }
                    )
    Path(out).parent.mkdir(exist_ok=True)
    Path(out).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} questions to {out}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))

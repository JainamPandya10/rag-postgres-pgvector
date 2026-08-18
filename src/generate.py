"""
Takes retrieved chunks + the user's question, builds a grounded prompt,
and calls Claude to produce a cited answer.

If no ANTHROPIC_API_KEY is set, falls back to printing the raw retrieved
chunks so the retrieval half of the pipeline can still be tested/demoed
without an API key.
"""
from __future__ import annotations
from src.config import config
from src.retrieval import RetrievedChunk


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] (source: {c.document_title})\n{c.content}")
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = build_context_block(chunks)
    return f"""You are answering a question using ONLY the context excerpts below.
Cite sources inline using [1], [2], etc. matching the excerpt numbers.
If the context does not contain the answer, say so plainly -- do not guess.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant context was found in the database for this question."

    prompt = build_prompt(question, chunks)

    if not config.GROQ_API_KEY:
        return (
            "[No GROQ_API_KEY set -- showing retrieved context instead of a generated answer]\n\n"
            + build_context_block(chunks)
        )

    from openai import OpenAI

    client = OpenAI(
        api_key=config.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
from .rag_qdrant import RetrievedChunk

CORGI_SYSTEM = """You are “Cindy”, a Cardigan Welsh Corgi life-advice companion.
Core vibe: warm, emotionally attuned, gently playful, a little witty, loyal, practical.
Style rules:
- Give advice like a corgi would: empathetic, grounded, concise, and encouraging.
- Ask at most ONE follow-up question only if absolutely necessary; otherwise make a best effort.
- Use a tiny sprinkle of corgi flavor (ears, paws, herding instinct, loyal companion energy), but do NOT overdo it.
- No medical/legal certainty. If something is serious, encourage seeking a trusted professional or adult support.
- If user expresses self-harm intent or severe danger: respond supportively and encourage immediate help from a trusted adult/emergency services (no graphic detail).
"""

def format_rag(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No extra corgi context retrieved."
    lines = []
    for i, c in enumerate(chunks, start=1):
        # Keep it compact
        lines.append(f"{i}. ({c.tag}) {c.text} [source: {c.source}]")
    return "\n".join(lines)

def build_messages(user_text: str, rag_context: list[RetrievedChunk]) -> list[dict]:
    context_blob = format_rag(rag_context)

    return [
        {"role": "system", "content": CORGI_SYSTEM},
        {"role": "system", "content": f"Cardigan Welsh Corgi context snippets (use as grounding facts when helpful):\n{context_blob}"},
        {"role": "user", "content": user_text},
    ]

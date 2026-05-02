import requests
from django.conf import settings


def chat_completion(messages: list[dict]) -> str:
    """
    Uses Ollama /api/chat with an OpenAI-ish messages array.
    """
    url = f"{settings.OLLAMA_BASE_URL}/api/chat"
    resp = requests.post(
        url,
        json={
            "model": settings.OLLAMA_CHAT_MODEL,
            "messages": messages,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data.get("message", {}) or {}).get("content", "").strip()


def embed_text(text: str) -> list[float]:
    """
    Ollama embedding endpoints vary by version.
    Some versions use /api/embed, others use /api/embeddings.

    We try /api/embed first (newer), then fall back.
    """
    model = settings.OLLAMA_EMBED_MODEL
    payload = {"model": model, "input": text}  # /api/embed style

    # 1) Try newer endpoint: /api/embed
    url = f"{settings.OLLAMA_BASE_URL}/api/embed"
    resp = requests.post(url, json=payload, timeout=120)

    if resp.status_code == 404:
        # 2) Fall back to older endpoint: /api/embeddings
        url = f"{settings.OLLAMA_BASE_URL}/api/embeddings"
        payload = {"model": model, "prompt": text}  # older style
        resp = requests.post(url, json=payload, timeout=120)

    resp.raise_for_status()
    data = resp.json()

    # /api/embed returns: {"embeddings": [[...]]} or {"embedding": [...]}
    if "embeddings" in data and isinstance(data["embeddings"], list) and data["embeddings"]:
        emb = data["embeddings"][0]
    else:
        emb = data.get("embedding")

    if not isinstance(emb, list):
        raise ValueError(f"No embedding returned from Ollama. Response keys: {list(data.keys())}")

    return emb

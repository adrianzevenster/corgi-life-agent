def safety_check_and_transform(user_text: str) -> str:
    """
    Lightweight safety layer.
    - We do NOT do deep classification here; we just avoid generating risky instructions.
    - Keep it simple and let the system prompt handle tone.
    """
    t = user_text.strip()

    # If user asks for explicit wrongdoing instructions, we can still respond with safer alternatives.
    blocked_phrases = [
        "how to make a bomb",
        "how to build a bomb",
        "how to hide drugs",
        "how to hurt myself",
        "how to kill myself",
    ]
    lowered = t.lower()
    if any(p in lowered for p in blocked_phrases):
        return (
            "I can’t help with instructions for harm or wrongdoing. "
            "If you’re feeling unsafe or stuck, tell me what’s going on and I’ll help you find a safer next step."
        )

    return t

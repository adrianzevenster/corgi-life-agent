import json
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

from .services.rag_qdrant import retrieve_context
from .services.ollama_client import chat_completion
from .services.prompts import build_messages
from .services.safety import safety_check_and_transform

def index(request):
    return render(request, "chat/index.html")

@csrf_exempt
def api_chat(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8"))
        user_text = (payload.get("message") or "").strip()
        if not user_text:
            return JsonResponse({"error": "Empty message"}, status=400)

        safe_user_text = safety_check_and_transform(user_text)

        rag = retrieve_context(safe_user_text)
        messages = build_messages(user_text=safe_user_text, rag_context=rag)

        answer = chat_completion(messages)

        rag_used = [
            {
                "text": c.text,
                "source": c.source,
                "tag": c.tag,
                "score": c.score,
            }
            for c in rag
        ]
        return JsonResponse({"answer": answer, "rag_used": rag_used})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

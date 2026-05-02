from django.urls import path
from .views import index, api_chat

urlpatterns = [
    path("", index, name="index"),
    path("api/chat", api_chat, name="api_chat"),
]

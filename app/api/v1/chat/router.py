from fastapi import APIRouter, HTTPException

CHAT_ROUTER = APIRouter(prefix="/chat", tags=["chat"])

@CHAT_ROUTER.post("")
def chat(message: str):
    if len(message) == "" or len(message) > 4000:
        raise HTTPException(status_code=400, detail="Maximum length reached")
    # call service for chatting with llm
    return message
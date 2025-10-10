from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ollama import chat, ChatResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    prompt: str
    model: str = "gemma:2b"

class PromptRequest(BaseModel):
    prompt: str

@app.post("/chat")
def chat_with_ollama(request_data: ChatRequest):
    response: ChatResponse = chat(
        model=request_data.model,
        messages=[
            {
                "role": "user",
                "content": request_data.prompt
            }
        ],
    )
    return {
        "model": request_data.model,
        "response": response.message.content
    }

@app.post("/chat/deepseek")
def chat_with_deepseek(request_data: PromptRequest):
    response: ChatResponse = chat(
        model="deepseek-r1:1.5b",
        messages=[
            {
                "role": "user",
                "content": request_data.prompt
            }
        ],
    )
    return {
        "model": "deepseek-r1:1.5b",
        "response": response.message.content
    }

@app.post("/chat/llama3")
def chat_with_llama3(request_data: PromptRequest):
    response: ChatResponse = chat(
        model="llama3:8b",
        messages=[
            {
                "role": "user",
                "content": request_data.prompt
            }
        ],
    )
    return {
        "model": "llama3:8b",
        "response": response.message.content
    }

@app.post("/chat/gpt-oss-20b")
def chat_with_gptoss(request_data: PromptRequest):
    response: ChatResponse = chat(
        model="gpt-oss:20b",
        messages=[
            {
                "role": "user",
                "content": request_data.prompt
            }
        ],
    )
    return {
        "model": "gpt-oss:20b",
        "response": response.message.content
    }

@app.post("/chat/tinyllama")
def chat_with_tinyllama(request_data: PromptRequest):
    response: ChatResponse = chat(
        model="tinyllama",
        messages=[
            {
                "role": "user",
                "content": request_data.prompt
            }
        ],
    )
    return {
        "model": "tinyllama",
        "response": response.message.content
    }

@app.post("/chat/fgce01")
def chat_with_fgce01(request_data: PromptRequest):
    response: ChatResponse = chat(
        model="fgce01",
        messages=[
            {
                "role": "user",
                "content": request_data.prompt
            }
        ],
    )
    return {
        "model": "fgce01",
        "response": response.message.content
    }
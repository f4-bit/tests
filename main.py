from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from huggingface_hub import hf_hub_download


path = hf_hub_download(
    repo_id="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
    filename="Qwen3-Coder-30B-A3B-Instruct-Q4_1.gguf"
)

# Inicializar modelo GGUF (ajusta la ruta y parámetros según tu hardware)
llm = Llama(
    model_path=path,
    n_ctx=2048,
    #n_threads=8,   # Número de hilos de CPU
    n_gpu_layers=-1  # Si tienes GPU compatible, ajusta este valor
)

# Crear la aplicación FastAPI
app = FastAPI(title="Llama.cpp FastAPI Server")

# Definir schemas
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

class EmbeddingRequest(BaseModel):
    text: str

# Endpoint: Chat/Generación de texto
@app.post("/generate")
def generate_text(request: PromptRequest):
    output = llm(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stop=["</s>"]
    )
    return {"prompt": request.prompt, "response": output["choices"][0]["text"]}

# Endpoint: Generar embeddings (si el modelo soporta)
@app.post("/embedding")
def generate_embedding(request: EmbeddingRequest):
    embedding = llm.embed(request.text)
    return {"text": request.text, "embedding": embedding["data"][0]["embedding"]}

# Endpoint raíz
@app.get("/")
def root():
    return {"message": "🚀 Llama.cpp API corriendo con FastAPI"}

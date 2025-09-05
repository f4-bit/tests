import os
from llama_cpp import Llama
from llama_cpp.server.app import create_app
from fastapi import FastAPI
from uvicorn import run
from huggingface_hub import hf_hub_download

# Configura la GPU 0 (RTX 5090)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Descarga el modelo desde Hugging Face
model_path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
    filename="qwen2.5-coder-32b-instruct-q4_k_m.gguf",
    local_dir="./models"  # Carpeta donde se guardará el modelo
)

# (Opcional) Inicializa el modelo directamente en Python, si lo necesitas para uso interno
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,   # Offload todas las capas a la GPU
    n_batch=512,       # Tamaño del batch para procesamiento
    n_ctx=4096,        # Tamaño del contexto
    n_parallel=2,      # Procesamiento paralelo
    verbose=True
)

# Crea la app FastAPI del servidor llama-cpp
llama_app = create_app(
    model_path=model_path,
    n_gpu_layers=-1,
    n_ctx=4096,
    n_parallel=2
)

# Crea una app FastAPI personalizada
custom_app = FastAPI(title="Servidor LLM con FastAPI")

@custom_app.get("/")
def root():
    return {
        "message": "Servidor LLM activo. Usa /docs para ver la API o prueba con curl en /v1/chat/completions."
    }

# Monta la app de llama-cpp en la ruta /v1
custom_app.mount("/v1", llama_app)

# Ejecuta el servidor
if __name__ == "__main__":
    run(custom_app, host="0.0.0.0", port=8000)

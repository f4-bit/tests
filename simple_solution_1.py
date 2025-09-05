import os
from llama_cpp import Llama
from llama_cpp.server.app import create_app
from fastapi import FastAPI
from uvicorn import run
from huggingface_hub import hf_hub_download

# Configura la GPU 0 (RTX 5090)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Inicializa el modelo con parámetros específicos
llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
    filename="qwen2.5-coder-32b-instruct-q4_k_m.gguf",
    n_gpu_layers=-1,
    n_batch=512,
    n_ctx=4096,
    verbose=True
)

# Crea la aplicación FastAPI usando el servidor de llama-cpp-python
app = create_app(
    llm=llm,
    settings={
        "n_parallel": 2,  # Máximo batching de 2 (número de solicitudes concurrentes)
        "host": "0.0.0.0",
        "port": 8000
    }
)

# Opcional: Monta endpoints personalizados en una app FastAPI
custom_app = FastAPI(title="Servidor LLM con FastAPI")

@custom_app.get("/")
def root():
    return {"message": "Servidor LLM activo. Usa /docs para ver la API o prueba con curl en /v1/chat/completions."}

# Monta el servidor de llama-cpp en la ruta /v1
custom_app.mount("/v1", app)

# Ejecuta el servidor
if __name__ == "__main__":
    run(custom_app, host="0.0.0.0", port=8000)
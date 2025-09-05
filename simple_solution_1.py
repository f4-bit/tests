import os
from uvicorn import run
from huggingface_hub import hf_hub_download
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import ServerSettings

# Usa GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Descarga modelo
model_path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
    filename="qwen2.5-coder-32b-instruct-q4_k_m.gguf",
    local_dir="./models",
)

# Configuración del servidor (usar dicts, no ModelSettings)
server_settings = ServerSettings(
    models=[
        {
            "model": model_path,
            "n_gpu_layers": -1,
            "n_ctx": 4096,
            "n_batch": 512,
            "verbose": True,
            # "chat_format": "qwen2",  # opcional
        }
    ]
)

# Crear la app
app = create_app(server_settings)

# Endpoint extra opcional
@app.get("/")
def root():
    return {"message": "Servidor LLM listo. Endpoints en /v1/*"}

if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000)

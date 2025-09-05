import os
from uvicorn import run
from huggingface_hub import hf_hub_download
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import ModelSettings, ConfigFileSettings

# Usa la GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Descarga el modelo
model_path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
    filename="qwen2.5-coder-32b-instruct-q4_k_m.gguf",
    local_dir="./models",
)

# Crea la app del servidor OpenAI-compatible
app = create_app(
    ConfigFileSettings(
        host="0.0.0.0",
        port=8000,
        models=[
            ModelSettings(
                model=model_path,   # ¡OJO! Es "model", no "model_path"
                n_gpu_layers=-1,    # offload completo a GPU (requiere build con CUDA)
                n_ctx=4096,
                n_batch=512,
                # chat_format="qwen2",  # opcional; útil si notas prompts raros
                verbose=True,
            )
        ],
    )
)

# Endpoint de saludo opcional
@app.get("/")
def root():
    return {"message": "Servidor LLM listo. Endpoints en /v1/*"}

if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000)

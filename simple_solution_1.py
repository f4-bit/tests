import os
from fastapi import FastAPI
from uvicorn import run
from huggingface_hub import hf_hub_download
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import ModelSettings, ConfigFileSettings

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
    filename="qwen2.5-coder-32b-instruct-q4_k_m.gguf",
    local_dir="./models",
)

llama_app = create_app(
    ConfigFileSettings(
        models=[ModelSettings(
            model=model_path, n_gpu_layers=-1, n_ctx=4096, n_batch=512, verbose=True
        )]
    )
)

custom_app = FastAPI(title="Servidor LLM con FastAPI")

@custom_app.get("/")
def root():
    return {"message": "Servidor LLM activo. API OpenAI-compatible bajo /v1/*"}

# ¡Montar en "/" (no en "/v1") porque el server ya expone /v1 internamente!
custom_app.mount("/", llama_app)

if __name__ == "__main__":
    run(custom_app, host="0.0.0.0", port=8000)

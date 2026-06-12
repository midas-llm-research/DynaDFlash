import modal

app = modal.App("qwen-downloader")
volume = modal.Volume.from_name("llm-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=image,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=1800,
)
def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "Qwen/Qwen2.5-3B",
        local_dir="/models/Qwen2.5-3B",
    )
    snapshot_download(
        "Qwen/Qwen2.5-0.5B",
        local_dir="/models/Qwen2.5-0.5B",
    )
    volume.commit()

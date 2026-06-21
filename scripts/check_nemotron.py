"""Quick check of Nemotron dataset field names."""
import modal

app = modal.App("nemotron-check")
image = modal.Image.debian_slim(python_version="3.11").pip_install("datasets", "huggingface_hub")

@app.function(image=image, secrets=[modal.Secret.from_name("huggingface")], timeout=120)
def check():
    from datasets import load_dataset
    for split in ["math", "code", "chat"]:
        ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2", split=split, streaming=True)
        for i, sample in enumerate(ds):
            print(f"\n=== Split: {split} ===")
            print(f"Keys: {list(sample.keys())}")
            for k, v in sample.items():
                print(f"  {k}: {repr(v)[:300]}")
            break

@app.local_entrypoint()
def main():
    check.remote()

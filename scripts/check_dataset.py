"""Check AM-Qwen3-Distilled dataset schema."""
import modal

app = modal.App("check-dataset")
image = modal.Image.debian_slim(python_version="3.11").pip_install("datasets", "huggingface_hub")

@app.function(image=image, secrets=[modal.Secret.from_name("huggingface")], timeout=120)
def check():
    from datasets import load_dataset
    ds = load_dataset("a-m-team/AM-Qwen3-Distilled", split="train", streaming=True)
    for i, sample in enumerate(ds):
        print(f"Keys: {list(sample.keys())}")
        for k, v in sample.items():
            print(f"  {k}: {repr(v)[:400]}")
        if i >= 1:
            break

@app.local_entrypoint()
def main():
    check.remote()

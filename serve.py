import modal

app = modal.App("qwen-3-8b")
volume = modal.Volume.from_name("llm-weights")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.8.5", "fastapi")
)


@app.cls(
    image=image,
    gpu="l40s",
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    container_idle_timeout=300,
    allow_concurrent_inputs=16,
)
class QwenServer:
    @modal.enter()
    def load(self):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model="/models/Qwen3-8B",
            tensor_parallel_size=1,
            max_model_len=8192,
        )
        self.SamplingParams = SamplingParams

    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 512):
        params = self.SamplingParams(max_tokens=max_tokens, temperature=0.7)
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text

    @modal.fastapi_endpoint(method="POST")
    def completions(self, request: dict):
        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 512)
        return {"text": self.generate.local(prompt, max_tokens)}

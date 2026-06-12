import modal

app = modal.App("qwen-2.5-3b-speculative")
volume = modal.Volume.from_name("llm-weights")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("vllm", "fastapi")
)


@app.cls(
    image=image,
    gpu="l40s",
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=300,
)
@modal.concurrent(max_inputs=16)
class QwenSpeculativeServer:
    @modal.enter()
    def load(self):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model="/models/Qwen2.5-3B",
            speculative_config={
                "model": "/models/Qwen2.5-0.5B",
                "num_speculative_tokens": 5,
                "draft_tensor_parallel_size": 1,
            },
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
        params = self.SamplingParams(max_tokens=max_tokens, temperature=0.7)
        outputs = self.llm.generate([prompt], params)
        return {"text": outputs[0].outputs[0].text}

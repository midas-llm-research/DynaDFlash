# MIDAS
At MIDAS we are researching something

# Hi andrew and reese
Create a venv with `python -m venv venv`

You have to install modal with `pip install modal` and `modal setup` afterwards connect it to your account in the workspace so that it has access to the modal workspace

You should then be able to run the commands `modal run download.py` to download the models and then `modal serve [serve.py OR serve_speculative.py]` to create the endpoint. 

You can then hit it with Postman or curl e.g.
```
curl --max-time 600 -X POST https://midas-llm-research--qwen-2-5-3b-speculative-qwenspec-4201f7-dev.modal.run \
    -H "Content-Type: application/json" \
    -d '{"prompt": "What is speculative decoding?", "max_tokens": 256}'```

Replace the URL with the one you see in your console

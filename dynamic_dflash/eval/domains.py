DOMAIN_BY_SOURCE = {'magpie_output.jsonl': 'chat', 'ultrachat_output.jsonl': 'chat', 'metamathqa_train_Qwen3-8B.jsonl': 'math', 'evol_codealpaca_train_Qwen3-8B.jsonl': 'code', 'nemotron_math_Qwen3-8B.jsonl': 'math', 'orca_math_train_Qwen3-8B.jsonl': 'math', 'nemotron_chat_Qwen3-8B.jsonl': 'chat', 'lmsys_arena_train_Qwen3-8B.jsonl': 'chat', 'nemotron_stem_Qwen3-8B.jsonl': 'chat', 'autoif_train_Qwen3-8B.jsonl': 'chat', 'tulu3_Qwen3-8B.jsonl': 'chat', 'ultrafeedback_train_sft_Qwen3-8B.jsonl': 'chat', 'ultrainteract_train_Qwen3-8B.jsonl': 'chat'}

def domain_for_source(source: str) -> str:
    if source not in DOMAIN_BY_SOURCE:
        raise ValueError(f'unknown source: {source}')
    return DOMAIN_BY_SOURCE[source]

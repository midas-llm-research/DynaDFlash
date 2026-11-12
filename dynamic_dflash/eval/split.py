from __future__ import annotations
from collections import defaultdict
from typing import Any

def stratified_subset_indices(validation: Any, source_counts: dict[str, int], limit: int, subset_index: int) -> list[int]:
    if limit < 1 or subset_index < 0:
        raise ValueError('invalid held-out selection parameters')
    source_counts = {source: count for source, count in source_counts.items() if count > 0}
    total = sum(source_counts.values())
    if total != len(validation):
        raise RuntimeError('validation source counts differ from data')
    quotas = {source: count * limit // total for source, count in source_counts.items()}
    remainder = limit - sum(quotas.values())
    for source in sorted(source_counts, key=lambda s: (-(source_counts[s] * limit % total), s))[:remainder]:
        quotas[source] += 1
    by_source: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for index in range(len(validation)):
        row = validation[index]
        by_source[row['source']].append((row['prompt_hash'], index))
    if set(by_source) != set(source_counts):
        raise RuntimeError('validation sources differ from frozen manifest')
    indices = []
    for source in sorted(source_counts):
        entries = sorted(by_source[source])
        count = quotas[source]
        start = subset_index * count
        stop = start + count
        if len(entries) != source_counts[source] or stop > len(entries):
            raise RuntimeError(f'not enough disjoint {source} validation prompts')
        indices.extend((index for _, index in entries[start:stop]))
    if len(indices) != limit or len(set(indices)) != limit:
        raise RuntimeError('held-out subset size or uniqueness differs')
    return indices

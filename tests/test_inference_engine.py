import torch

from llm_from_scratch.generation import SamplingParams
from llm_from_scratch.kv_cache import KVCache, KVCacheConfig
from llm_from_scratch.paged_kv_cache import PagedKVCache, PagedKVCacheConfig
from llm_from_scratch.scheduler import (
    InferenceRequest,
    RequestStatus,
    Scheduler,
    SchedulerConfig,
)
from llm_from_scratch.transformer import Transformer


def test_cached_decode_matches_full_forward():
    torch.manual_seed(0)

    model = Transformer(
        vocab_size=16,
        embedding_dim=8,
        context_length=8,
        attention_heads=2,
        ff_hidden_dim=16,
        n_decoders=2,
        p_dropout=0,
    ).eval()
    cache = KVCache(
        KVCacheConfig(
            batch_size=1,
            max_seq_len=8,
            n_layers=2,
            n_heads=2,
            head_dim=4,
            device="cpu",
            dtype=torch.float32,
        )
    )
    input_ids = torch.randint(0, 16, (1, 5))

    with torch.no_grad():
        full_logits = model(input_ids)
        model(
            input_ids[:, :4],
            kv_cache=cache,
            start_positions=torch.tensor([0]),
            use_cache=True,
        )
        cached_logits = model(
            input_ids[:, 4:],
            kv_cache=cache,
            start_positions=torch.tensor([4]),
            use_cache=True,
        )

    torch.testing.assert_close(cached_logits[:, -1], full_logits[:, -1])


def test_paged_cache_reuses_blocks_across_layers():
    torch.manual_seed(0)

    cache = PagedKVCache(
        PagedKVCacheConfig(
            num_blocks=4,
            block_size=4,
            n_layers=2,
            n_heads=2,
            head_dim=4,
            device="cpu",
            dtype=torch.float32,
        )
    )
    cache.create_request("req1")

    layer_values = []
    for layer_idx in range(2):
        keys = torch.randn(2, 5, 4)
        values = torch.randn(2, 5, 4)
        layer_values.append((keys, values))
        cache.append("req1", layer_idx, keys, values, start_pos=0)

    table = cache.get_block_table("req1")
    assert table.token_count == 5
    assert len(table.block_ids) == 2
    for layer_idx, expected in enumerate(layer_values):
        keys, values = cache.get("req1", layer_idx)
        torch.testing.assert_close(keys, expected[0])
        torch.testing.assert_close(values, expected[1])

    cache.free_request("req1")
    assert cache.num_free_blocks() == 4


def test_scheduler_request_lifecycle():
    scheduler = Scheduler(SchedulerConfig(max_batch_size=2))
    requests = [
        InferenceRequest(str(i), f"prompt {i}", SamplingParams(max_new_tokens=1))
        for i in range(2)
    ]
    for request in requests:
        scheduler.add_request(request)

    prefill_batch = scheduler.get_prefill_batch()
    assert all(request.status == RequestStatus.PREFILLING for request in prefill_batch)

    for request in prefill_batch:
        scheduler.mark_prefill_complete(request)
        request.append_token(1)
        assert request.is_finished()
        scheduler.mark_finished(request)

    assert not scheduler.has_work()
    assert scheduler.active_requests() == []
    assert scheduler.finished_requests() == requests

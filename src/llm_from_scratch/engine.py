from uuid import uuid4
import torch
from dataclasses import dataclass
from llm_from_scratch.generation import SamplingParams
from llm_from_scratch.transformer import Transformer, TransformerConfig
from llm_from_scratch.generation import sample_next_token
from llm_from_scratch.tokenizer import (
    CARD_START,
    CharTokenizer,
    TokenizerConfig,
    TokenizerType,
    BPETokenizer,
)
from pathlib import Path
from llm_from_scratch.kv_cache import KVCache, KVCacheConfig
from llm_from_scratch.scheduler import (
    Scheduler,
    SchedulerConfig,
    InferenceRequest,
    RequestStatus,
)


@dataclass
class InferenceEngineConfig:
    model_path: str
    device: str = "cpu"
    max_seq_len: int = 1024
    max_batch_size: int = 1
    use_kv_cache: bool = False
    use_paged_cache: bool = False  # not supported yet (WIP)
    num_blocks: int = 1024
    block_size: int = 16


@dataclass
class GenerationOutput:
    request_id: str
    prompt: str
    text: str
    token_ids: list[int]
    num_generated_tokens: int
    finish_reason: str


class InferenceEngine:
    def __init__(self, config: InferenceEngineConfig):
        self.config: InferenceEngineConfig = config
        self.load_model()
        self.scheduler = Scheduler(
            SchedulerConfig(max_batch_size=self.config.max_batch_size)
        )
        # Request batching is a work in progress, currently each request needs a seperate cache
        self.request_caches: dict[str, KVCache] = {}

    def load_model(self) -> None:
        model_checkpoint = torch.load(
            self.config.model_path, map_location=self.config.device, weights_only=False
        )

        tokenizer_config = TokenizerConfig(**model_checkpoint["tokenizer_config"])
        if tokenizer_config.tokenizer_type == TokenizerType.CHAR:
            self.tokenizer = CharTokenizer(
                mapping_path=Path(tokenizer_config.mapping_path)
            )
        elif tokenizer_config.tokenizer_type == TokenizerType.BPE:
            self.tokenizer = BPETokenizer(
                mapping_path=Path(tokenizer_config.mapping_path)
            )
        else:
            raise ValueError(
                f"Invalid tokenizer type: {tokenizer_config.tokenizer_type}"
            )
        self.model_config = TransformerConfig(**model_checkpoint["model_config"])
        if self.config.max_seq_len > self.model_config.context_length:
            raise ValueError(
                f"Inference engine max sequence length = {self.config.max_seq_len} can't be longer than model context length {self.model_config.context_length}"
            )
        self.model: Transformer = Transformer(
            vocab_size=self.tokenizer.vocab_size(),
            embedding_dim=self.model_config.embedding_dim,
            context_length=self.model_config.context_length,
            ff_hidden_dim=self.model_config.ff_hidden_dim,
            attention_heads=self.model_config.attention_heads,
            n_decoders=self.model_config.n_decoders,
            p_dropout=0,
        ).to(self.config.device)
        self.model.load_state_dict(model_checkpoint["model_state_dict"])
        self.model.eval()

        print(f"Successfully loaded {self.config.model_path}")
        print("# Model:")
        print(f" - Embedding dim: {self.model_config.embedding_dim}")
        print(f" - Context length: {self.model_config.context_length}")
        print(f" - FF hidden dim: {self.model_config.ff_hidden_dim}")
        print(f" - Attention Heads: {self.model_config.attention_heads}")
        print(f" - N decoders: {self.model_config.n_decoders}")
        print("# Tokenizer:")
        print(f" - Vocab_size: {self.tokenizer.vocab_size()}")
        print(f" - Type: {self.tokenizer.get_type()}")

    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> GenerationOutput:
        # Helper method to benchmark inference engine on single request
        request_id = self.add_request(prompt, sampling_params)

        while self.scheduler.has_work():
            for output in self.step():
                if output.request_id == request_id:
                    return output

        raise RuntimeError(f"Request {request_id} finished without an output")

    def add_request(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> str:
        # My Yu-Gi-Oh! model has learned this to be the start token, I don't expect users to pass this.
        model_prompt = f"{CARD_START}\n{prompt}"
        input_ids = (
            self.tokenizer.encode(model_prompt).to(self.config.device).unsqueeze(0)
        )
        prompt_len = input_ids.shape[1]
        if prompt_len >= self.config.max_seq_len:
            raise ValueError(
                f"Prompt has too many tokens: {prompt_len} >= {self.config.max_seq_len}"
            )

        request = InferenceRequest(
            request_id=str(uuid4()),
            prompt=prompt,
            sampling_params=sampling_params,
            input_ids=input_ids,
            prompt_len=prompt_len,
        )
        self.scheduler.add_request(request=request)
        return request.request_id

    def step(self) -> list[GenerationOutput]:
        # Runs an iteration/step of the engine
        # 1. Try to prefill a batch of requests
        prefill_batch = self.scheduler.get_prefill_batch()
        self._prefill(requests=prefill_batch)
        prefill_finished = [
            request
            for request in prefill_batch
            if request.status == RequestStatus.FINISHED
        ]

        # 2. Run decode on a batch of requests
        decode_batch = self.scheduler.get_decode_batch()
        self._decode_one_step(requests=decode_batch)
        decode_finished = [
            request
            for request in decode_batch
            if request.status == RequestStatus.FINISHED
        ]

        # 3. Format outputs of requests that go completed this iteration
        return [
            self._make_output(request) for request in prefill_finished + decode_finished
        ]

    def _should_finish(self, request: InferenceRequest) -> bool:
        matches_stop_string = False
        # during benchmarking I don't want a stop string
        if request.sampling_params.stop_string is not None:
            stop_ids = self.tokenizer.encode(
                request.sampling_params.stop_string
            ).tolist()
            matches_stop_string = request.generated_ids[-len(stop_ids) :] == stop_ids

        return (
            request.is_finished()
            or request.prompt_len + request.num_generated >= self.config.max_seq_len
            or matches_stop_string
        )

    def _prefill(
        self,
        requests: list[InferenceRequest],
    ) -> None:
        # Currently each request is processed synchronously, goal is implement proper batching soon
        for request in requests:
            if request.is_finished():
                self.scheduler.mark_finished(request)
                self.request_caches.pop(request.request_id, None)
                continue

            with torch.no_grad():
                if self.config.use_kv_cache:
                    cache = KVCache(
                        config=KVCacheConfig(
                            batch_size=1,
                            max_seq_len=self.config.max_seq_len,
                            n_layers=self.model_config.n_decoders,
                            n_heads=self.model_config.attention_heads,
                            head_dim=self.model_config.embedding_dim
                            // self.model_config.attention_heads,
                            device=self.config.device,
                            dtype=torch.float,
                        )
                    )
                    self.request_caches[request.request_id] = cache
                    logits = self.model(
                        input_ids=request.input_ids,
                        kv_cache=cache,
                        request_ids=[request.request_id],
                        start_positions=torch.zeros(
                            1, dtype=torch.long, device=self.config.device
                        ),
                        use_cache=True,
                    )[:, -1, :]
                else:
                    logits = self.model(input_ids=request.input_ids)[:, -1, :]

                next_token_id = sample_next_token(
                    logits=logits, params=request.sampling_params
                )
                request.append_token(int(next_token_id[0].item()))

            self.scheduler.mark_prefill_complete(request)
            if self._should_finish(request):
                self.scheduler.mark_finished(request)
                self.request_caches.pop(request.request_id, None)

    def _decode_one_step(
        self,
        requests: list[InferenceRequest],
    ) -> None:
        # Currently each request is processed synchronously, goal is implement proper batching soon
        for request in requests:
            if self._should_finish(request):
                self.scheduler.mark_finished(request)
                self.request_caches.pop(request.request_id, None)
                continue

            with torch.no_grad():
                if self.config.use_kv_cache:
                    input_ids = torch.tensor(
                        [[request.generated_ids[-1]]],
                        device=self.config.device,
                    )
                    start_pos = request.prompt_len + request.num_generated - 1
                    logits = self.model(
                        input_ids=input_ids,
                        kv_cache=self.request_caches[request.request_id],
                        request_ids=[request.request_id],
                        start_positions=torch.tensor(
                            [start_pos], device=self.config.device
                        ),
                        use_cache=True,
                    )[:, -1, :]
                else:
                    generated_ids = torch.tensor(
                        [request.generated_ids], device=self.config.device
                    )
                    input_ids = torch.cat((request.input_ids, generated_ids), dim=1)
                    logits = self.model(input_ids=input_ids)[:, -1, :]

                next_token_id = sample_next_token(
                    logits=logits, params=request.sampling_params
                )
                request.append_token(int(next_token_id[0].item()))

            if self._should_finish(request):
                self.scheduler.mark_finished(request)
                self.request_caches.pop(request.request_id, None)

    def _make_output(
        self,
        request: InferenceRequest,
    ) -> GenerationOutput:
        finish_reason = "max_new_tokens"
        if request.sampling_params.stop_string is not None:
            stop_ids = self.tokenizer.encode(
                request.sampling_params.stop_string
            ).tolist()
            if request.generated_ids[-len(stop_ids) :] == stop_ids:
                finish_reason = "stop_token"
        if request.prompt_len + request.num_generated >= self.config.max_seq_len:
            finish_reason = "max_seq_len"

        generated_text = self.tokenizer.decode(torch.tensor(request.generated_ids))
        text = request.prompt + generated_text
        input_token_ids = (
            request.input_ids.squeeze(0).tolist()
            if request.input_ids is not None
            else []
        )

        return GenerationOutput(
            request_id=request.request_id,
            prompt=request.prompt,
            text=text,
            token_ids=input_token_ids + request.generated_ids,
            num_generated_tokens=request.num_generated,
            finish_reason=finish_reason,
        )

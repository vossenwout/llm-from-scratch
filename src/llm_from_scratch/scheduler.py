from dataclasses import dataclass, field
from enum import Enum
from torch import Tensor
from llm_from_scratch.generation import SamplingParams


class RequestStatus(Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass
class InferenceRequest:
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    input_ids: Tensor | None = None
    generated_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    prompt_len: int = 0
    num_generated: int = 0
    error: str | None = None

    def is_finished(self) -> bool:
        return (
            self.num_generated >= self.sampling_params.max_new_tokens
            or self.status == RequestStatus.FINISHED
            or self.status == RequestStatus.FAILED
        )

    def append_token(self, token_id: int) -> None:
        self.generated_ids.append(token_id)
        self.num_generated += 1


@dataclass
class SchedulerConfig:
    max_batch_size: int = 8
    max_waiting_requests: int = 128
    max_active_requests: int = 32


class Scheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.requests = {}
        self.finished = {}
        self.active_requests_count = 0
        self.waiting_requests_count = 0

    def add_request(self, request: InferenceRequest) -> None:
        if request.request_id in self.requests or request.request_id in self.finished:
            raise ValueError(f"Request {request.request_id} already exists")
        if request.status != RequestStatus.WAITING:
            raise ValueError("A new request must have waiting status")
        if self.waiting_requests_count >= self.config.max_waiting_requests:
            raise Exception(
                f"Adding new request would exceed limit of max {self.config.max_waiting_requests} requests"
            )
        self.requests[request.request_id] = request
        self.waiting_requests_count += 1

    def has_work(self) -> bool:
        return (self.active_requests_count + self.waiting_requests_count) > 0

    def get_prefill_batch(self) -> list[InferenceRequest]:
        batch = []
        available_slots = self.config.max_active_requests - self.active_requests_count
        batch_size = min(self.config.max_batch_size, available_slots)
        if batch_size <= 0:
            return batch

        for request in self.requests.values():
            if request.status == RequestStatus.WAITING:
                request.status = RequestStatus.PREFILLING
                self.active_requests_count += 1
                self.waiting_requests_count -= 1
                batch.append(request)
                if len(batch) == batch_size:
                    break
        return batch

    def get_decode_batch(self) -> list[InferenceRequest]:
        batch = []
        for request in self.requests.values():
            if request.status == RequestStatus.DECODING:
                batch.append(request)
                if len(batch) == self.config.max_batch_size:
                    break
        return batch

    def mark_prefill_complete(self, request: InferenceRequest) -> None:
        if request.status != RequestStatus.PREFILLING:
            raise ValueError("Only a prefilling request can start decoding")
        request.status = RequestStatus.DECODING

    def mark_finished(self, request: InferenceRequest) -> None:
        if request.request_id not in self.requests:
            raise ValueError(f"Request {request.request_id} is not active")
        request.status = RequestStatus.FINISHED
        self.active_requests_count -= 1
        del self.requests[request.request_id]
        self.finished[request.request_id] = request

    def active_requests(self) -> list[InferenceRequest]:
        return [
            request
            for request in self.requests.values()
            if request.status in (RequestStatus.PREFILLING, RequestStatus.DECODING)
        ]

    def finished_requests(self) -> list[InferenceRequest]:
        return list(self.finished.values())

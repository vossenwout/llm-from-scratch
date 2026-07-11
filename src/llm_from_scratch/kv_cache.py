from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class KVCacheConfig:
    batch_size: int
    max_seq_len: int
    n_layers: int
    n_heads: int
    head_dim: int
    device: torch.device | str
    dtype: torch.dtype


class KVCache:
    """
    Simple contiguous kv cache

    Stores:
        keys: [n_layers, B, H, max_seq_len, D]
        values: [n_layers, B, H, max_seq_len, D]
    """

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.keys = torch.empty(
            size=(
                self.config.n_layers,
                self.config.batch_size,
                self.config.n_heads,
                self.config.max_seq_len,
                self.config.head_dim,
            ),
            device=self.config.device,
            dtype=self.config.dtype,
        )
        self.values = torch.empty(
            size=(
                self.config.n_layers,
                self.config.batch_size,
                self.config.n_heads,
                self.config.max_seq_len,
                self.config.head_dim,
            ),
            device=self.config.device,
            dtype=self.config.dtype,
        )

    def append(
        self,
        layer_idx: int,
        batch_idx: int,
        key: Tensor,
        value: Tensor,
        start_pos: int,
    ) -> None:
        # keys/values: h x T_new x d
        # start_pos: first position to write into
        self.keys[layer_idx, batch_idx, :, start_pos : start_pos + key.shape[1]] = key
        self.values[layer_idx, batch_idx, :, start_pos : start_pos + value.shape[1]] = (
            value
        )

    def append_batch(
        self,
        layer_idx: int,
        keys: Tensor,
        values: Tensor,
        start_positions: Tensor,
    ) -> None:
        # keys/values: B x H x T_new x D
        # start_positions: B
        # I naively assume all start positions are equal
        start_pos = int(start_positions[0].item())
        self.keys[layer_idx, :, :, start_pos : start_pos + keys.shape[2]] = keys
        self.values[layer_idx, :, :, start_pos : start_pos + values.shape[2]] = values

    def get(
        self,
        layer_idx: int,
        batch_idx: int,
        end_pos: int,
    ) -> tuple[Tensor, Tensor]:
        # returns key/value: H x end_pos x D
        return (
            self.keys[layer_idx, batch_idx, :, :end_pos],
            self.values[layer_idx, batch_idx, :, :end_pos],
        )

    def get_batch(
        self,
        layer_idx: int,
        end_positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # simple version assume all end_positions are equal. Later versions can support padding/masking for variable lengths.
        # how to do that?
        # should we just take max and then pad for that oor something?
        end_pos = int(end_positions[0].item())
        return (
            self.keys[layer_idx, :, :, :end_pos],
            self.values[layer_idx, :, :, :end_pos],
        )

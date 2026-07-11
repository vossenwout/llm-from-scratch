from dataclasses import dataclass, field
import torch
from torch import Tensor


@dataclass
class PagedKVCacheConfig:
    num_blocks: int
    block_size: int
    n_layers: int
    n_heads: int
    head_dim: int
    device: torch.device | str
    dtype: torch.dtype


@dataclass
class BlockTable:
    request_id: str
    # mapping from logical -> physical block
    block_ids: list[int] = field(default_factory=list)
    token_count: int = 0

    def logical_to_physical(
        self,
        logical_token_idx: int,
        block_size: int,
    ) -> tuple[int, int]:
        """
        Returns:
            block_id, offset
        """
        if not (0 <= logical_token_idx < self.token_count):
            raise ValueError(
                f"logical_token_idx {logical_token_idx} is invalid for block table with {self.token_count} tokens"
            )
        return self.block_ids[
            logical_token_idx // block_size
        ], logical_token_idx % block_size

    def num_blocks_for(
        self,
        token_count: int,
        block_size: int,
    ) -> int:
        return (token_count + block_size - 1) // block_size

    def additional_blocks_for(
        self,
        token_count: int,
        block_size: int,
    ) -> int:
        return max(
            0,
            self.num_blocks_for(token_count, block_size) - len(self.block_ids),
        )


class PagedKVCache:
    """
    Toy paged KV cache.

    Physical storage:
        key_blocks: [num_blocks, n_layers, H, block_size, D]
        value_blocks: [num_blocks, n_layers, H, block_size, D]
    """

    def __init__(self, config: PagedKVCacheConfig):
        self.config = config
        self.key_blocks = torch.empty(
            size=(
                self.config.num_blocks,
                self.config.n_layers,
                self.config.n_heads,
                self.config.block_size,
                self.config.head_dim,
            ),
            device=self.config.device,
            dtype=self.config.dtype,
        )
        self.value_blocks = torch.empty(
            size=(
                self.config.num_blocks,
                self.config.n_layers,
                self.config.n_heads,
                self.config.block_size,
                self.config.head_dim,
            ),
            device=self.config.device,
            dtype=self.config.dtype,
        )
        self.request_to_bt = dict()
        self.free_block_ids = list(range(self.config.num_blocks))
        # count how many block_tables refer to this block (prefix sharing later)
        self.block_ref_counts = [0] * self.config.num_blocks

    def create_request(self, request_id: str) -> None:
        if request_id in self.request_to_bt:
            raise ValueError(f"Request {request_id} has already been created")
        self.request_to_bt[request_id] = BlockTable(request_id)

    def free_request(self, request_id: str) -> None:
        block_table = self.get_block_table(request_id=request_id)

        for block_id in block_table.block_ids:
            self.block_ref_counts[block_id] -= 1
            if self.block_ref_counts[block_id] == 0:
                self.free_block(block_id)

        del self.request_to_bt[request_id]

    def allocate_block(self) -> int:
        if self.num_free_blocks() == 0:
            raise Exception("Can't allocate new block, all are full")
        return self.free_block_ids.pop()

    def free_block(self, block_id: int) -> None:
        self.free_block_ids.append(block_id)

    def ensure_capacity(
        self,
        request_id: str,
        final_token_count: int,
    ) -> None:
        block_table = self.get_block_table(request_id=request_id)
        additional_blocks = block_table.additional_blocks_for(
            token_count=final_token_count,
            block_size=self.config.block_size,
        )
        num_free_blocks = self.num_free_blocks()
        if additional_blocks > num_free_blocks:
            raise RuntimeError(
                f"request: {request_id} requires {additional_blocks} new blocks but only {num_free_blocks} are free"
            )

    def append(
        self,
        request_id: str,
        layer_idx: int,
        key: Tensor,
        value: Tensor,
        start_pos: int,
    ) -> None:
        """
        Args:
            key: shape [H, T_new, D]
            value: shape [H, T_new, D]
            start_pos: logical position of the first new token
        """
        block_table = self.get_block_table(request_id=request_id)
        if not (0 <= start_pos <= block_table.token_count):
            raise ValueError(
                f"start_pos {start_pos} would leave a gap after "
                f"{block_table.token_count} cached tokens"
            )

        num_new_tokens = key.shape[1]
        write_end = start_pos + num_new_tokens
        self.ensure_capacity(
            request_id=request_id,
            final_token_count=write_end,
        )

        additional_blocks = block_table.additional_blocks_for(
            token_count=write_end,
            block_size=self.config.block_size,
        )
        for _ in range(additional_blocks):
            block_id = self.allocate_block()
            block_table.block_ids.append(block_id)
            self.block_ref_counts[block_id] += 1

        # Write one block-sized slice at a time. Logical token positions choose
        # the physical block and offsets within that block.
        source_start = 0
        while source_start < num_new_tokens:
            logical_pos = start_pos + source_start
            logical_block = logical_pos // self.config.block_size
            block_offset = logical_pos % self.config.block_size
            physical_block = block_table.block_ids[logical_block]

            n_tokens = min(
                num_new_tokens - source_start,
                self.config.block_size - block_offset,
            )
            source_end = source_start + n_tokens
            block_end = block_offset + n_tokens

            self.key_blocks[physical_block, layer_idx, :, block_offset:block_end] = key[
                :, source_start:source_end
            ]
            self.value_blocks[physical_block, layer_idx, :, block_offset:block_end] = (
                value[:, source_start:source_end]
            )

            source_start = source_end

        block_table.token_count = write_end

    def get(
        self,
        request_id: str,
        layer_idx: int,
    ) -> tuple[Tensor, Tensor]:
        """Toy implementation gathers blocks into contiguous K/V tensors

        Returns:
            key: shape [H, T_total, D]
            value: shape [H, T_total, D]
        """
        block_table = self.get_block_table(request_id=request_id)

        if block_table.token_count == 0:
            raise Exception("Called get on a request without any tokens cached")

        start = 0
        end = min(start + self.config.block_size, block_table.token_count)
        physical_block, _ = block_table.logical_to_physical(
            logical_token_idx=start, block_size=self.config.block_size
        )
        # H, T_in_block, head_dim
        key = self.key_blocks[physical_block, layer_idx, :, : end - start]
        val = self.value_blocks[physical_block, layer_idx, :, : end - start]

        for logical_block in range(1, len(block_table.block_ids)):
            start = logical_block * self.config.block_size
            end = min(start + self.config.block_size, block_table.token_count)
            physical_block, _ = block_table.logical_to_physical(
                logical_token_idx=start, block_size=self.config.block_size
            )
            key = torch.cat(
                tensors=(
                    key,
                    self.key_blocks[physical_block, layer_idx, :, : end - start],
                ),
                dim=1,
            )
            val = torch.cat(
                tensors=(
                    val,
                    self.value_blocks[physical_block, layer_idx, :, : end - start],
                ),
                dim=1,
            )

        return key, val

    def get_block_table(self, request_id: str) -> BlockTable:
        if request_id not in self.request_to_bt:
            raise ValueError(f"Request {request_id} doesn't exist in block table")
        return self.request_to_bt[request_id]

    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def num_used_blocks(self) -> int:
        return self.config.num_blocks - self.num_free_blocks()

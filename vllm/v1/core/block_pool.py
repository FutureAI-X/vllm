# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from collections.abc import Iterable
from typing import Callable, Optional

from vllm.distributed.kv_events import (AllBlocksCleared, BlockRemoved,
                                        BlockStored, KVCacheEvent)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (BlockHash, BlockHashWithGroupId,
                                         FreeKVCacheBlockQueue, KVCacheBlock,
                                         generate_block_hash_extra_keys,
                                         hash_block_tokens)
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        enable_kv_cache_events: Whether to enable kv cache events.
    """
    """
    BlockPool 用于管理 KVCacheBlocks
    1. 它提供了分配、释放和缓存 kv cache blocks 的方法
    2. free_block_queue: 按驱逐顺序存储空闲块，以支持分配、释放和缓存驱逐
    3. cached_block_hash_to_block: 在块哈希和缓存块之间建立映射，以支持通过块哈希查找缓存块
    
    Args:
        num_gpu_blocks: 可用的 block 数量
        enable_caching: 是否启用 prefix caching.
        enable_kv_cache_events: 是否启用 kv cache events.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        enable_kv_cache_events: bool = False,
    ):
        # 参数校验与赋值
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        # 是否启用 prefix caching
        self.enable_caching = enable_caching

        # 在此处初始化所有的 KV cache block
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]

        # 空闲 block 队列
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # {block_hash: {block ID: block}}. A cached block is
        # a full block with a block hash that can be used for prefix caching.
        # The cached block may be used by running requests or in the
        # free_block_queue that could potentially be evicted.
        # NOTE: We currently don't de-duplicate the blocks in the cache,
        # meaning that if a block becomes full and is cached, we don't check
        # if there is already an identical block in the cache. This is because
        # we want to make sure the allocated block IDs won't change so that
        # block tables are append-only.
        """
        {block_hash: {block ID: block}}
        一个 cached block 是一个 full block (token 已满), 它可以用于 prefix caching, 它有一个 block hash 
        cached block 可能被正在运行的请求使用，也可能位于可能被驱逐的空闲块队列（free_block_queue）中
        
        注意：我们目前不对缓存中的块进行去重处理，这意味着如果一个块变满并被缓存，我们不会检查缓存中是否已存在相同的块。这是因为我们希望确保已分配的块ID不会改变，从而使块表保持仅追加（append-only）的特性
        """
        self.cached_block_hash_to_block: dict[BlockHashWithGroupId, dict[
            int, KVCacheBlock]] = defaultdict(dict)

        """
        用于表示一个 block ID为0的占位块（placeholder block）。
        null_block 的引用计数（ref_cnt）不会被维护，在使用时需要特别注意，避免释放它。
        """
        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        # KV Cache event 相关 (分布式环境需要)
        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

    def get_cached_block(
            self, block_hash: BlockHash,
            kv_cache_group_ids: list[int]) -> Optional[list[KVCacheBlock]]:
        """Get the cached block by the block hash for each group in 
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        """
        获取 cached block
        
        Args:
            block_hash          要查找的 block hash 值
            kv_cache_group_ids  kv cache groups
        """
        # Step1 定义返回值
        cached_blocks = []
        # Step2 遍历每个 group
        for group_id in kv_cache_group_ids:
            # 1. 获取 cached block
            cached_blocks_one_group = self.cached_block_hash_to_block.get(
                BlockHashWithGroupId(block_hash, group_id))
            # 2. 如果找不到, 直接返回 None
            if not cached_blocks_one_group:
                return None
            # 3. 如果有重复的, 取第1个
            first_block = next(iter(cached_blocks_one_group.values()))
            cached_blocks.append(first_block)

        # Step3 执行返回
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        block_hashes: list[BlockHash],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        hash_fn: Callable,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it computes the
        block hashes for the blocks starting from `num_cached_blocks` to
        `num_full_blocks`, updating the metadata for each block
        and caching them in the `cached_block_hash_to_block`.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            block_hashes: Block hashes of the blocks in the request. Note that
            this list may be shorter than the blocks list. In this case the
            missed block hash will be computed in this function.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
            hash_fn: The hash function to use for block hashes.
        """
        """
        Cache a list of full blocks for prefix caching
        - 该函数接收一个块列表，这些块的块哈希元数据将被更新和缓存
        - 给定一个请求，它会计算从 num_cached_blocks 到 num_full_blocks 的块的块哈希
        - 更新每个块的元数据并将它们缓存在 cached_block_hash_to_block 中
        
        Args:
            request: 需要缓存块的请求
            blocks: 请求中的所有块
            block_hashes: 请求中块的块哈希。注意这个列表可能比块列表短。在这种情况下，缺失的块哈希将在此函数中计算
            num_cached_blocks: 已经缓存的块数量
            num_full_blocks: 变满并应该在此函数后缓存的块数量
            block_size: 每个块中的令牌数量
            kv_cache_group_id: KV缓存组的ID
            hash_fn: 用于块哈希的哈希函数
        
        """
        # Step1 前置处理
        # 1. 如果所有的块都已经被缓存, 则直接跳过
        if num_cached_blocks == num_full_blocks:
            return
        # 2. 未被 cache 的 full block
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        # 3. 从参数中获取未被 cache 的 block hashes
        assert len(block_hashes) >= num_cached_blocks
        new_block_hashes = block_hashes[num_cached_blocks:]

        # 4. 获取已被 cache 的最后一个 full block 的 hash
        # Update the new blocks with the block hashes through the chain.
        if num_cached_blocks == 0:
            prev_block_hash_value = None
        else:
            prev_block = blocks[num_cached_blocks - 1]
            assert prev_block.block_hash is not None
            prev_block_hash_value = prev_block.block_hash.get_hash_value()

        parent_block_hash = prev_block_hash_value
        new_hashes: Optional[list[int]] = ([] if self.enable_kv_cache_events
                                           else None)

        # Step2 循环处理所有需要 cache 的 full block
        for i, blk in enumerate(new_full_blocks):
            assert blk.block_hash is None

            # 1. 获取 block hash
            if i < len(new_block_hashes):
                # (1) 如果入参 new_block_hashes 中有 当前 block 的 hash, 则直接使用
                # The block hash may already be computed in
                # "get_computed_blocks" if the tokens are not generated by
                # this request (either the prompt tokens or the previously
                # generated tokens with preemption), or by other
                # single_type_managers with the same block_size.
                # In this case we simply reuse the block hash.
                block_hash = new_block_hashes[i]
            else:
                # (2) 否则, 则计算 block hash
                # Otherwise compute the block hash and cache it in the request
                # in case it will be preempted in the future.
                # 获取当前 block 的 token
                blk_idx = num_cached_blocks + i
                start_token_idx = blk_idx * block_size
                end_token_idx = (blk_idx + 1) * block_size
                block_tokens = request.all_token_ids[
                    start_token_idx:end_token_idx]
                assert len(block_tokens) == block_size, (
                    f"Expected {block_size} tokens, got "
                    f"{len(block_tokens)} at {blk_idx}th block for request "
                    f"{request.request_id}({request})")

                # 为多模态输入生成额外的键。请注意，由于我们只有在块已完成生成令牌时才会到达此分支，因此我们只需要考虑最后一个多媒体输入。
                # Generate extra keys for multi-modal inputs. Note that since
                # we reach to this branch only when the block is completed with
                # generated tokens, we only need to consider the last mm input.
                extra_keys, _ = generate_block_hash_extra_keys(
                    request, start_token_idx, end_token_idx, -1)

                # 计算当前 block 的 hash
                # Compute the hash of the current block.
                block_hash = hash_block_tokens(hash_fn, prev_block_hash_value,
                                               block_tokens, extra_keys)
                block_hashes.append(block_hash)

            # 2. 将当前 block 更新到缓存中
            # Update and added the full block to the cache.
            block_hash_with_group_id = BlockHashWithGroupId(
                block_hash, kv_cache_group_id)
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block[block_hash_with_group_id][
                blk.block_id] = blk

            # 3. 分布式 KV Cache Event 数据处理
            if new_hashes is not None:
                new_hashes.append(block_hash.hash_value)

            # 4. 更新 prev_block_hash_value, 以便下一个 block 使用
            prev_block_hash_value = block_hash.hash_value

        # Step3 分布式 KV Cache Event 处理
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.
                    all_token_ids[num_cached_blocks *
                                  block_size:num_full_blocks * block_size],
                    block_size=block_size,
                    lora_id=request.lora_request.id
                    if request.lora_request else None,
                ))

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        """
        从 free block pool 中获取 new block，注意此处不会检查 block cache
        
        Args:
            num_blocks: 要分配的 block 个数
            
        Returns:
            new block list
        """
        # 1. 检查 free block pool 中是否有足够的 blocks 待分配
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool")

        # 2. 从 free block pool 中获取 n个 new block
        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        # 3. 缓存处理 and 引用计数
        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1

        # 4. 执行返回
        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        """
        [当分配新块时会调用此方法] 如果新分配的 block 缓存在 cached_block_hash_to_block，重置 block hash 并从 cache 中移除 block
        """
        # Step1 前置处理
        # 1. 获取 block hash
        block_hash = block.block_hash
        # 2. 如果 block hash 为 None, 则不需要处理缓存
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return False

        # Step2 缓存处理
        # 1. 根据 block hash 从 cache 中获取 block
        blocks_by_id = self.cached_block_hash_to_block.get(block_hash)
        # 2. 如果 block 不在 cache 中，则不需要处理缓存
        if blocks_by_id is None:
            # block_hash not found in cached_block_hash_to_block,
            # eviction is not needed
            return False

        # 3. 重置 block hash
        block.reset_hash()
        # 4. 从 cache 中移除 block
        blocks_by_id.pop(block.block_id, None)
        if len(blocks_by_id) == 0:
            del self.cached_block_hash_to_block[block_hash]

        # Step3 分布式 KV Cache Event 处理
        if self.enable_kv_cache_events:
            # FIXME (Chen): Not sure whether we should return `hash_value`
            # or `(hash_value, group_id)` here. But it's fine now because
            # we disable hybrid kv cache manager when kv cache event is
            # enabled, so there is only one group.
            self.kv_event_queue.append(
                BlockRemoved(block_hashes=[block_hash.get_hash_value()]))

        # Step4 返回
        return True

    def touch(self, blocks: tuple[list[KVCacheBlock], ...]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        """
        将 block 的引用计数 +1, 并可能从空闲队列中移除该块, 当一个块被具有相同前缀的另一个请求命中时使用此功能
        """
        for blocks_per_group in blocks:
            for block in blocks_per_group:
                # ref_cnt=0 means this block is in the free list (i.e. eviction
                # candidate), so remove it.
                # 如果ref_cnt=0, 说明该块在空闲队列中, 需要从 free 队列中移除
                if block.ref_cnt == 0 and not block.is_null:
                    self.free_block_queue.remove(block)
                # 引用次数 +1
                block.ref_cnt += 1

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        """
        释放 blocks list，这些块应该按照它们的驱逐优先级排序，其中第一个块将首先被驱逐
        """
        # Materialize the iterable to allow multiple passes.
        # 1. 将 Iterable 转换为 list，以便进行多次遍历
        blocks_list = list(ordered_blocks)

        # 2. 遍历 blocks list，将 ref_cnt -1
        for block in blocks_list:
            block.ref_cnt -= 1

        # 3. 将 ref_cnt 为 0 的 block 添加到 free 队列
        self.free_block_queue.append_n([
            block for block in blocks_list
            if block.ref_cnt == 0 and not block.is_null
        ])

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        """
        此函数可用于RLHF流程中，在权重更新后使前缀缓存失效，或用于重置基准测试的前缀缓存状态
        """
        # 获取已使用的块数
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        # 因为 null block 在取出的时候将 num_free_block - 1, 所以当差值为 1 认为, 所有的块都是 free
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet", num_used_blocks - 1)
            return False

        # 清空 cache
        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = defaultdict(dict)

        # 重置 block hash
        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        # 日志记录
        logger.info("Successfully reset prefix cache")

        # KV cache event
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        # 执行返回
        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        """获取 free blocks pool 中的数量"""
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        """获取 KV Cache 使用率"""
        return 1.0 - (self.get_num_free_blocks() / self.num_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.
        
        Returns:
            A list of KV cache events.
        """
        """
        原子性地获取所有事件并清空队列
        
        Returns:
            KV cache events 事件列表
        """
        # 1. 如果未启用 KV cache 事件, 则返回空列表
        if not self.enable_kv_cache_events:
            return []

        # 2. 获取所有事件
        events = self.kv_event_queue

        # 3. 清空事件队列
        self.kv_event_queue = []

        # 4. 返回事件
        return events

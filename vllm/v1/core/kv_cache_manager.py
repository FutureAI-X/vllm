# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.utils import sha256, sha256_cbor_64bit
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_utils import (BlockHash, KVCacheBlock,
                                         hash_request_tokens, init_none_hash)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """
    """KVCacheManager 的分配结果，作为调度器和 KVCacheManager 之间的接口，用于向调度器隐藏 KVCacheManager 的内部数据结构。"""
    blocks: tuple[list[KVCacheBlock], ...]
    """
    blocks[i][j] refers to the i-th kv_cache_group and the j-th block of tokens.
    We don't use block of tokens as the outer dimension because it assumes all
    kv_cache_groups have the same number of blocks, which is true for now but 
    will be broken if we want to give different block_size to different 
    kv_cache_groups in the future.
    """
    """blocks[i][j] 表示第 i 个 kv_cache_group 和第 j 个 token 块。 我们不使用 token 块作为外层维度，因为这假设所有 kv_cache_groups 具有相同数量的块，目前这是正确的，但如果我们想要为不同的 kv_cache_groups 提供不同的 block_size，这种假设将不再成立。"""

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        """拼接两个 KVCacheBlocks 实例
        
        示例:
            # 第一个实例，包含2个缓存组，每组各有2个块
            blocks1 = KVCacheBlocks((
                [block_a1, block_a2],      # 第1个KV缓存组
                [block_b1, block_b2]       # 第2个KV缓存组
            ))
            
            # 第二个实例，同样包含2个缓存组，每组各有1个块
            blocks2 = KVCacheBlocks((
                [block_a3],                # 第1个KV缓存组
                [block_b3]                 # 第2个KV缓存组
            ))
            
            # 相加操作
            result = blocks1 + blocks2
            
            # 结果为：
            result = KVCacheBlocks((
                [block_a1, block_a2, block_a3],  # 第1个KV缓存组，块列表连接
                [block_b1, block_b2, block_b3]   # 第2个KV缓存组，块列表连接
            ))
        """
        return KVCacheBlocks(
            tuple(blk1 + blk2
                  for blk1, blk2 in zip(self.blocks, other.blocks)))

    def get_block_ids(self) -> tuple[list[int], ...]:
        """
        Converts the KVCacheBlocks instance to block_ids.
        
        Returns:
            tuple[list[int], ...]: A tuple of lists where
            * the outer tuple corresponds to KV cache groups
            * each inner list contains the block_ids of the blocks in that group
        """
        """
        它遍历每个KV缓存组中的所有块，并提取每个块的 block_id。
        返回值是一个嵌套结构：外层元组对应不同的KV缓存组，内层列表包含该组中所有块的ID
        """
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        """获取未哈希块的块ID列表"""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [
            block.block_id for block in self.blocks[0]
            if block.block_hash is None
        ]

    def new_empty(self) -> "KVCacheBlocks":
        """Creates a new KVCacheBlocks instance with no blocks."""
        """这个方法创建一个新的空 KVCacheBlocks 实例，保持与当前实例相同的KV缓存组数量结构，但不包含任何实际块。"""
        return KVCacheBlocks(tuple([] for _ in range(len(self.blocks))))


class KVCacheManager:

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        caching_hash_algo: str = "builtin",
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
    ) -> None:
        # 模型最大长度
        self.max_model_len = max_model_len

        # 没有 Attention 的 Model 无需 KV cache，因此也没有 prefix cache
        if len(kv_cache_config.kv_cache_groups) == 0:
            # Attention free models don't have kv cache,
            # thus don't need prefix caching.
            enable_caching = False
        self.enable_caching = enable_caching

        # kv cache hash 函数
        self.caching_hash_fn = (
            sha256_cbor_64bit if caching_hash_algo == "sha256_cbor_64bit" else
            sha256 if caching_hash_algo == "sha256" else hash)
        init_none_hash(self.caching_hash_fn)

        # 是否使用 eagle
        self.use_eagle = use_eagle
        self.log_stats = log_stats

        # 监控统计信息
        # FIXME: make prefix cache stats conditional on log_stats
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        # 设置 block_size 大小
        self.block_size: Optional[int] = None
        if self.enable_caching:
            # 检查所有 kv cache group 的 block_size 是否一致，set函数用于去重
            assert len(
                set(g.kv_cache_spec.block_size
                    for g in kv_cache_config.kv_cache_groups)
            ) == 1, "Only one block size is supported for now"
            self.block_size = kv_cache_config.kv_cache_groups[
                0].kv_cache_spec.block_size

        # 获取 kv cache coordinator
        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            caching_hash_fn=self.caching_hash_fn,
            enable_kv_cache_events=enable_kv_cache_events,
        )

        # kv cache group 个数
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)

        # block pool
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        """
        一个 dict/map, 维护了每个 request_id 到 多个 KV Block Hash 的映射
        dict[request_id -> list[BlockHash]]
        此属性存在的目的是为了避免在调用 get_computed_blocks 与 allocate_slots 时重新计算 block hashes
        """
        self.req_to_block_hashes: defaultdict[
            str, list[BlockHash]] = defaultdict(list)

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        """KV Cache 的使用率"""
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> Optional[PrefixCacheStats]:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        """获取并重置 Prefix cache 的统计信息"""
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(self,
                            request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        """
        从本地缓存中获取请求已计算(缓存)的 blocks, 注意必须是完整的块
        
        Args:
            request: 要计算的请求
            
        Returns:
            - 请求已计算的 blocks list
            - 已计算的令牌数
        """
        # Step1 是否跳过前缀缓存判断
        # 如果满足任一条件, 则跳过前缀缓存查找, 返回空
        # 1. 禁用了前缀缓存
        # 2. 请求需要 prompt logprobs
        # Prefix caching is disabled or
        # When the request requires prompt logprobs, we skip prefix caching.
        if (not self.enable_caching
                or (request.sampling_params is not None
                    and request.sampling_params.prompt_logprobs is not None)):
            return self.create_empty_block_list(), 0

        # Step2 获取 request 的 Block 的 Hash
        # 1. 根据 request_id 获取已缓存的 Block Hash: 如果此 request 之前尝试过调度，则从 req_to_block_hashes 中获取
        # The block hashes for the request may already be computed
        # if the scheduler has tried to schedule the request before.
        block_hashes = self.req_to_block_hashes[request.request_id]

        # 2. 如果根据 request id 没有找到，则使用 hash_request_tokens 函数计算哈希值并存储到 req_to_block_hashes 中
        if not block_hashes:
            # 判断 block_size
            assert self.block_size is not None
            # 计算 block hash
            block_hashes = hash_request_tokens(self.caching_hash_fn,
                                               self.block_size, request)
            # 更新到 req_to_block_hashes
            self.req_to_block_hashes[request.request_id] = block_hashes

        # Step3 获取已缓存的块
        """
        注意:
        - 当所有令牌都命中缓存时，我们必须重新计算最后一个令牌以获得logits
        - 因此，将max_cache_hit_length设置为prompt_length - 1
        - 这可能会触发整个块的重新计算，而不仅仅是单个最后一个令牌, 因为allocate_slots()要求num_computed_tokens与块大小对齐
        - 移除这个限制可能会在未来略微提升性能
        """
        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        # 1. 设置最大缓存命中长度为请求令牌数减1, 这是因为即使所有令牌都命中缓存，也需要重新计算最后一个令牌以获取 logits
        max_cache_hit_length = request.num_tokens - 1
        # 2. 调用协调器的 find_longest_cache_hit 方法查找最长的缓存命中 (返回已计算的块和已计算的令牌数)
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(block_hashes,
                                                    max_cache_hit_length))

        # Step4 如果启用了统计日志，则更新前缀缓存统计信息
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.requests += 1
            self.prefix_cache_stats.queries += request.num_tokens
            self.prefix_cache_stats.hits += num_new_computed_tokens

        # Step5 返回已缓存的块
        return KVCacheBlocks(computed_blocks), num_new_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVCacheBlocks] = None,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
    ) -> Optional[KVCacheBlocks]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of tokens to allocate, including external
                tokens. Note that this does not include tokens that have
                already been computed locally (i.e. new_computed_blocks).
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed 
                tokens.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such 
                as eagle.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.

        Blocks layout:
        ```
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        ```
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        """
        Args:
            request: 需要分配槽位的请求对象
            num_new_tokens: 需要分配的新token数量，包括外部token
            num_new_computed_tokens: 新命中的前缀缓存token数量
            new_computed_blocks: 新命中的缓存块
            num_lookahead_tokens: 需要预分配的投机解码token数量
            delay_cache_blocks: 是否延迟缓存块
        """
        # Step1 参数验证
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        # Step2 获取或初始化新的已计算 Block
        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = tuple(
                [] for _ in range(len(self.kv_cache_config.kv_cache_groups)))

        # Step3 清理跳过的 Block (如滑动窗口外的 token)
        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        self.coordinator.remove_skipped_blocks(request.request_id,
                                               request.num_computed_tokens)

        # Step4 计算总的需要槽位的 token 数量，确保不会超出模型长度
        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = (request.num_computed_tokens +
                               num_new_computed_tokens)
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len)

        # Step5 计算需要分配的 Block 数量
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None

        # Step5 前缀缓存的处理
        # 如果启用 prefix caching, 则标记已计算的块防止被驱逐
        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_block_list)
        else:
            assert not any(new_computed_block_list), (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # 将新计算的块保存到请求块中
        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        self.coordinator.save_new_computed_blocks(request.request_id,
                                                  new_computed_block_list)

        # Step6 分配新的 Block
        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot)

        # Step7 缓存处理和返回结果
        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        # 如果不启用缓存或需要延迟缓存，则直接返回新分配的块
        if not self.enable_caching or delay_cache_blocks:
            return KVCacheBlocks(new_blocks)

        # 缓存已确认的token，排除可能被拒绝的推测token
        # NOTE(woosuk): We want to commit (cache) up to num_computed_tokens +
        # num_new_tokens, but must exclude "non-committable" tokens (e.g.,
        # draft tokens that could be rejected). Therefore, we cap the number
        # at `request.num_tokens`, ensuring only "finalized" tokens are cached.
        num_tokens_to_cache = min(num_computed_tokens + num_new_tokens,
                                  request.num_tokens)
        self.coordinator.cache_blocks(
            request,
            self.req_to_block_hashes[request.request_id],
            num_tokens_to_cache,
        )

        return KVCacheBlocks(new_blocks)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that he tail blocks are evicted 
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        self.coordinator.free(request.request_id)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> list[int]:
        """Calculate the number of common prefix blocks shared by all requests
        in the RUNNING state for each kv cache group.

        The function determines this by selecting any request and iterating
        through its blocks.  A block is considered a common prefix block if its
        `ref_cnt` equals the total number of requests in the RUNNING state.

        NOTE(woosuk): The number of requests in the RUNNING state is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because the RUNNING state only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must be in the RUNNING state, the inverse
        is not necessarily true. There may be RUNNING requests that are not
        scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled RUNNING requests that do not
        share the common prefix. Currently, this case cannot be easily detected,
        so the function returns 0 in such cases.

        Args:
            request: Any request in the RUNNING state, used to identify the
                common prefix blocks.
            num_running_requests: The total number of requests in the RUNNING
                state. This can be different from the number of scheduled
                requests in the current step.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache 
            group.
        """
        assert request.status == RequestStatus.RUNNING
        return self.coordinator.get_num_common_prefix_blocks(
            request.request_id, num_running_requests)

    def free_block_hashes(self, request: Request) -> None:
        """Discard the block hashes for the request.

        NOTE: Unlike `free`, this method should be called only when the request
        is finished, not when it is preempted.
        """
        self.req_to_block_hashes.pop(request.request_id, None)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        return self.block_pool.take_events()

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        return KVCacheBlocks(
            self.coordinator.get_blocks(request_id)).get_block_ids()

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled."""
        if self.enable_caching:
            block_hashes = self.req_to_block_hashes[request.request_id]
            self.coordinator.cache_blocks(request, block_hashes,
                                          num_computed_tokens)

    def create_empty_block_list(self) -> KVCacheBlocks:
        """Creates a new KVCacheBlocks instance with no blocks."""
        """创建一个空的 KVCacheBlocks, 并没有实际的块"""
        return KVCacheBlocks(tuple([]
                                   for _ in range(self.num_kv_cache_groups)))

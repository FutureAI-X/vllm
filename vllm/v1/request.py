# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from vllm.multimodal.inputs import MultiModalKwargsItem, PlaceholderRange
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import is_list_of
from vllm.v1.engine import (EngineCoreEvent, EngineCoreEventType,
                            EngineCoreRequest, FinishReason)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash


class Request:
    """由 Scheduler 调度的请求"""

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        multi_modal_kwargs: Optional[list[MultiModalKwargsItem]],
        multi_modal_hashes: Optional[list[str]],
        multi_modal_placeholders: Optional[list[PlaceholderRange]],
        sampling_params: Optional[SamplingParams],
        pooling_params: Optional[PoolingParams],
        eos_token_id: Optional[int],
        client_index: int = 0,
        arrival_time: Optional[float] = None,
        lora_request: Optional["LoRARequest"] = None,
        structured_output_request: Optional["StructuredOutputRequest"] = None,
        cache_salt: Optional[str] = None,
        priority: int = 0,
        block_hasher: Optional[Callable[["Request"],
                                        list["BlockHash"]]] = None,
    ) -> None:
        """
            request_id: str：请求的唯一标识符
            prompt_token_ids: list[int]：提示词的token ID列表
            multi_modal_inputs: Optional[list[MultiModalKwargs]]：多模态输入数据
            multi_modal_hashes: Optional[list[str]]：多模态输入的哈希值
            multi_modal_placeholders: Optional[list[PlaceholderRange]]：多模态占位符范围
            sampling_params: Optional[SamplingParams]：采样参数
            pooling_params: Optional[PoolingParams]：池化参数
            eos_token_id: Optional[int]：结束符token ID
            client_index: int = 0：客户端索引，默认为0
            arrival_time: Optional[float] = None：请求到达时间，默认为当前时间
            lora_request: Optional["LoRARequest"]：LoRA请求参数
            structured_output_request: Optional["StructuredOutputRequest"]：结构化输出请求
            cache_salt: Optional[str] = None：缓存盐值
            priority: int = 0：请求优先级，默认为0
        """
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
        self.structured_output_request = structured_output_request
        self.arrival_time = arrival_time if arrival_time is not None else \
            time.time()

        self.status = RequestStatus.WAITING
        self.use_structured_output = False
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: Union[int, str, None] = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: Optional[dict[str, Any]] = None

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if sampling_params.guided_decoding is not None:
                self.status = RequestStatus.WAITING_FOR_FSM
                self.use_structured_output = True

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = \
                    sampling_params.extra_args.get("kv_transfer_params")
        else:
            raise ValueError(
                "sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = self.prompt_token_ids.copy()
        self.num_output_placeholders = 0  # Used in async scheduling.
        # 投机解码 token
        self.spec_token_ids: list[int] = []
        """
        已计算的 token 数, 表示模型已经实际处理并生成输出的 token 数量 = prompt tokens + output tokens
        
        主要用途:
            调度管理：帮助调度器了解请求的处理进度
            缓存管理：确定哪些tokens的结果已经被缓存
            抢占恢复：在请求被抢占后恢复时，知道从哪里继续计算
            性能优化：避免重复计算已经处理过的tokens
        """
        self.num_computed_tokens = 0
        self.cache_salt: Optional[str] = cache_salt

        # Multi-modal related
        self.mm_positions = multi_modal_placeholders or []
        self.mm_kwargs = multi_modal_kwargs or []
        self.mm_hashes: list[str] = multi_modal_hashes or []
        self.num_encoder_inputs = len(self.mm_kwargs)
        self.has_encoder_inputs = self.num_encoder_inputs > 0

        # Sanity check
        assert len(self.mm_kwargs) == len(self.mm_positions)
        if self.mm_hashes:
            assert len(self.mm_kwargs) == len(self.mm_hashes)

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)

        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        """
        是一个计数器，用于记录在模型推理过程中 logits（模型输出的原始分数）中出现 NaN（Not a Number）值的数量
        
        背景知识
            Logits：神经网络最后一层的原始输出，通常是未归一化的分数
            NaN：在浮点运算中表示无效或未定义的结果（如 0/0 或 ∞-∞）
        出现原因：
            数值不稳定（如梯度爆炸）
            模型训练或推理中的异常情况
            硬件或计算错误
        用途
            质量控制：检测模型输出是否可靠
            错误处理：当输出损坏时可以采取相应措施（如终止请求）
            调试诊断：帮助识别模型推理中的问题
        """
        self.num_nans_in_logits = 0

        self.block_hashes: list[BlockHash] = []
        self.get_hash_new_full_blocks: Optional[Callable[
            [], list[BlockHash]]] = None
        if block_hasher is not None:
            self.get_hash_new_full_blocks = partial(block_hasher, self)
            self.block_hashes = self.get_hash_new_full_blocks()

    @classmethod
    def from_engine_core_request(
        cls, request: EngineCoreRequest,
        block_hasher: Optional[Callable[["Request"], list["BlockHash"]]]
    ) -> "Request":
        if request.mm_kwargs is not None:
            mm_kwargs_lst = list(request.mm_kwargs)
            assert is_list_of(mm_kwargs_lst, MultiModalKwargsItem), (
                "mm_kwargs was not updated in EngineCore.add_request")
        else:
            mm_kwargs_lst = None

        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            multi_modal_kwargs=mm_kwargs_lst,
            multi_modal_hashes=request.mm_hashes,
            multi_modal_placeholders=request.mm_placeholders,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            structured_output_request=StructuredOutputRequest(
                sampling_params=request.sampling_params) \
                    if request.sampling_params else None,
            cache_salt=request.cache_salt,
            priority=request.priority,
            block_hasher=block_hasher,
        )

    def append_output_token_ids(
        self,
        token_ids: Union[int, list[int]],
    ) -> None:
        """添加(记录)新生成的 token"""
        # 单个 token
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        # 多个 token
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        if self.get_hash_new_full_blocks is not None:
            self.block_hashes.extend(self.get_hash_new_full_blocks())

    @property
    def is_output_corrupted(self) -> bool:
        """检查模型输出是否已损坏"""
        return self.num_nans_in_logits > 0

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> Union[FinishReason, None]:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_tokens(self, input_id: int) -> int:
        assert input_id < len(self.mm_positions)
        num_tokens = self.mm_positions[input_id].length
        return num_tokens

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: Optional[float] = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> Optional[list[EngineCoreEvent]]:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events


class RequestStatus(enum.IntEnum):
    """Status of a request."""
    """
    enum.auto() 是 Python 标准库 enum 模块中的一个函数，用于自动为枚举成员分配值，每个枚举成员会自动获得一个递增的整数值，从 1 开始。
    """
    # 等待处理
    WAITING = enum.auto()
    # 等待FSM处理
    WAITING_FOR_FSM = enum.auto()
    # 等待远程KV处理
    WAITING_FOR_REMOTE_KVS = enum.auto()
    # 运行中
    RUNNING = enum.auto()
    # 被抢占/暂停
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    # 请求正常完成
    FINISHED_STOPPED = enum.auto()
    # 请求因达到最大长度限制而完成
    FINISHED_LENGTH_CAPPED = enum.auto()
    # 请求被中止
    FINISHED_ABORTED = enum.auto()
    # 请求被忽略（如提示过长）
    FINISHED_IGNORED = enum.auto()

    def __str__(self):
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(
            status: "RequestStatus") -> Union[FinishReason, None]:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
}

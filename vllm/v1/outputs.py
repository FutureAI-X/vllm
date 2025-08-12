# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch


class LogprobsLists(NamedTuple):

    """
    二维列表，
    - 第1维: 请求
    - 第2维： 概率最高的n个token Id
    """
    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: list[list[int]]

    """
    二维列表
    - 第1维: 请求
    - 第2维: 概率最高的n个 token Id 对应的概率值
    """
    # [num_reqs, max_num_logprobs + 1]
    logprobs: list[list[float]]

    """
    存储每个请求的采样 token 的 rank
    """
    # [num_reqs]
    sampled_token_ranks: list[int]

    def slice(self, start: int, end: int):
        return LogprobsLists(
            self.logprob_token_ids[start:end],
            self.logprobs[start:end],
            self.sampled_token_ranks[start:end],
        )


class LogprobsTensors(NamedTuple):
    """用于存储和传递与 logprobs 相关的张量数据。

    概念介绍:
    Logprobs(对数概率) 是概率值的自然对数表示形式，即 logprobs = ln(probability)

    概率越高, logprob 值越接近 0; 概率越低, logprob 值越负

    为什么使用 Logprobs ?
    1. 数值稳定性: 概率值通常很小（如 0.001），直接计算可能导致数值下溢。使用对数可以避免这个问题。
    2. 计算效率: 对数空间中的加法比原始概率空间中的乘法更高效。
    3. 直观比较: 在对数空间中，更容易比较不同概率的相对大小。
    """

    """
    Token ID 张量, shape 为 [num_reqs, max_num_logprobs + 1]
        num_reqs 表示请求数量
        max_num_logprobs 表示每个请求的 logprobs 数目
    第一列通常是实际采样的 token ID，后续列是概率最高的 k 个 token ID
    """
    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: torch.Tensor

    """
    logprobs 张量, 包含对应 token 的概率值, shape 为 [num_reqs, max_num_logprobs + 1]
    """
    # [num_reqs, max_num_logprobs + 1]
    logprobs: torch.Tensor

    # 表示采样 token 在所有可能 token 中按概率排序的位置, 0表示该 token 是概率最高的(排名第1)
    # [num_reqs]
    selected_token_ranks: torch.Tensor

    def tolists(self):
        """将 GPU 张量转换为 CPU 上的列表形式"""
        return LogprobsLists(
            self.logprob_token_ids.tolist(),
            self.logprobs.tolist(),
            self.selected_token_ranks.tolist(),
        )

    @staticmethod
    def empty_cpu(num_positions: int,
                  num_tokens_per_position: int) -> "LogprobsTensors":
        """Create empty LogprobsTensors on CPU."""

        logprob_token_ids = torch.empty(
            (num_positions, num_tokens_per_position),
            dtype=torch.int32,
            device="cpu")
        logprobs = torch.empty_like(logprob_token_ids, dtype=torch.float32)
        selected_token_ranks = torch.empty(num_positions,
                                           dtype=torch.int32,
                                           device="cpu")
        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids,
            logprobs=logprobs,
            selected_token_ranks=selected_token_ranks,
        )


@dataclass
class SamplerOutput:
    """用于封装采样器的输出结果
    
    使用 @dataclass 装饰器，自动为类生成 __init__、__repr__ 等方法。

    示例输出：
    假设有一个包含 3 个请求的批次，每个请求生成 1 个 token
    SamplerOutput(
        sampled_token_ids=torch.tensor([
            [1234],  # 请求1生成的token ID
            [5678],  # 请求2生成的token ID
            [9012]   # 请求3生成的token ID
        ]),
        logprobs_tensors=LogprobsTensors(
            logprob_token_ids=torch.tensor([
                [1234, 100, 200],  # 请求1: 采样token + top-2 tokens
                [5678, 150, 250],  # 请求2: 采样token + top-2 tokens
                [9012, 180, 280]   # 请求3: 采样token + top-2 tokens
            ]),
            logprobs=torch.tensor([
                [-0.1, -2.3, -3.1],  # 请求1的logprobs
                [-0.2, -2.5, -3.2],  # 请求2的logprobs
                [-0.3, -2.7, -3.3]   # 请求3的logprobs
            ]),
            selected_token_ranks=torch.tensor([0, 0, 0])  # 采样token的排名(都是第1个)
        )
    )
    """

    # [num_reqs, max_num_generated_tokens]
    # Different requests can have different number of generated tokens.
    # All requests are padded to max_num_generated_tokens.
    # PLACEHOLDER_TOKEN_ID (-1 by default) is used for padding.
    """
    采样得到的 tokenId 张量. shape 为 [num_reqs, max_num_generated_tokens]
        num_reqs: 请求的数量
        max_num_generated_tokens: 最大生成的 token 数量
    不同请求可能生成不同数量的 token，所有请求使用占位符进行填充, 默认使用 PLACEHOLDER_TOKEN_ID (-1) 作为填充值
    """
    sampled_token_ids: torch.Tensor
    # logprobs 信息
    logprobs_tensors: Optional[LogprobsTensors]


@dataclass
class KVConnectorOutput:
    # [req_ids]
    finished_sending: Optional[set[str]] = None
    finished_recving: Optional[set[str]] = None


# ModelRunnerOutput is serialized and sent to the scheduler process.
# This is expensive for torch.Tensor so prefer to use list instead.
@dataclass
class ModelRunnerOutput:

    """request_id 列表，长度为请求个数"""
    # [num_reqs]
    req_ids: list[str]

    """request_id 与 index 映射，其中 index 指该请求在结果中的索引未至"""
    # req_id -> index
    req_id_to_index: dict[str, int]

    """
    二维列表，存储每个 request 生成的 tokens
    - 第1维: 请求维度
    - 第2维: 每个请求生成的 tokens
    """
    # num_reqs x num_generated_tokens
    # num_generated_tokens is the number of tokens
    # generated in the current step. It can be different for
    # each request due to speculative/jump decoding.
    sampled_token_ids: list[list[int]]

    """
    二维列表，存储每个 request 生成的 draft token
    - 第1维: 请求维度
    - 第2维: 投机解码生成的 draft token
    """
    # num_reqs x num_spec_tokens
    spec_token_ids: Optional[list[list[int]]]

    """存储最后一个 sampled token 相关的对数概率信息"""
    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs]
    logprobs: Optional[LogprobsLists]

    # req_id -> (token_ids, logprobs, ranks)
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len]
    prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]]

    """每个请求的池化输出张量"""
    # [num_reqs, hidden_size]
    pooler_output: list[Optional[torch.Tensor]]

    """KV Cache Connector 的输出信息"""
    kv_connector_output: Optional[KVConnectorOutput] = None

    """记录每个请求在 logits 中 NaN 值的数量"""
    # req_id -> num_nans_in_logits
    num_nans_in_logits: Optional[dict[str, int]] = None


EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(req_ids=[],
                                              req_id_to_index={},
                                              sampled_token_ids=[],
                                              spec_token_ids=None,
                                              logprobs=None,
                                              prompt_logprobs_dict={},
                                              pooler_output=[],
                                              num_nans_in_logits=None)

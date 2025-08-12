# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Optional

import torch

from vllm.v1.sample.logits_processor import LogitsProcessorManager

@dataclass
class SamplingMetadata:
    """用于存储和传输采样过程所需的所有元数据信息，包括采样参数、惩罚设置、词汇限制等各种配置信息。

    使用 @dataclass 装饰器，自动为类生成 __init__、__repr__ 等方法。
    """

    # 采样参数
    temperature: Optional[torch.Tensor]                 # 采样温度，控制采样随机性。值越小输出越确定，越大输出越随机。
    all_greedy: bool                                    # 是否所有请求都使用贪心采样（选择概率最高的 token）
    all_random: bool                                    # 是否所有请求都使用随机采样（从概率分布中随机选择一个 token）

    top_p: Optional[torch.Tensor]                       # 保留累积概率达到 top_p 的最小 token 集合
    top_k: Optional[torch.Tensor]                       # 只考虑概率最高的 k 个 token

    # 随即数生成器: 每个请求对应的随机数生成器，用于保证采样的可重现性
    generators: dict[int, torch.Generator]

    # None means no logprobs, 0 means sampled token logprobs only
    # 需要返回的 logprobs 的数量: None 表示不需要 logprobs，0 表示只返回采样的 token 的 logprobs，不为0则返回top-k个token的概率
    max_num_logprobs: Optional[int]

    # 惩罚机制
    no_penalties: bool                                  # 是否不使用惩罚机制
    prompt_token_ids: Optional[torch.Tensor]            # 提示词的 token id，用于计算惩罚
    frequency_penalties: torch.Tensor                   # 频率惩罚参数, 根据 token 出现频率进行惩罚
    presence_penalties: torch.Tensor                    # 存在惩罚参数，对已经出现过的 token 进行惩罚
    repetition_penalties: torch.Tensor                  # 重复惩罚参数，减少重复 token 的出现

    output_token_ids: list[list[int]]                   # 每个请求已经生成的 token Id 列表

    # 词汇限制
    # `allowed_token_ids_mask` is a 2D bool tensor of shape (max batch size,
    # vocab size).
    allowed_token_ids_mask: Optional[torch.Tensor]      # 允许的 token Id 掩码，形状为 (max batch size, vocab size)，用于限制可生成的词汇。

    # req_index -> bad_words_token_ids
    bad_words_token_ids: dict[int, list[list[int]]]     # 不良词汇的 token Id, 用于屏蔽不良词汇

    # Loaded logits processors. logits 处理器, 用于管理和应用各种 logits 处理逻辑
    logitsprocs: LogitsProcessorManager

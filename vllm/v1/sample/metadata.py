# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Optional

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors


@dataclass
class SamplingMetadata:

    temperature: Optional[torch.Tensor]
    all_greedy: bool
    all_random: bool

    top_p: Optional[torch.Tensor]
    top_k: Optional[torch.Tensor]

    generators: dict[int, torch.Generator]

    # None means no logprobs, 0 means sampled token logprobs only
    # 需要返回的 logprobs 的数量: None 表示不需要 logprobs，0 表示只返回采样的 token 的 logprobs，不为0则返回top-k个token的概率
    max_num_logprobs: Optional[int]

    no_penalties: bool
    prompt_token_ids: Optional[torch.Tensor]
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

    output_token_ids: list[list[int]]

    # 词汇限制
    # `allowed_token_ids_mask` is a 2D bool tensor of shape (max batch size,
    # vocab size).
    allowed_token_ids_mask: Optional[torch.Tensor]

    # req_index -> bad_words_token_ids
    bad_words_token_ids: dict[int, list[list[int]]]

    # Loaded logits processors
    logitsprocs: LogitsProcessors

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that samples the next tokens from the model's outputs."""

import torch
import torch.nn as nn

from vllm.config import LogprobsMode
from vllm.utils import is_pin_memory_available
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words
from vllm.v1.sample.ops.logprobs import batched_count_greater_than
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):

    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs"):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler()
        self.pin_memory = is_pin_memory_available()
        self.logprobs_mode = logprobs_mode

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """采样的过程的核心方法，负责根据模型输出的 logits 和采样参数生成下一个 token
        
        Args:
            logits (torch.Tensor):
                The logits of the current token. 模型输出的 logits
            sampling_metadata (SamplingMetadata):
                The metadata of the current sampling step. 采样参数
        Returns:
            SamplerOutput:
                The output of the sampler. 采样结果
        """
        # NOTE(woosuk): Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
        # TODO(rob): provide option for logprobs post sampling.
        # See https://vllm-dev.slack.com/archives/C07UUL8E61Z/p1735907856007919 # noqa: E501

        # Step 1: Logprobs 计算 (处理前)
        # 判断是否需要 logprobs
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            # 取对数概率
            if self.logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            # 使用原始 logits
            elif self.logprobs_mode == "raw_logits":
                raw_logprobs = logits.clone()

        # Step2: Logits 处理
        # Use float32 for the logits. 使用 float32 数值类型
        logits = logits.to(torch.float32)
        # Apply allowed token ids. 应用允许的 token Id 限制(屏蔽不允许的 token)
        logits = self.apply_allowed_token_ids(logits, sampling_metadata)
        # Apply bad words exclusion. 排除不良词汇
        logits = self.apply_bad_words(logits, sampling_metadata)

        # Apply logits processors which can impact greedy sampling 应用其他 logits 处理器
        for processor in (sampling_metadata.logitsprocs.non_argmax_invariant):
            logits = processor.apply(logits)

        # Apply penalties (e.g., min_tokens, freq_penalties). 应用惩罚机制
        logits = self.apply_penalties(logits, sampling_metadata)

        # Step3: Logprobs 计算 (处理后)
        # Get the process logprobs or logits.
        if num_logprobs is not None:
            if self.logprobs_mode == "processed_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            elif self.logprobs_mode == "processed_logits":
                raw_logprobs = logits.clone()

        # Step4: 执行采样
        # Sample the next token.
        sampled = self.sample(logits, sampling_metadata)

        # Step5: 采样后处理
        # Convert sampled token ids to int64 (long) type to ensure compatibility
        # with subsequent operations that may use these values as indices.
        # This conversion is necessary because FlashInfer sampling operations
        # return int32 (while PyTorch argmax and topk return int64).
        # 将采样结果转换为 int64 类型，以保持兼容性。
        sampled = sampled.long()

        # Gather the logprobs of the topk and sampled token (if requested).
        # Get logprobs and rank tensors (if requested)
        # 收集 logprobs
        logprobs_tensors = None if num_logprobs is None else \
            self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=sampled)

        # Use int32 to reduce the tensor size. 转换为 int32 以减少空间
        sampled = sampled.to(torch.int32)

        # Step6 结果组装与返回
        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    def apply_temperature(
        self,
        logits: torch.Tensor,
        temp: torch.Tensor,
    ) -> torch.Tensor:
        # Use in-place division to avoid creating a new tensor.
        return logits.div_(temp.unsqueeze(dim=1))

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """贪婪采样
        
        Args:
            logits (torch.Tensor): 模型输出的 logits, shape: [num_tokens, vocab_size] 而不是 [batch_size, seq_len, vocab_size]
        
        Returns:
            torch.Tensor: 采样结果，shape: [num_tokens]

        关于形状的说明：
        vLLM 将所有请求的 token 扁平化处理，而不是保持批次和序列维度。这样做的原因包括：
        1. 性能优化
        - 简化了张量操作
        - 提高了 GPU 并行计算效率
        - 减少了内存碎片
        2. 动态批处理
        vLLM 支持动态批处理，不同请求可能有不同的序列长度，扁平化处理更适合这种场景。
        """
        return logits.argmax(dim=-1).view(-1)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Sample logits based on sampling metadata. 
        根据采样元数据对logits进行采样
        
        The various logits processing functions called in this method
        may update the logits tensor in-place.
        此方法中调用的各种 logits 处理函数可能会在原地更新logits张量

        Args:
            logits (torch.Tensor): The logits tensor. logits张量
            sampling_metadata (SamplingMetadata): The sampling metadata. 采样元数据

        Returns:
            torch.Tensor: sahpe [num_tokens]
        """

        # Step1 确保不会同时启用全贪婪采样和全随即采样
        assert not (sampling_metadata.all_greedy
                    and sampling_metadata.all_random)
        
        # Step2 贪婪采样处理
        if sampling_metadata.all_random:                    # 如果启用了全随机采样，跳过贪婪采样
            greedy_sampled = None
        else:                                               # 否则进行贪婪采样
            greedy_sampled = self.greedy_sample(logits)     # 执行贪婪采样(选择概率最高的 token)
            if sampling_metadata.all_greedy:                # 如果所有请求都使用贪心采样，直接返回贪心采样结果，无需执行后续逻辑
                return greedy_sampled

        # Step3 温度处理
        assert sampling_metadata.temperature is not None    # 温度不能为 None

        # Apply temperature. 使用温度参数调整 logits 分布, 控制采样的随机性
        logits = self.apply_temperature(logits, sampling_metadata.temperature)

        # Step4 应用 argmax 不变的 logits 处理器(只会对随机采样有影响)
        # Apply logits processors that only apply to random sampling
        # (argmax invariant)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        # Step5 执行基于 top_k 和 top_p 的随机采样
        # Apply top_k and/or top_p.
        random_sampled = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        # Step6 结果合并
        # 如果没有贪婪采样结果，则直接返回随机采样结果
        if greedy_sampled is None:
            return random_sampled

        # 根据温度值决定使用贪婪采样结果还是随机采样结果
        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled

    def compute_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        """将模型输出输出的 logits 转换为对数概率(logprobs)
        
        Args:
            logits (torch.Tensor): 模型输出的 logits, 形状为 [batch_size, seq_len, vocab_size]

        处理逻辑:
            在最后一个维度上先进行 softmax, 再取对数
        """
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    def gather_logprobs(
        self,
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: minimum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == torch.int64
        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs,
                                                 num_logprobs,
                                                 dim=-1)

        # Get with the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute the ranks of the actual token.
        token_ranks = batched_count_greater_than(logprobs, token_logprobs)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)

    def apply_penalties(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if not sampling_metadata.no_penalties:
            assert sampling_metadata.prompt_token_ids is not None
            logits = apply_all_penalties(
                logits,
                sampling_metadata.prompt_token_ids,
                sampling_metadata.presence_penalties,
                sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties,
                sampling_metadata.output_token_ids,
            )
        return logits

    def apply_allowed_token_ids(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask,
                                float("-inf"))
        return logits

    def apply_bad_words(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.bad_words_token_ids:
            apply_bad_words(
                logits,
                sampling_metadata.bad_words_token_ids,
                sampling_metadata.output_token_ids,
            )
        return logits

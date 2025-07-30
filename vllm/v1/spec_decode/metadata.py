# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SpecDecodeMetadata:
    """存储推测解码过程中所需的各种元数据信息"""

    # [num_tokens]  所有 draft token id, 将批次中所有序列的草稿令牌展平存储
    draft_token_ids: torch.Tensor
    # [batch_size]  每个批次中 draft token 的数量
    num_draft_tokens: list[int]
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor
    # [num_tokens]  目标模型 logits 在输出中的索引位置
    target_logits_indices: torch.Tensor
    # [batch_size]  奖励令牌的 logits 索引
    bonus_logits_indices: torch.Tensor
    # [num_tokens + batch_size] 综合索引，包含所有需要的 logits 位置信息
    logits_indices: torch.Tensor

    def __post_init__(self):
        self.max_spec_len = max(self.num_draft_tokens)

    @classmethod
    def make_dummy(
        cls,
        draft_token_ids: list[list[int]],
        device: torch.device,
    ) -> "SpecDecodeMetadata":
        """创建一个虚拟的推测解码元数据对象，这个方法主要用于测试、初始化或在没有真实数据时创建占位符对象
        
        Args:
            draft_token_ids (list[list[int]]): 草稿令牌ID列表
                外层数组: 批次数量/序列数量
                内层数组: 每个序列的草稿令牌ID
                示例输入：[[10, 20, 30], [40, 50], [60]]
            device (torch.device): 设备

        示例输入(draft_token_ids):
            [
                [10, 20, 30],
                [40, 50],
                [60]
            ]
        """
        # Step1 计算基本维度信息
        # 批次大小, 等于输入序列的数量, 示例: 3
        batch_size = len(draft_token_ids)
        # 每个序列中草稿令牌数量, 示例: [3, 2, 1]
        num_draft_tokens = [len(ids) for ids in draft_token_ids]
        # 将所有草稿令牌ID展开成1维向量, 示例: [10, 20, 30, 40, 50, 60]
        flattened_draft_token_ids = sum(draft_token_ids, [])
        # 总的草稿令牌数量, 示例：6
        num_tokens = len(flattened_draft_token_ids)

        # Step2 创建张量
        # 将展平的草稿令牌ID转换成张量, 示例: tensor([10, 20, 30, 40, 50, 60])
        draft_token_ids_tensor = torch.tensor(flattened_draft_token_ids,
                                              dtype=torch.int32,
                                              device=device)
        # 计算累积和并转换为张量, 示例: [3, 5, 6]
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        cu_num_draft_tokens_tensor = torch.from_numpy(cu_num_draft_tokens).to(
            device)

        # 目标 logits 索引
        target_logits_indices = torch.zeros(num_tokens,
                                            dtype=torch.int32,
                                            device=device)
        # 奖励 logits 索引
        bonus_logits_indices = torch.zeros(batch_size,
                                           dtype=torch.int32,
                                           device=device)
        # 总 logits 索引
        logits_indices = torch.zeros(num_tokens + batch_size,
                                     dtype=torch.int32,
                                     device=device)
        return cls(
            draft_token_ids=draft_token_ids_tensor,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens_tensor,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

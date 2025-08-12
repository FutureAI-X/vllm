# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

from vllm.v1.request import Request, RequestStatus


def check_stop(request: Request,
               max_model_len: int,
               pooler_output: Optional[torch.Tensor] = None) -> bool:
    """检查请求是否应该 stop"""
    # 1. 如果总长度达到 max_model_len 或 生成的 token 数达到 max_tokens 则 stop
    if (request.num_tokens >= max_model_len
            or request.num_output_tokens >= request.max_tokens):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    # 2. 池化模型相关判断
    if request.pooling_params:
        if pooler_output is not None:
            request.status = RequestStatus.FINISHED_STOPPED
            return True
        return False

    # 3. 根据采样参数判断
    sampling_params = request.sampling_params
    assert sampling_params is not None
    last_token_id = request.output_token_ids[-1]
    # (1) 如果 last_token_id 是 eos_token_id 则 stop
    if (not sampling_params.ignore_eos
            and last_token_id == request.eos_token_id):
        request.status = RequestStatus.FINISHED_STOPPED
        return True
    # (2) 如果 last_token_id 是 stop_token_ids 中的一个则 stop
    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True

    # 4. 默认不终止
    return False

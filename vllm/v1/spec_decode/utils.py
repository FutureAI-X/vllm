# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.sampling_params import SamplingParams

_SAMPLING_EPS = 1e-5


def is_spec_decode_unsupported(sampling_params: SamplingParams) -> bool:
    """True if request is incompatible with speculative decoding"""
    """
    基于采样参数判断是否不支持投机解码
    
    满足以下任一条件时，认为不支持投机解码
        - 频率惩罚（frequency_penalty）不为0.0
        - 存在惩罚（presence_penalty）不为0.0
        - 重复惩罚（repetition_penalty）不等于1.0
        - 最小p值（min_p）大于采样阈值
        - 需要输出对数概率（logprobs）
    
    Returns
        True: 不支持投机解码
        False: 支持投机解码
    """
    return (sampling_params.frequency_penalty != 0.0
            or sampling_params.presence_penalty != 0.0
            or sampling_params.repetition_penalty != 1.0
            or sampling_params.min_p > _SAMPLING_EPS
            or sampling_params.logprobs is not None)

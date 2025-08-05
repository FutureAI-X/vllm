# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from array import array
from typing import Any, Type

from vllm.sequence import VLLM_TOKEN_ID_ARRAY_TYPE

"""这些钩子函数主要用于在 vLLM 的分布式执行环境中序列化和反序列化包含 token ID 的数组数据，确保在不同进程或节点之间高效传输序列数据。

目前主要是在 Ray Distributed Executor 中使用
"""
def encode_hook(obj: Any) -> Any:
    """Custom msgspec enc hook that supports array types.

    See https://jcristharif.com/msgspec/api.html#msgspec.msgpack.Encoder
    """
    """自定义 msgspec enc hook: 将 Python array 对象编码为字节串 (msgspec 原生不支持 Python array)
    
    示例输入:
        obj = array('l', [1, 2, 3, 4, 5])
        
    示例输出:
        b'\x01\x00\x00\x00'
    """
    if isinstance(obj, array):
        assert obj.typecode == VLLM_TOKEN_ID_ARRAY_TYPE, (
            f"vLLM array type should use '{VLLM_TOKEN_ID_ARRAY_TYPE}' type. "
            f"Given array has a type code of {obj.typecode}.")
        return obj.tobytes()


def decode_hook(type: Type, obj: Any) -> Any:
    """Custom msgspec dec hook that supports array types.

    See https://jcristharif.com/msgspec/api.html#msgspec.msgpack.Encoder
    """
    """自定义 msgspec dec hook: 将字节串解码回 Python array 对象, """
    if type is array:
        deserialized = array(VLLM_TOKEN_ID_ARRAY_TYPE)
        deserialized.frombytes(obj)
        return deserialized

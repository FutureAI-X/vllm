# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from abc import ABC, abstractmethod
from functools import cached_property
from typing import (Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple,
                    Union)

import torch.nn as nn
from typing_extensions import TypeVar

import vllm.platforms
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, PoolerOutput
from vllm.tasks import SupportedTask
from vllm.utils import make_async
from vllm.worker.worker_base import WorkerBase

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)


class ExecutorBase(ABC):
    """Base class for all executors.

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor 
    that can execute the model on multiple devices.
    """
    """所有执行器的基类
    1. 执行器负责在一个设备上执行模型
    2. 分布式执行器可以在多个设备上执行模型
    """

    # 是否使用 Ray 进行编排
    uses_ray: bool  # whether the executor uses Ray for orchestration.

    # 是否支持流水线并行（Pipeline Parallelism, PP）, 流水线并行是一种将模型按层分割到不同设备上的并行方式
    supports_pp: bool = False  # whether the executor supports PP

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self._init_executor()
        self.is_sleeping = False
        self.sleeping_tags: set[str] = set()

    @abstractmethod
    def _init_executor(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def collective_rpc(self,
                       method: Union[str, Callable[..., _R]],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict[str, Any]] = None) -> List[_R]:
        """
        Execute an RPC call on all workers.

        Args:
            method: Name of the worker method to execute, or a callable that
                is serialized and sent to all workers to execute.

                If the method is a callable, it should accept an additional
                `self` argument, in addition to the arguments passed in `args`
                and `kwargs`. The `self` argument will be the worker object.
            timeout: Maximum time in seconds to wait for execution. Raises a
                [`TimeoutError`][] on timeout. `None` means wait indefinitely.
            args: Positional arguments to pass to the worker method.
            kwargs: Keyword arguments to pass to the worker method.

        Returns:
            A list containing the results from each worker.
        
        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """
        """
        在全部 workers 上执行一次 RPC 调用
        
        Args:
            method: 要执行的 worker method
            timeout: 等待执行完成的最长时间（秒）。如果超时，则会引发 [`TimeoutError`][]。`None` 表示无限等待
            args: 传递给 worker method 的位置参数
            kwargs: 传递给 worker method 的关键字参数
            
        Returns:
            一个列表，包含每个 worker 返回的结果
            
        Note:
            建议仅使用此API传递控制消息，并设置 data-plane communication 来传输数据
        """
        raise NotImplementedError

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available blocks for the GPU KV cache and
        swappable CPU KV cache.

        Normally, this should simply delegate to the underlying Worker. Some
        ExecutorBase may require modification of the result, e.g. to ensure the
        selected cache sizes are compatible with all workers.

        Returns a Tuple[num_gpu_blocks, num_cpu_blocks], where num_gpu_blocks
        are blocks that are "active" on the device and can be appended to.
        num_cpu_blocks refers to "swapped" blocks in CPU memory and cannot be
        appended to.
        """
        """
        确定 GPU KV cache 以及 swappable CPU KV cache 的可用 blocks 数量
        
        通常，此操作应直接委托给底层的 Worker。某些 ExecutorBase 可能需要对结果进行调整，
        例如，确保所选择的缓存大小与所有工作节点兼容。
        
        Returns:
            num_gpu_blocks  可用的 GPU blocks 数量
            num_cpu_blocks  可用的 CPU blocks 数量
        """
        # Step1 所有 worker 返回的 list 集合
        results = self.collective_rpc("determine_num_available_blocks")

        # Step2 获取所有 worker 中的最小值
        a = min([r[0] for r in results])
        b = min([r[1] for r in results])

        # Step3 返回
        return a, b

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        """通过调用底层 Worker 来初始化 KV cache"""
        # Step1 日志记录
        # NOTE: This is logged in the executor because there can be >1 workers.
        logger.info("# %s blocks: %d, # CPU blocks: %d",
                    vllm.platforms.current_platform.device_name,
                    num_gpu_blocks, num_cpu_blocks)
        max_concurrency = (num_gpu_blocks * self.cache_config.block_size /
                           self.model_config.max_model_len)
        logger.info("Maximum concurrency for %s tokens per request: %.2fx",
                    self.model_config.max_model_len, max_concurrency)

        # Step2 配置更新
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Step3 远程调用所有 worker 进行 KV cache 初始化
        self.collective_rpc("initialize_cache",
                            args=(num_gpu_blocks, num_cpu_blocks))

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        """
        Run a function directly on the model inside each worker,
        returning the result for each of them.
        """

        def rpc_func(worker: WorkerBase) -> _R:
            return func(worker.get_model())

        return self.collective_rpc(rpc_func)

    @cached_property  # Avoid unnecessary RPC calls
    def supported_tasks(self) -> tuple[SupportedTask, ...]:
        output = self.collective_rpc("get_supported_tasks")
        return output[0]

    def execute_model(
        self, execute_model_req: ExecuteModelRequest
    ) -> Optional[List[Union[SamplerOutput, PoolerOutput]]]:
        """执行模型调用"""
        # Step1 远程调用所有 worker 进行模型调用
        output = self.collective_rpc("execute_model",
                                     args=(execute_model_req, ))

        # Step2 返回第一个 worker 的输出 (为什么是第1个worker?)
        return output[0]

    def stop_remote_worker_execution_loop(self) -> None:
        """Releases parallel workers from model loop."""
        return

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("add_lora", args=(lora_request, )))

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("remove_lora", args=(lora_id, )))

    def pin_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("pin_lora", args=(lora_id, )))

    def list_loras(self) -> Set[int]:
        sets = self.collective_rpc("list_loras")
        for s in sets:
            assert s == sets[0], "All workers should have the same LORAs."
        return sets[0]

    def start_profile(self) -> None:
        self.collective_rpc("start_profile")

    def stop_profile(self) -> None:
        self.collective_rpc("stop_profile")

    def sleep(self, level: int = 1):
        if self.is_sleeping:
            logger.warning("Executor is already sleeping.")
            return
        time_before_sleep = time.perf_counter()
        self.collective_rpc("sleep", kwargs=dict(level=level))
        time_after_sleep = time.perf_counter()
        self.sleeping_tags = {"weights", "kv_cache"}
        self.is_sleeping = True
        logger.info("It took %.6f seconds to fall asleep.",
                    time_after_sleep - time_before_sleep)

    def wake_up(self, tags: Optional[list[str]] = None):
        if not self.is_sleeping:
            logger.warning("Executor is not sleeping.")
            return
        if tags:
            for tag in tags:
                if tag not in self.sleeping_tags:
                    logger.warning("Tag %s is not in sleeping tags %s", tag,
                                   self.sleeping_tags)
                    return
        time_before_wakeup = time.perf_counter()
        self.collective_rpc("wake_up", kwargs=dict(tags=tags))
        time_after_wakeup = time.perf_counter()
        logger.info("It took %.6f seconds to wake up tags %s.",
                    time_after_wakeup - time_before_wakeup,
                    tags if tags is not None else self.sleeping_tags)
        if tags:
            for tag in tags:
                self.sleeping_tags.remove(tag)
        else:
            self.sleeping_tags.clear()
        if not self.sleeping_tags:
            self.is_sleeping = False

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.collective_rpc("save_sharded_state",
                            kwargs=dict(path=path,
                                        pattern=pattern,
                                        max_size=max_size))

    @abstractmethod
    def check_health(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Shutdown the executor."""
        return

    def __del__(self):
        self.shutdown()

    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        """Executes one model step on the given sequences."""
        output = await make_async(self.execute_model)(execute_model_req)
        return output

    async def stop_remote_worker_execution_loop_async(self) -> None:
        """Releases parallel workers from model loop."""
        return

    async def check_health_async(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        self.check_health()


class DistributedExecutorBase(ExecutorBase):
    """Abstract superclass of distributed executor implementations."""
    """分布式执行器的抽象父类"""

    def __init__(self, *args, **kwargs):
        # This is non-None when the execute model loop is running
        # in the parallel workers. It's a coroutine in the AsyncLLMEngine case.
        # 在并行工作程序中运行时，此为非 None。当使用 AsyncLLMEngine 时，它是一个协程。
        self.parallel_worker_tasks: Optional[Union[Any, Awaitable[Any]]] = None

        super().__init__(*args, **kwargs)

    def execute_model(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[SamplerOutput]:
        # Step1 启动 worker execution loop
        # TODO: unify into collective_rpc
        if self.parallel_worker_tasks is None:
            self.parallel_worker_tasks = self._run_workers(
                "start_worker_execution_loop",
                async_run_tensor_parallel_workers_only=True)

        # Step2 在 driver worker 节点运行 execute_model 方法
        # Only the driver worker returns the sampling results.
        driver_outputs = self._driver_execute_model(execute_model_req)

        # Step3 校验执行结果并返回
        assert driver_outputs is not None
        return driver_outputs

    def stop_remote_worker_execution_loop(self) -> None:
        if self.parallel_worker_tasks is None:
            return

        self._driver_execute_model(execute_model_req=None)
        parallel_worker_tasks = self.parallel_worker_tasks
        self.parallel_worker_tasks = None
        # Ensure that workers exit model loop cleanly
        # (this will raise otherwise)
        self._wait_for_tasks_completion(parallel_worker_tasks)

    @abstractmethod
    def _driver_execute_model(
        self, execute_model_req: Optional[ExecuteModelRequest]
    ) -> Optional[List[SamplerOutput]]:
        """Run execute_model in the driver worker.

        Passing None will cause the driver to stop the model execution loop
        running in each of the remote workers. In this case, this method
        returns None. Otherwise, this method returns the model output.
        """
        """
        在 driver worker 上执行 execute_model
        
        传递 None 将导致驱动程序停止在各个 remote worker 上运行的模型执行循环。在这种情况下，该方法返回 None。否则，该方法返回模型输出。
        """
        raise NotImplementedError

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict] = None) -> List[Any]:
        return self._run_workers(method, *args, **(kwargs or {}))

    @abstractmethod
    def _run_workers(
        self,
        method: Union[str, Callable],
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers.

        Args:
            async_run_tensor_parallel_workers_only: If True the method will be
                run only in the remote TP workers, not the driver worker.
                It will also be run asynchronously and return a list of futures
                rather than blocking on the results.
        
        # TODO: simplify and merge with collective_rpc
        """
        """
        在所有的 workers 上执行给定的方法
        
        Args:
            async_run_tensor_parallel_workers_only: 如果为 True, 则该方法将仅在远程的 TP workers 节点运行, 而不会在 driver worker
                同时，该方法将以异步方式运行，返回一个 future 列表，而不会阻塞等待结果
        """
        raise NotImplementedError

    @abstractmethod
    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        raise NotImplementedError

    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        if self.parallel_worker_tasks is None:
            # Start model execution loop running in the parallel workers
            self.parallel_worker_tasks = asyncio.create_task(
                self._start_worker_execution_loop())

        # Only the driver worker returns the sampling results.
        return await self._driver_execute_model_async(execute_model_req)

    async def stop_remote_worker_execution_loop_async(self) -> None:
        if self.parallel_worker_tasks is None:
            return

        await self._driver_execute_model_async()
        parallel_worker_tasks = self.parallel_worker_tasks
        self.parallel_worker_tasks = None
        # Ensure that workers exit model loop cleanly
        # (this will raise otherwise)
        await parallel_worker_tasks

    @abstractmethod
    async def _driver_execute_model_async(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> List[SamplerOutput]:
        """Execute the model asynchronously in the driver worker.

        Passing None will cause the driver to stop the model execution
        loop running in each of the remote workers.
        """
        raise NotImplementedError

    @abstractmethod
    async def _start_worker_execution_loop(self):
        """Run execution loop on all workers. It guarantees all workers run
        the loop or None of them is running the loop. Loop can be stopped by
        `stop_remote_worker_execution_loop`.
        The API is idempotent (guarantee only 1 loop run at any moment)."""
        raise NotImplementedError

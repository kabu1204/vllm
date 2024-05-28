from collections import defaultdict
from itertools import islice, repeat
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Set

import os
import torch

import vllm.envs as envs
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, DeviceConfig
from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import ExecuteModelRequest, SamplerOutput

from vllm.executor.distributed_gpu_executor import (  # yapf: disable
    DistributedGPUExecutor, DistributedGPUExecutorAsync)
from vllm.executor.ray_utils import RayWorkerWrapper, ray
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        get_vllm_instance_id, make_async)

if ray is not None:
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

USE_RAY_COMPILED_DAG = False

class HybridExecutor(DistributedGPUExecutor):
    def _init_executor(self) -> None:
        assert self.parallel_config.distributed_executor_backend == "hybrid"
        assert self.lora_config is None, "HybridExecutor doesn't support LoRA"
        assert self.parallel_config.disable_custom_all_reduce, "HybridExecutor doesn't support custom all-reduce"
        
        # GPU worker args
        self.model_config: ModelConfig
        self.cache_config: CacheConfig
        self.scheduler_config: SchedulerConfig
        self.device_config: DeviceConfig
        
        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
        
        placement_group = self.parallel_config.placement_group
        # Instantiate ray GPU workers and load the model to GPUs.
        self._init_ray_workers(placement_group)

        # Instantiate CPU workers and load the model to CPU.
        # self._init_ray_cpu_workers()

    def _init_ray_workers(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        if self.parallel_config.tensor_parallel_size == 1:
            # For single GPU case, we use a ray worker with constrained memory.
            num_gpus_per_gpu_worker = self.cache_config.gpu_memory_utilization
        else:
            # Otherwise, the ray workers are allocated with a full GPU.
            num_gpus_per_gpu_worker = 1

        # The driver dummy worker does not actually use any resources.
        # It holds the resource for the driver worker.
        self.driver_dummy_worker: Optional[RayWorkerWrapper] = None
        # The remaining workers are the actual ray actors.
        self.gpu_workers: List[RayWorkerWrapper] = []
        self.cpu_workers: List[RayWorkerWrapper] = []
        self.workers: List[RayWorkerWrapper] = []

        if self.parallel_config.ray_workers_use_nsight:
            raise ValueError("HybridExecutor currently doesn't support NVIDIA Nsight system")

        # Create the workers.
        driver_ip = get_ip()
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            if self.speculative_config is not None:
                raise ValueError("HybridExecutor currently doesn't support speculative decoding")
            
            worker_module_name = "vllm.worker.hybrid_worker"
            worker_class_name = "HybridWorker"

            is_cpu_worker = False if bundle.get("GPU", 0) else True
            num_cpus = bundle.get("CPU") if is_cpu_worker else 0
            num_gpus = 0 if is_cpu_worker else num_gpus_per_gpu_worker

            worker = ray.remote(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                **ray_remote_kwargs,
            )(RayWorkerWrapper).remote(
                worker_module_name=worker_module_name,
                worker_class_name=worker_class_name,
                trust_remote_code=self.model_config.trust_remote_code,
            )

            worker_ip = ray.get(worker.get_node_ip.remote())
            print(f"worker_ip: {worker_ip}")
            if worker_ip == driver_ip and self.driver_dummy_worker is None:
                # If the worker is on the same node as the driver, we use it
                # as the resource holder for the driver process.
                self.driver_dummy_worker = worker
                self.driver_worker = RayWorkerWrapper(
                    worker_module_name=worker_module_name,
                    worker_class_name=worker_class_name,
                    trust_remote_code=self.model_config.trust_remote_code,
                )
            else:
                # Else, added to the list of workers.
                self.workers.append(worker)
                if is_cpu_worker:
                    self.cpu_workers.append(worker)
                else:
                    self.gpu_workers.append(worker)

        if self.driver_dummy_worker is None:
            raise ValueError(
                "Ray does not allocate any GPUs on the driver node. Consider "
                "adjusting the Ray placement group or running the driver on a "
                "GPU node.")

        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = self._run_workers("get_node_and_gpu_ids",
                                                    use_dummy_driver=True)
        print(__file__, worker_node_and_gpu_ids)
        node_workers = defaultdict(list)
        node_gpus = defaultdict(list)

        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids):
            node_workers[node_id].append(i)
            node_gpus[node_id].extend(gpu_ids)
        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

        VLLM_INSTANCE_ID = get_vllm_instance_id()

        # Set environment variables for the driver and workers.
        all_args_to_update_environment_variables = [({
            "CUDA_VISIBLE_DEVICES":
            ",".join(map(str, node_gpus[node_id])),
            "VLLM_INSTANCE_ID":
            VLLM_INSTANCE_ID,
            "VLLM_TRACE_FUNCTION":
            str(envs.VLLM_TRACE_FUNCTION),
        }, ) for (node_id, _) in worker_node_and_gpu_ids]
        self._run_workers("update_environment_variables",
                          all_args=all_args_to_update_environment_variables)

        distributed_init_method = get_distributed_init_method(
            driver_ip, get_open_port())

        # Initialize the actual workers inside worker wrapper.
        init_worker_all_kwargs = [
            self._get_worker_kwargs(
                local_rank=node_workers[node_id].index(rank),
                rank=rank,
                distributed_init_method=distributed_init_method,
                is_cpu_worker=(len(gpus) == 0),
            ) for rank, (node_id, gpus) in enumerate(worker_node_and_gpu_ids)
        ]
        self._run_workers("init_worker", all_kwargs=init_worker_all_kwargs)

        self._run_workers("init_device")
        self._run_workers("load_model",
                          max_concurrent_workers=self.parallel_config.
                          max_parallel_loading_workers)
        
        # for debug
        # exit(-1)

    def _get_worker_kwargs(
            self,
            local_rank: int = 0,
            rank: int = 0,
            distributed_init_method: Optional[str] = None,
            is_cpu_worker = False) -> Dict[str, Any]:
        """Return worker init args for a given rank."""
        if distributed_init_method is None:
            distributed_init_method = get_distributed_init_method(
                get_ip(), get_open_port())
        if is_cpu_worker:
            model_config = _verify_and_get_cpu_model_config(self.model_config)
            scheduler_config = _verify_and_get_cpu_scheduler_config(self.scheduler_config)
            cache_config = _verify_and_get_cpu_cache_config(self.cache_config)
            device_config = _verify_and_get_cpu_device_config(self.device_config)
        else:
            model_config = self.model_config
            scheduler_config = self.scheduler_config
            cache_config = self.cache_config
            device_config = self.device_config

        return dict(
            model_config=model_config,
            parallel_config=self.parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            cache_config=cache_config,
            load_config=self.load_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            vision_language_config=self.vision_language_config,
            speculative_config=self.speculative_config,
            is_driver_worker=rank == 0,
            is_cpu_worker=is_cpu_worker,
        )

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        all_outputs = self._run_workers(
            "execute_model",
            driver_kwargs={"execute_model_req": execute_model_req},
            use_ray_compiled_dag=USE_RAY_COMPILED_DAG)
        # Only the driver worker returns the sampling results.
        return all_outputs[0]

    def check_health(self) -> None:
        # CPUExecutor will always be healthy as long as
        # it's running.
        self._check_if_any_actor_is_dead()

    def _check_if_any_actor_is_dead(self):
        if not self.workers:
            return

        dead_actors = []
        for actor in self.workers:
            actor_state = ray.state.actors(actor._ray_actor_id.hex())  # pylint: disable=protected-access
            if actor_state["State"] == "DEAD":
                dead_actors.append(actor)
        if dead_actors:
            raise RuntimeError("At least one Worker is dead. "
                               f"Dead Workers: {dead_actors}. ")

    def _run_workers(
        self,
        method: str,
        *args,
        driver_args: Optional[Tuple[Any, ...]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        all_args: Optional[List[Tuple[Any, ...]]] = None,
        all_kwargs: Optional[List[Dict[str, Any]]] = None,
        use_dummy_driver: bool = False,
        max_concurrent_workers: Optional[int] = None,
        use_ray_compiled_dag: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers. Can be used in the following
        ways:

        - args/kwargs: All workers share the same args/kwargs
        - args/kwargs and driver_args/driver_kwargs: Driver worker has
          different args
        - all_args/all_kwargs: args/kwargs for each worker are specified
          individually
        """

        if max_concurrent_workers:
            raise NotImplementedError(
                "max_concurrent_workers is not supported yet.")

        if driver_args is None:
            driver_args = args if all_args is None else all_args[0]
        if driver_kwargs is None:
            driver_kwargs = kwargs if all_kwargs is None else all_kwargs[0]

        count = len(self.workers)
        all_worker_args = repeat(args, count) if all_args is None \
            else islice(all_args, 1, None)
        all_worker_kwargs = repeat(kwargs, count) if all_kwargs is None \
            else islice(all_kwargs, 1, None)

        if use_ray_compiled_dag:
            raise ValueError("HybridExecutor doesn't support compiled DAG.")
        
        # Start the ray workers first.
        ray_worker_outputs = [
            worker.execute_method.remote(method, *worker_args,
                                            **worker_kwargs)
            for (worker, worker_args, worker_kwargs
                    ) in zip(self.workers, all_worker_args, all_worker_kwargs)
        ]

        # Start the driver worker after all the ray workers.
        if not use_dummy_driver:
            driver_worker_output = self.driver_worker.execute_method(
                method, *driver_args, **driver_kwargs)
        else:
            assert self.driver_dummy_worker is not None
            driver_worker_output = ray.get(
                self.driver_dummy_worker.execute_method.remote(
                    method, *driver_args, **driver_kwargs))
        # Get the results of the ray workers.
        if self.workers:
            ray_worker_outputs = ray.get(ray_worker_outputs)

        return [driver_worker_output] + ray_worker_outputs


def _verify_and_get_cpu_model_config(config: ModelConfig) -> ModelConfig:
    import copy
    config = copy.deepcopy(config)
    if config.dtype == torch.float16:
        logger.warning("float16 is not supported on CPU, casting to bfloat16.")
        config.dtype = torch.bfloat16
    if not config.enforce_eager:
        logger.warning(
            "CUDA graph is not supported on CPU, fallback to the eager "
            "mode.")
        config.enforce_eager = True
    return config


def _verify_and_get_cpu_scheduler_config(
        config: SchedulerConfig) -> SchedulerConfig:
    import copy
    config = copy.deepcopy(config)
    if config.chunked_prefill_enabled:
        logger.warning("Chunked prefill is not supported on CPU, disable it.")
        config.chunked_prefill_enabled = False
    return config


def _verify_and_get_cpu_cache_config(config: CacheConfig) -> CacheConfig:
    import copy
    config = copy.deepcopy(config)
    _GB = 1 << 30
    if config.enable_prefix_caching:
        # TODO: Support prefix caching on CPU.
        logger.warning("Prefix caching is not supported on CPU, disable it.")
        config.enable_prefix_caching = False

    kv_cache_space = envs.VLLM_CPU_KVCACHE_SPACE

    if kv_cache_space >= 0:
        if kv_cache_space == 0:
            config.cpu_kvcache_space_bytes = 4 * _GB  # type: ignore
            logger.warning("Environment variable VLLM_CPU_KVCACHE_SPACE (GB) "
                           "for CPU backend is not set, using 4 by default.")
        else:
            config.cpu_kvcache_space_bytes = kv_cache_space * _GB  # type: ignore
    else:
        raise RuntimeError(
            "Invalid environment variable VLLM_CPU_KVCACHE_SPACE"
            f" {kv_cache_space}, expect a positive integer value.")

    return config

def _verify_and_get_cpu_device_config(config: DeviceConfig) -> DeviceConfig:
    return DeviceConfig("cpu")
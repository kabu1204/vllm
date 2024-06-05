#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <sys/mman.h>
#include "c10/util/intrusive_ptr.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "nccl-cpu.hpp"
#include "torch/csrc/distributed/c10d/Work.hpp"

#define CUDA_CHECK(cmd)                                                      \
  do {                                                                       \
    cudaError_t e = cmd;                                                     \
    if (e != cudaSuccess) {                                                  \
      fprintf(stderr, "Failed: Cuda error %s:%d '%s' at %s:%d\n", __FILE__, __LINE__, \
              cudaGetErrorString(e), __FILE__, __LINE__);                                        \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

#define POSIX_CHECK(cmd)                                                     \
  do {                                                                       \
    int e = cmd;                                                             \
    if (e != 0) {                                                            \
      perror(#cmd);                                                          \
      fprintf(stderr, "Failed: POSIX error %s at %s:%d\n", strerror(e),      \
              __FILE__, __LINE__);                                           \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

namespace c10d {

static inline bool check_device(c10::Device dev1, c10::Device dev2) {
  return dev1.index() == dev2.index() && dev1.type() == dev2.type();
}

static inline int get_cpu_worker_num() {
  static int cpu_worker_num = -1;
  if (cpu_worker_num == -1) {
    const char* cpu_process_num = getenv("VLLM_CPU_WORKER_NUM");
    if (cpu_process_num == NULL) {
      throw std::runtime_error("env[\"VLLM_CPU_WORKER_NUM\"] is not set");
    }
    cpu_worker_num = atoi(cpu_process_num);
  }
  return cpu_worker_num;
}

static inline bool is_cpu_process(int rank, int size) {
  static int cpu_worker_num = -1;
  if (cpu_worker_num == -1) {
    cpu_worker_num = get_cpu_worker_num();
  }
  return rank >= size - cpu_worker_num;
}

bool NcclCPUWork::isCompleted() {
  return true;
  throw std::runtime_error("NcclCPUWork::isCompleted: not supported");
}

bool NcclCPUWork::isSuccess() const {
  return true;
  throw std::runtime_error("NcclCPUWork::isSuccess: not supported");
}

bool NcclCPUWork::wait(std::chrono::milliseconds timeout) {
  // ncclWork_->wait();
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future> NcclCPUWork::getFuture() {
  throw std::runtime_error("NcclCPUWork::getFuture: not supported");
}

#define NCCL_CPU_CPU_WORLD_SIZE 0
#define NCCL_CPU_SHM_SIZE 81920
#define NCCL_CPU_CUDA_ZERO_COPY_BUFFER_SIZE 81920

// If necessary, pass store/rank/size to the ctor and exchange connection
// information here
NcclCPUBackend::NcclCPUBackend(const c10::intrusive_ptr<::c10d::Store>& store,
                               int rank, int size)
    : Backend(rank, size),
    ncclPG(store, rank, size) {
  // 0. Initial check and setup
  if (size >= NCCL_CPU_MAX_WORLD_SIZE) {
    throw std::runtime_error("world size is too big.");
  }
  isCPU = is_cpu_process(rank, size);
  
  // 1. Create host page-locked shared memory
  size_t sharedMemorySize = NCCL_CPU_SHM_SIZE;
  if (rank == 0) {
    shm_unlink("/nccl_cpu_shm");
    shm_unlink("/nccl_cpu_shm");
    shm_unlink("/nccl_cpu_shm");
    shm_unlink("/nccl_cpu_shm");
    SharedMemoryCreate("/nccl_cpu_shm", sizeof(SharedMemory) + sharedMemorySize,
                     &sharedMemoryInfo);
  } else {
    sleep(1);   // a simple way to make sure shm segment is created.
    SharedMemoryOpen("/nccl_cpu_shm", sizeof(SharedMemory) + sharedMemorySize,
                   &sharedMemoryInfo);
  }

  // 1.1 Register host shared memory for CUDA devices
  shm = (SharedMemory*)sharedMemoryInfo.ptr;
  SharedMemoryRegisterPortable(&dptr, shm->data, sharedMemorySize);
  for (int i = 0; i < size; i++) {
    buffers[i] = shm->data + i * sharedMemorySize / size;
    if (i == rank) {
      localBuffer = buffers[i];
    }
  }

  // 1.2 Initialize pthread barrier
  if (rank == 0) {
    pthread_barrierattr_t barrier_attr;
    POSIX_CHECK(pthread_barrierattr_setpshared(&barrier_attr, PTHREAD_PROCESS_SHARED));
    POSIX_CHECK(pthread_barrier_init(&shm->barrier, &barrier_attr, size));
    shm->size = sharedMemorySize;
  } else {
    while(shm->size != sharedMemorySize);
  }

  // 2. Create CUDA IPC shared memory
  if (!isCPU) {
    cudaIpcInitPool(rank, NCCL_CPU_CUDA_ZERO_COPY_BUFFER_SIZE,
                  &localPool, &shm->ghandle[rank]);
    cudaLocalBuffer = localPool.base;
    printf("Init pool: %p\n", localPool.base);
  }

  pthread_barrier_wait(&shm->barrier);
  for (int i = 0; i < size; i++)
  {
    if (i != rank && !is_cpu_process(i, size)) {
      CUDA_CHECK(cudaIpcOpenMemHandle(&cudaBuffers[i], shm->ghandle[i], cudaIpcMemLazyEnablePeerAccess));
    } else {
      cudaBuffers[i] = NULL;
    }
  }
  
  printf("SharedMemoryCreate: %p\n", shm->data);
}

NcclCPUBackend::~NcclCPUBackend() {
  pthread_barrier_destroy(&shm->barrier);
  SharedMemoryUnregisterPortable(shm->data);
  SharedMemoryClose(&sharedMemoryInfo);
  printf("NcclCPUBackend::~NcclCPUBackend\n");
}

// This is a dummy allgather that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
// this function will be called {world_size} times
c10::intrusive_ptr<Work> NcclCPUBackend::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const AllgatherOptions& options) {
  return allgatherZeroCopy(outputTensors, inputTensors, options);
  auto& tensor = inputTensors[0];
  int rank = getRank();
  bool is_cuda = tensor.device().is_cuda();
  auto& localTensor = outputTensors[0][rank];
  check_device(tensor.device(), localTensor.device());
    // auto opt = torch::TensorOptions()
  //                .dtype(torch::kFloat32)
  //                .layout(torch::kStrided)
  //                .device(torch::kCPU)
  //                .requires_grad(false);
  // torch::from_blob(buffers[i], inputTensors[0].sizes(), opt);
  if (is_cuda) {
    cudaMemcpy(localTensor.data_ptr(), tensor.data_ptr(), tensor.nbytes(), cudaMemcpyDeviceToDevice);
    cudaMemcpy(localBuffer, tensor.data_ptr(), tensor.nbytes(), cudaMemcpyDeviceToHost);
  } else {
    std::memcpy(localTensor.data_ptr(), tensor.data_ptr(), tensor.nbytes());
    std::memcpy(localBuffer, tensor.data_ptr(), tensor.nbytes());
  }
  pthread_barrier_wait(&shm->barrier);

  for (int i = 0; i < getSize(); i++) {
    auto& outputTensor = outputTensors[0][i];
    if (i == rank) continue;
    cudaMemcpy(outputTensor.data_ptr(), buffers[i], tensor.nbytes(), cudaMemcpyDefault);
    cudaMemcpy(outputTensor.data_ptr(), buffers[i], tensor.nbytes(), cudaMemcpyDefault);
  }

  printf("allgather: %zu %zu %zu\n", inputTensors.size(), outputTensors.size(),
         outputTensors[0].size());
  return c10::make_intrusive<NcclCPUWork>(OpType::ALLGATHER, nullptr);
  // return ncclPG.allgather(outputTensors, inputTensors, options);
}

c10::intrusive_ptr<Work> NcclCPUBackend::allgatherZeroCopy(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts) {
  auto& tensor = inputTensors[0];
  int rank = getRank();
  bool is_cuda = tensor.device().is_cuda();
  auto& localTensor = outputTensors[0][rank];
  check_device(tensor.device(), localTensor.device());
  if (is_cuda) {
    cudaMemcpy(localTensor.data_ptr(), tensor.data_ptr(), tensor.nbytes(), cudaMemcpyDeviceToDevice);
    struct cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, cudaLocalBuffer));
    printf("addr: %p, type %d\n", cudaLocalBuffer, attr.type);
    CUDA_CHECK(cudaMemcpy(cudaLocalBuffer, tensor.data_ptr(), tensor.nbytes(), cudaMemcpyDeviceToDevice));
    *(uint64_t*)(localBuffer) = 0;  // offset
    // struct cudaPointerAttributes attr;
    // CUDA_CHECK(cudaPointerGetAttributes(&attr, tensor.data_ptr()));
    // printf("addr: %p, type %d\n", tensor.data_ptr(), attr.type);
  } else {
    std::memcpy(localTensor.data_ptr(), tensor.data_ptr(), tensor.nbytes());
    std::memcpy(localBuffer, tensor.data_ptr(), tensor.nbytes());
  }
  pthread_barrier_wait(&shm->barrier);

  for (int i = 0; i < getSize(); i++) {
    // auto& outputTensor = outputTensors[0][i];
    if(!is_cuda) {
      CUDA_CHECK(cudaHostRegister(outputTensors[0][i].data_ptr(), tensor.nbytes(), cudaHostRegisterPortable));
    }
    auto outputTensor = at::from_blob(outputTensors[0][i].data_ptr(), tensor.sizes(), tensor.strides(), 
                  tensor.options().device(c10::kCUDA));
    if (i == rank) continue;
    void* addr;
    if (!is_cpu_process(i, getSize())) {
      addr = (char*)(cudaBuffers[i]) + (*(uint64_t*)(buffers[i])); 
    } else {
      addr = buffers[i];
    }

    struct cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, addr));
    printf("buffers[%d]: %p vs %p, type: %d\n", i, buffers[i], addr, attr.type);
    outputTensor += at::from_blob(addr, tensor.sizes(), tensor.strides(), 
                  tensor.options().device(c10::kCUDA));
  }

  printf("allgather: %d %d %d\n", inputTensors.size(), outputTensors.size(),
         outputTensors[0].size());
  return c10::make_intrusive<NcclCPUWork>(OpType::ALLGATHER, nullptr);
  // return ncclPG.allgather(outputTensors, inputTensors, options);
}

c10::intrusive_ptr<Work> NcclCPUBackend::_allgather_base(
    at::Tensor& /* unused */, at::Tensor& /* unused */,
    const AllgatherOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

// This is a dummy allreduce that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> NcclCPUBackend::allreduce(
    std::vector<at::Tensor>& tensors, const AllreduceOptions& opts) {
  // auto& tensor = tensors[0];
  // int rank = getRank();
  // bool is_cuda = tensor.device().is_cuda();

  // cudaMemcpy(localBuffer, tensor.data_ptr(), tensor.nbytes(), cudaMemcpyDefault);

  // pthread_barrier_wait(&shm->barrier);

  // // auto outputTensor = at::zeros(tensor.sizes(), tensor.options().device(c10::kCUDA));
  // auto outputTensor = at::from_blob(localBuffer, tensor.sizes(), tensor.strides(), 
  //                 tensor.options().device(c10::kCUDA));

  // outputTensor += at::ones(tensor.sizes(), tensor.options().device(c10::kCUDA));
  // for (int i = 0; i < getSize(); i++) {
  //   // if (i == rank) continue;
  //   outputTensor += at::from_blob(buffers[i], tensor.sizes(), tensor.strides(), 
  //                 tensor.options().device(c10::kCUDA));
  // }
  // cudaMemcpy(tensor.data_ptr(), outputTensor.data_ptr(), tensor.nbytes(), cudaMemcpyDefault);

  return c10::make_intrusive<NcclCPUWork>(OpType::ALLREDUCE, nullptr);
}

c10::intrusive_ptr<Work> NcclCPUBackend::broadcast(
    std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
  auto& tensor = tensors[0];
  int rank = getRank();
  bool is_cuda = tensor.device().is_cuda();

  if (opts.rootRank == rank) {
    if (is_cuda) {
      cudaMemcpy(localBuffer, tensor.data_ptr(), tensor.nbytes(), cudaMemcpyDeviceToHost);
    } else {
      std::memcpy(localBuffer, tensor.data_ptr(), tensor.nbytes());
    }
    pthread_barrier_wait(&shm->barrier);
  } else {
    pthread_barrier_wait(&shm->barrier);
    if (is_cuda) {
      cudaMemcpy(tensor.data_ptr(), buffers[opts.rootRank], tensor.nbytes(), cudaMemcpyHostToDevice);
    } else {
      std::memcpy(tensor.data_ptr(), buffers[opts.rootRank], tensor.nbytes());
    }
  }

  return c10::make_intrusive<NcclCPUWork>(OpType::BROADCAST, nullptr);
}

c10::intrusive_ptr<Work> NcclCPUBackend::allreduce_coalesced(
    std::vector<at::Tensor>& tensors, const AllreduceCoalescedOptions& opts) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> NcclCPUBackend::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const AllToAllOptions& opts) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> NcclCPUBackend::alltoall_base(
    at::Tensor& outputTensor, at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> NcclCPUBackend::barrier(const BarrierOptions& opts) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> NcclCPUBackend::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const GatherOptions& opts) {
  return c10::make_intrusive<NcclCPUWork>(OpType::GATHER, nullptr);
}

c10::intrusive_ptr<Work> NcclCPUBackend::reduce(
    std::vector<at::Tensor>& tensors, const ReduceOptions& opts) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> NcclCPUBackend::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> NcclCPUBackend::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> NcclCPUBackend::send(std::vector<at::Tensor>& tensors,
                                              int dstRank, int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> NcclCPUBackend::recv(std::vector<at::Tensor>& tensors,
                                              int srcRank, int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> NcclCPUBackend::recvAnysource(
    std::vector<at::Tensor>& tensors, int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Backend> NcclCPUBackend::createNcclCPUBackend(
    const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size,
    const std::chrono::duration<float>& /* unused */) {
  return c10::make_intrusive<NcclCPUBackend>(store, rank, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createNcclCPUBackend", &NcclCPUBackend::createNcclCPUBackend);
}

}  // namespace c10d
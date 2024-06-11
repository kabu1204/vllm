#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <pthread.h>
#include <stdexcept>
#include <stdlib.h>
#include <sys/mman.h>
#include "ATen/ops/from_blob.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorImpl.h"
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

static inline bool check_dtype(c10::ScalarType dtype, bool is_cuda) {
  if (is_cuda) {
    return dtype == c10::ScalarType::Half;
  } else {
    return dtype == c10::ScalarType::BFloat16;
  }
}

#define ASSERT_MSG(cond, msg) \
  do {                        \
    if (!(cond)) {             \
        throw std::runtime_error(msg); \
    }                           \
  } while(0)

#define ASSERT_DEVICE(t1, t2) \
  ASSERT_MSG(check_device(t1.device(), t2.device()), "device mismatch")

#define ASSERT_DTYPE(tensor) \
  ASSERT_MSG(check_dtype(tensor.scalar_type(), tensor.device().is_cuda()), "dtype mismatch")

#define ASSERT_SIZE_LIMIT(size, limit) \
  ASSERT_MSG(tensor.is_contiguous(), "tensor is not contiguous")

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

static inline size_t align4KB(size_t size) {
    return (size + 4095) & ~4095;
}

static inline int getPointerType(void* ptr) {
  struct cudaPointerAttributes attr;
  CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
  return attr.type;
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
      tensorStore = &shm->tensorStorage[i];
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
}

c10::intrusive_ptr<Work> NcclCPUBackend::allgatherZeroCopy(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts) {
  int rank = getRank();
  auto& tensor = inputTensors[0];
  auto& localTensor = outputTensors[0][rank];
  bool is_cuda = tensor.device().is_cuda();
  ASSERT_DEVICE(tensor, localTensor);
  ASSERT_DTYPE(tensor);
  if (is_cuda) {
    // TODO(ycy): we do not need this
    CUDA_CHECK(cudaMemcpy(localTensor.data_ptr(), tensor.data_ptr(), tensor.nbytes(), cudaMemcpyDeviceToDevice));
    uint64_t fp16_offset = 0;
    uint64_t bf16_offset = align4KB(tensor.nbytes());
    void* bf16_ptr = (char*)cudaLocalBuffer + bf16_offset;
    CUDA_CHECK(cudaMemcpy(cudaLocalBuffer, tensor.data_ptr(), tensor.nbytes(), cudaMemcpyDeviceToDevice));
    auto bf16Tensor = at::from_blob(bf16_ptr, tensor.sizes(), tensor.strides(), 
                  tensor.options().dtype(c10::kBFloat16)).copy_(tensor);
    tensorStore->fp16Offset = fp16_offset;
    tensorStore->bf16Offset = bf16_offset;
    ASSERT_MSG(bf16_offset + tensor.nbytes() < NCCL_CPU_CUDA_ZERO_COPY_BUFFER_SIZE, "tensor too large");
  } else {
    std::memcpy(localTensor.data_ptr(), tensor.data_ptr(), tensor.nbytes());
    uint64_t fp16_offset = align4KB(tensor.nbytes());
    void* fp16_ptr = (char*)localBuffer + fp16_offset;
    std::memcpy(localBuffer, tensor.data_ptr(), tensor.nbytes());  // bf16
    auto fp16Tensor = at::from_blob(fp16_ptr, tensor.sizes(), tensor.strides(), 
        tensor.options().device(c10::kCPU).dtype(c10::kHalf)).copy_(tensor);    // fp16
    tensorStore->fp16Offset = fp16_offset;
    tensorStore->bf16Offset = 0;
  }
  pthread_barrier_wait(&shm->barrier);

  for (int i = 0; i < getSize(); i++) {
    if (i == rank) continue;

    if(!is_cuda && !outputTensors[0][i].is_pinned()) {
      CUDA_CHECK(cudaHostRegister(outputTensors[0][i].data_ptr(), tensor.nbytes(), cudaHostRegisterPortable));
    }
    auto outputTensor = at::from_blob(outputTensors[0][i].data_ptr(), tensor.sizes(), tensor.strides(), 
                  tensor.options().device(c10::kCUDA));

    
    size_t offset = (is_cuda ? shm->tensorStorage[i].fp16Offset : shm->tensorStorage[i].bf16Offset);
    void* addr;
    if (!is_cpu_process(i, getSize())) {
      addr = (char*)(cudaBuffers[i]) + offset; 
    } else {
      addr = (char*)(buffers[i]) + offset; 
    }

    outputTensor += at::from_blob(addr, tensor.sizes(), tensor.strides(), 
                  tensor.options().device(c10::kCUDA));
  }

  return c10::make_intrusive<NcclCPUWork>(OpType::ALLGATHER, nullptr);
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
  return allreduceZeroCopy(tensors, opts);
}

c10::intrusive_ptr<Work> NcclCPUBackend::allreduceZeroCopy(
    std::vector<at::Tensor>& tensors, const AllreduceOptions& opts) {
  auto& tensor = tensors[0];
  int rank = getRank();
  bool is_cuda = tensor.device().is_cuda();
  
  ASSERT_DTYPE(tensor);

  if (is_cuda) {
    CUDA_CHECK(cudaMemcpy(cudaLocalBuffer, tensor.data_ptr(), tensor.nbytes(), cudaMemcpyDeviceToDevice));
    tensorStore->fp16Offset = 0;
  } else {
    auto fp16Tensor = at::from_blob(localBuffer, tensor.sizes(), tensor.strides(), 
        tensor.options().device(c10::kCPU).dtype(c10::kHalf)).copy_(tensor);    // fp16
    tensorStore->fp16Offset = 0;
  }

  pthread_barrier_wait(&shm->barrier);

  if (!isCPU) {
    for (int i = 0; i < getSize(); i++) {
      if (i == rank) continue;
      size_t offset = shm->tensorStorage[i].fp16Offset;
      void* addr;
      if (is_cpu_process(i, getSize())) {
        addr = (char*)(buffers[i]) + offset;
      } else {
        addr = (char*)(cudaBuffers[i]) + offset;
      }
      tensor += at::from_blob(addr, tensor.sizes(), tensor.strides(), 
                    tensor.options().dtype(c10::kHalf));
    }
  } else {
    if (!tensor.is_pinned()) {
      CUDA_CHECK(cudaHostRegister(tensor.data_ptr(), tensor.nbytes(), cudaHostRegisterPortable));
    }
    auto out = at::from_blob(tensor.data_ptr(), tensor.sizes(), tensor.strides(), 
                    tensor.options().device(c10::kCUDA));
    for (int i = 0; i < getSize(); i++) {
      if (i == rank) continue;
      size_t offset = shm->tensorStorage[i].fp16Offset;
      void* addr;
      if (is_cpu_process(i, getSize())) {
        addr = (char*)(buffers[i]) + offset;
      } else {
        addr = (char*)(cudaBuffers[i]) + offset;
      }
      auto tmp = at::from_blob(addr, tensor.sizes(), tensor.strides(), 
                    tensor.options().dtype(c10::kHalf).device(c10::kCUDA));
      out += tmp;
    }
  }

  return c10::make_intrusive<NcclCPUWork>(OpType::ALLREDUCE, nullptr);
}

c10::intrusive_ptr<Work> NcclCPUBackend::broadcast(
    std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
  auto& tensor = tensors[0];
  int rank = getRank();
  bool is_cuda = tensor.device().is_cuda();

  ASSERT_DTYPE(tensor);

  if (opts.rootRank == rank) {
    if (is_cuda) {
      cudaMemcpy(cudaLocalBuffer, tensor.data_ptr(), tensor.nbytes(), cudaMemcpyDeviceToDevice);
      tensorStore->fp16Offset = 0;
    } else {
      std::memcpy(localBuffer, tensor.data_ptr(), tensor.nbytes());
      tensorStore->bf16Offset = 0;
    }
    pthread_barrier_wait(&shm->barrier);
  } else {
    pthread_barrier_wait(&shm->barrier);
    at::Tensor srcTensor, out;
    size_t offset;
    void* addr;
    if (is_cpu_process(opts.rootRank, getSize())) {
      offset = shm->tensorStorage[opts.rootRank].bf16Offset;
      addr = (char*)(buffers[opts.rootRank]) + offset;
      // TODO(ycy): current device
      srcTensor = at::from_blob(addr, tensor.sizes(), tensor.strides(), 
                    tensor.options().dtype(c10::kBFloat16).device(c10::kCUDA));
    } else {
      offset = shm->tensorStorage[opts.rootRank].fp16Offset;
      addr = (char*)(cudaBuffers[opts.rootRank]) + offset;
      srcTensor = at::from_blob(addr, tensor.sizes(), tensor.strides(), 
                    tensor.options().dtype(c10::kHalf).device(c10::kCUDA));
    }
    if (is_cuda) {
      out = at::from_blob(tensor.data_ptr(), tensor.sizes(), tensor.strides(), 
                    tensor.options());
    } else {
      out = at::from_blob(tensor.data_ptr(), tensor.sizes(), tensor.strides(), 
                    tensor.options().device(c10::kCUDA));
    }
    out.zero_() += srcTensor;
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
  auto& tensor = inputTensors[0];
  int rank = getRank();
  bool is_cuda = tensor.device().is_cuda();
  bool is_dst = (rank == opts.rootRank);

  ASSERT_DTYPE(tensor);

  if (is_dst) {
    ASSERT_DEVICE(tensor, outputTensors[0][rank]);
    outputTensors[0][rank].copy_(tensor);
  } else {
    if (is_cuda) {
      CUDA_CHECK(cudaMemcpy(cudaLocalBuffer, tensor.data_ptr(), tensor.nbytes(), cudaMemcpyDeviceToDevice));
      tensorStore->fp16Offset = 0;
    } else {
      std::memcpy(localBuffer, tensor.data_ptr(), tensor.nbytes());
      tensorStore->bf16Offset = 0;
    }
  }

  pthread_barrier_wait(&shm->barrier);
  
  if (rank != opts.rootRank) {
    return c10::make_intrusive<NcclCPUWork>(OpType::GATHER, nullptr);
  }

  for (int i = 0; i < getSize(); i++) {
    if (i == rank) continue;
    size_t offset;
    void* addr;
    at::Tensor srcTensor, out;
    if (is_cpu_process(i, getSize())) {
      offset = shm->tensorStorage[i].bf16Offset;
      addr = (char*)(buffers[i]) + offset;
      srcTensor = at::from_blob(addr, tensor.sizes(), tensor.strides(), 
                    tensor.options().dtype(c10::kBFloat16).device(c10::kCUDA));
    } else {
      offset = shm->tensorStorage[i].fp16Offset;
      addr = (char*)(cudaBuffers[i]) + offset;
      srcTensor = at::from_blob(addr, tensor.sizes(), tensor.strides(), 
                    tensor.options().dtype(c10::kHalf).device(c10::kCUDA));
    }

    if(!is_cuda && !outputTensors[0][i].is_pinned()) {
      // TODO(ycy): maybe use tensor.pin_memory() ?
      CUDA_CHECK(cudaHostRegister(outputTensors[0][i].data_ptr(), tensor.nbytes(), cudaHostRegisterPortable));
    }

    out = at::from_blob(outputTensors[0][i].data_ptr(), tensor.sizes(), tensor.strides(), 
                  tensor.options().device(c10::kCUDA));
    out.zero_() += srcTensor;
  }
  
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

const std::string NcclCPUBackend::getBackendName() const {
  return "Nccl-CPU";
}

}  // namespace c10d
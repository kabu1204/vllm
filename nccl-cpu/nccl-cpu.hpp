#pragma once

#include <torch/python.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#define USE_C10D_NCCL
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <pybind11/chrono.h>
#include "cuda_ipc.h"
#include <pthread.h>

#define NCCL_CPU_MAX_WORLD_SIZE 8

struct SharedTensorStorage {
    size_t fp16Offset;
    size_t bf16Offset;
};

// Shared memory struct holds a shared memory region.
struct SharedMemory {
  pthread_barrier_t barrier;
  volatile size_t size;
  cudaIpcMemHandle_t ghandle[NCCL_CPU_MAX_WORLD_SIZE];  // shared cuda IPC mem handle
  SharedTensorStorage tensorStorage[NCCL_CPU_MAX_WORLD_SIZE];
  char data[];
};

namespace c10d {

class NcclCPUBackend : public Backend {
 public:

  NcclCPUBackend(const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size);

  ~NcclCPUBackend() override;

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgatherZeroCopy(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions());

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  const std::string getBackendName() const override;

  // void setInternalNcclBackend();

  static c10::intrusive_ptr<Backend> createNcclCPUBackend(
      const c10::intrusive_ptr<::c10d::Store>& store,
      int rank,
      int size,
      const std::chrono::duration<float>& timeout);

  static void NcclCPUBackendConstructor() __attribute__((constructor)) {
    py::object module = py::module::import("torch.distributed");
    py::object register_backend =
        module.attr("Backend").attr("register_backend");
    std::vector<std::string> devices = {"cpu", "cuda"};
    register_backend("nccl-cpu", py::cpp_function(createNcclCPUBackend), false, devices);
  }

private:
  ProcessGroupNCCL ncclPG;
  SharedMemoryHandle sharedMemoryInfo;
  SharedMemory* shm;
  cudaIpcMemPool localPool;
  void* cudaBuffers[NCCL_CPU_MAX_WORLD_SIZE];   // device shared buffer addresses
  void* cudaLocalBuffer;

  // used for communication
  void* buffers[NCCL_CPU_MAX_WORLD_SIZE];   // host shared buffer addresses
  void* localBuffer;
  SharedTensorStorage* tensorStore;
  void* dptr;   // device pointer of shared host memory

  bool isCPU;
#ifdef NCCL_CPU_POLL
  std::thread* pollWorker_{nullptr};
#endif
};

class NcclCPUWork : public Work {
    friend class NcclCPUBackend;
public:
    NcclCPUWork(
        OpType opType,
        c10::intrusive_ptr<Work> ncclWork)
        : Work(
              -1, // rank, only used by recvAnySource, irrelevant in this demo
              opType),
          ncclWork_(std::move(ncclWork)) {}
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
    virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

private:
    // c10::intrusive_ptr<c10::ivalue::Future> future_;
    c10::intrusive_ptr<Work> ncclWork_;
};

} // namespace c10d
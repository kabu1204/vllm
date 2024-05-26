#include <iostream>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "nccl-cpu.hpp"

#define CUDA_CHECK(cmd) do { \
    cudaError_t e = cmd; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

namespace c10d {

bool NcclCPUWork::isCompleted() {
  throw std::runtime_error("NcclCPUWork::isCompleted: not supported");
}

bool NcclCPUWork::isSuccess() const {
  throw std::runtime_error("NcclCPUWork::isSuccess: not supported");

}

bool NcclCPUWork::wait(std::chrono::milliseconds timeout) {
  ncclWork_->wait();
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future> NcclCPUWork::getFuture() {
  throw std::runtime_error("NcclCPUWork::getFuture: not supported");
}

// If necessary, pass store/rank/size to the ctor and exchange connection
// information here
NcclCPUBackend::NcclCPUBackend(const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size)
    : Backend(rank, size), ncclPG(store, rank, size) {
      size_t sharedMemorySize = 81920;
      SharedMemoryCreate("/nccl_cpu_shm", sizeof(SharedMemory) + sharedMemorySize, &sharedMemoryInfo);
      SharedMemoryRegisterPortable(&dptr, sharedMemoryInfo.ptr, sharedMemoryInfo.size);
      shm = (SharedMemory*)sharedMemoryInfo.ptr;
      localBuffer = shm->data + rank * sharedMemorySize / size;
      for (int i = 0; i < size; i++)
      {
        buffers[i] = shm->data + i * sharedMemorySize / size;
      }
      if (rank == 0)
      {
        shm->size = sharedMemorySize;
        SemBarrierInit(&shm->barrier, size);
      }
    }

// This is a dummy allgather that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
// this function will be called {world_size} times
c10::intrusive_ptr<Work> NcclCPUBackend::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& options)
{
  CUDA_CHECK(cudaMemcpy(localBuffer, inputTensors[0].data_ptr(), inputTensors[0].nbytes(), cudaMemcpyDeviceToHost));
  auto opt = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCPU).requires_grad(false);
  SemBarrierWait(&shm->barrier);
  if (getRank() == 0)
  {
    for (int i = 0; i < getSize(); i++)
    {
      torch::Tensor tmp = torch::from_blob(buffers[i], inputTensors[0].sizes(), opt);
      std::cout<<"rank: "<< i << ":" << std::endl << tmp << std::endl;
    }
  }
  
  printf("allgather: %d %d %d\n", inputTensors.size(), outputTensors.size(), outputTensors[0].size());
  return ncclPG.allgather(outputTensors, inputTensors, options);
}

c10::intrusive_ptr<Work> NcclCPUBackend::_allgather_base(
    at::Tensor& /* unused */,
    at::Tensor& /* unused */,
    const AllgatherOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

// This is a dummy allreduce that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> NcclCPUBackend::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {

  printf("allreduce: %d\n", tensors.size());
  return ncclPG.allreduce(tensors, opts);
}

c10::intrusive_ptr<Work> NcclCPUBackend::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  return ncclPG.allreduce_coalesced(tensors, opts);
}

c10::intrusive_ptr<Work> NcclCPUBackend::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts) {
  ncclPG.alltoall(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<Work> NcclCPUBackend::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> NcclCPUBackend::barrier(
    const BarrierOptions& opts) {
  return ncclPG.barrier(opts);
}

c10::intrusive_ptr<Work> NcclCPUBackend::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  return ncclPG.broadcast(tensors, opts);
}

c10::intrusive_ptr<Work> NcclCPUBackend::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  return ncclPG.gather(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<Work> NcclCPUBackend::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  return ncclPG.reduce(tensors, opts);
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
  return ncclPG.scatter(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<Work> NcclCPUBackend::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> NcclCPUBackend::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> NcclCPUBackend::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Backend> NcclCPUBackend::createNcclCPUBackend(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& /* unused */) {
  return c10::make_intrusive<NcclCPUBackend>(store, rank, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createNcclCPUBackend", &NcclCPUBackend::createNcclCPUBackend);
}

} // namespace c10d
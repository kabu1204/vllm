#include <stdlib.h>
#include "cuda_ipc.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#define CUDA_CHECK(cmd) do { \
    cudaError_t e = cmd; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

static void SemBarrierUp(SemBarrier *barrier) {
  sem_wait(&barrier->mutex);
  if (++barrier->count == barrier->n) {
    int i;
    for (i = 0; i < barrier->n; i++) {
      sem_post(&barrier->turnstile);
    }
  }
  sem_post(&barrier->mutex);
  sem_wait(&barrier->turnstile);
}

static void SemBarrierDown(SemBarrier *barrier) {
  sem_wait(&barrier->mutex);
  if (--barrier->count == 0) {
    int i;
    for (i = 0; i < barrier->n; i++) {
      sem_post(&barrier->turnstile2);
    }
  }
  sem_post(&barrier->mutex);
  sem_wait(&barrier->turnstile2);
}

void SemBarrierInit(SemBarrier *barrier, int n) {
    barrier->n = n;
    barrier->count = 0;
    sem_init(&barrier->mutex, 1, 1);
    sem_init(&barrier->turnstile, 1, 0);
    sem_init(&barrier->turnstile2, 1, 0);
}

void SemBarrierWait(SemBarrier *barrier) {
    SemBarrierUp(barrier);
    SemBarrierDown(barrier);
}

void SharedMemoryRegisterPortable(void** dptr, void *ptr, size_t sz) {
    CUDA_CHECK(cudaHostRegister(ptr, sz, cudaHostRegisterPortable));
    CUDA_CHECK(cudaHostGetDevicePointer(dptr, ptr, 0));
    // printf("SharedMemoryRegisterPortable: %p\n", dptr);
}

void SharedMemoryUnregisterPortable(void *ptr) {
    CUDA_CHECK(cudaHostUnregister(ptr));
}

int SharedMemoryCreate(const char *name, size_t sz, SharedMemoryHandle *info) {
    sz = (sz + 4095) & ~4095;   // align 4KB boundary

    // Create a shared memory segment
    int shm_fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0777);
    if (shm_fd == -1) {
        perror("shm_open");
        exit(EXIT_FAILURE);
    }
    
    // Set the size of the shared memory segment
    if (ftruncate(shm_fd, sz) == -1) {
        perror("ftruncate");
        shm_unlink(name);
        exit(EXIT_FAILURE);
    }
    
    // Map the shared memory segment into the address space of the process
    void* ptr = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        shm_unlink(name);
        exit(EXIT_FAILURE);
    }
    
    info->fd = shm_fd;
    info->size = sz;
    info->ptr = ptr;
    strncpy(info->name, name, sizeof(info->name));

    return 0;
}

int SharedMemoryOpen(const char *name, size_t sz, SharedMemoryHandle *info) {
    sz = (sz + 4095) & ~4095;   // align 4KB boundary

    // Create a shared memory segment
    int shm_fd = shm_open(name, O_RDWR, 0777);
    if (shm_fd == -1) {
        perror("shm_open");
        exit(EXIT_FAILURE);
    }
    
    // Map the shared memory segment into the address space of the process
    void* ptr = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        shm_unlink(name);
        exit(EXIT_FAILURE);
    }
    
    info->fd = shm_fd;
    info->size = sz;
    info->ptr = ptr;
    strncpy(info->name, name, sizeof(info->name));

    return 0;
}

void SharedMemoryClose(SharedMemoryHandle *info) {
    // Unmap the shared memory segment
    if (munmap(info->ptr, info->size) == -1) {
        perror("munmap");
        exit(EXIT_FAILURE);
    }
    
    // Close the shared memory segment
    shm_unlink(info->name);
    if (close(info->fd) == -1) {
        perror("close");
        exit(EXIT_FAILURE);
    }
}

void cudaIpcInitPool(int devIdx, size_t sz, cudaIpcMemPool* pool, cudaIpcMemHandle_t* handle) {
    sz = (sz + ~66535) & ~66535;   // align 64KB boundary
    
    CUDA_CHECK(cudaSetDevice(devIdx));
    CUDA_CHECK(cudaMalloc(&pool->base, sz));


    struct cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, pool->base));
    printf("IPC INIT: %p, type %d\n", pool->base, attr.type);

    CUDA_CHECK(cudaIpcGetMemHandle(handle, pool->base));

    pool->handle = handle;
    pool->sz = sz;
    pool->devIdx = devIdx;
}

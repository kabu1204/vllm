#include <sys/shm.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

int CreateSharedMemory(const char* shmpath, size_t size) {
  // Create a shared memory segment
  int shm_fd = shm_open(shmpath, O_CREAT | O_RDWR | O_EXCL, 0666);
  if (shm_fd == -1) {
    perror("shm_open");
    return -1;
  }

  // Set the size of the shared memory segment
  if (ftruncate(shm_fd, size) == -1) {
    perror("ftruncate");
    return -1;
  }

  // Close the shared memory segment
  if (close(shm_fd) == -1) {
    perror("close");
    return -1;
  }

  return 0;
}

int OpenSharedMemory(const char* shmpath, size_t size) {
  // Open a shared memory segment
  int shm_fd = shm_open(shmpath, O_RDWR, 0666);
  if (shm_fd == -1) {
    perror("shm_open");
    return -1;
  }

  return shm_fd;
}

void* MmapSharedMemory(int fd, size_t size) {
  // Map the shared memory segment into the address space of the process
  void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    perror("mmap");
    return -1;
  }

  return ptr;
}

void UnmapSharedMemory(void* ptr, size_t size) {
  // Unmap the shared memory segment
  if (munmap(ptr, size) == -1) {
    perror("munmap");
    return -1;
  }
}

void shared_memory_example(size_t size) {
  // Create a shared memory segment
  int shm_fd = shm_open("/tmp/nccl_cpu_shm", O_CREAT | O_RDWR | O_EXCL, 0666);
  if (shm_fd == -1) {
    perror("shm_open");
    return;
  }

  // Set the size of the shared memory segment
  if (ftruncate(shm_fd, size) == -1) {
    perror("ftruncate");
    return;
  }

  // Map the shared memory segment into the address space of the process
  void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (ptr == MAP_FAILED) {
    perror("mmap");
    return;
  }

  // Write to the shared memory segment
  sprintf((char*)ptr, "Hello, shared memory!");

  // Unmap the shared memory segment
  if (munmap(ptr, size) == -1) {
    perror("munmap");
    return;
  }

  // Close the shared memory segment
  if (close(shm_fd) == -1) {
    perror("close");
    return;
  }

  // Remove the shared memory segment
  if (shm_unlink("/example") == -1) {
    perror("shm_unlink");
    return;
  }
}
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <memory.h>
#include <sys/un.h>
#include <vector>
#include <semaphore.h>

struct SemBarrier {
  int n;
  int count;
  sem_t mutex;
  sem_t turnstile;
  sem_t turnstile2;
};

void SemBarrierInit(SemBarrier *barrier, int n);
void SemBarrierWait(SemBarrier *barrier);

struct SharedMemoryHandle {
  int fd;
  size_t size;
  void* ptr;
};

int SharedMemoryCreate(const char *name, size_t sz, SharedMemoryHandle *info);

// int SharedMemoryOpen(const char *name, size_t sz, SharedMemoryHandle *info);

void SharedMemoryClose(SharedMemoryHandle *info);

void SharedMemoryRegisterPortable(void** dptr, void *ptr, size_t sz);

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// LD_PRELOAD workaround: FABRIC (MNNVL) memory after cuCheckpointProcess restore, NVIDIA driver 610.43.02.
//
// A restore recreates the process's RM client but loses two pieces of state that libcuda set up once and never
// sets up again:
//  1. The IMEX channel subscription (NV0000_CTRL_CMD_CLIENT_SUBSCRIBE_TO_IMEX_CHANNEL). Without it, FABRIC
//     allocation and import fail with CUDA_ERROR_NOT_PERMITTED.
//  2. The OS event libcuda waits on for fabric imports. libcuda then passes pOsEvent = -1 in the import
//     allocation (NV00F9), RM fails it with NV_ERR_OBJECT_NOT_FOUND (CUDA_ERROR_INVALID_VALUE), and libcuda waits
//     on fd -1.
// This library watches libcuda's RM ioctls and retries the FABRIC calls that RM refuses:
//  - FABRIC allocation, export or import refused with NV_ERR_INSUFFICIENT_PERMISSIONS on a client that is not
//    subscribed: subscribe it to the channel libcuda used, then retry.
//  - Import still refused while pOsEvent = -1: allocate an equivalent OS event on the client, retry with it and
//    redirect libcuda's wait on fd -1 to it.
// Calls that RM accepts, including those the driver's restore makes, pass through unchanged.
//
// Build: gcc -O2 -shared -fPIC -o libfabric_restore_shim.so fabric_restore_shim.c -ldl -lpthread
// Use:   LD_PRELOAD=/path/to/libfabric_restore_shim.so in the environment of the vLLM workers.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define NV_IOCTL_TYPE 0x46
#define NR_RM_CONTROL 0x2a
#define NR_RM_ALLOC 0x2b
#define NR_REGISTER_FD 0xc9
#define NR_ALLOC_OS_EVENT 0xce
#define NV_MEMORY_FABRIC 0xf8
#define NV_MEMORY_FABRIC_IMPORT_V2 0xf9
#define NV_MEMORY_EXPORT 0xe0
#define NV_ERR_INSUFFICIENT_PERMISSIONS 0x1b
#define SUBSCRIBE_TO_IMEX_CHANNEL 0xd08
#define MAX_CLIENTS 64

typedef struct { uint32_t hClient, hObject, cmd, flags; uint64_t params; uint32_t paramsSize, status; } RmControl;
typedef struct {
  uint32_t hRoot, hObjectParent, hObjectNew, hClass;
  uint64_t pAllocParms, pRightsRequested;
  uint32_t paramsSize, flags, status, pad;
} RmAlloc;                                                              // NVOS64_PARAMETERS
typedef struct { uint64_t devDescriptor; uint32_t channel, pad; } SubscribeParams;
typedef struct { uint32_t hClient, hDevice, fd, status; } OsEventParams;  // nv_ioctl_alloc_os_event_t
typedef struct {
  uint32_t imexChannel;
  uint8_t expPacket[32];
  uint16_t index;
  uint32_t flags;
  uint64_t pOsEvent, id;
} FabricImportParams;                                                   // NV00F9_ALLOCATION_PARAMETERS

static int (*real_ioctl)(int, unsigned long, ...);
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static int channel = -1;                    // channel libcuda subscribed to
static uint32_t subscribed[MAX_CLIENTS];
static int n_subscribed;
static OsEventParams os_event;              // libcuda's last OS-event allocation
static char os_event_dev[64];
static uint32_t event_client;               // client our replacement OS event belongs to
static int event_fd = -1;
static __thread int wait_fd = -1;           // stands in for libcuda's fd -1 on this thread
static int verbose = -1;

#define LOG(...) do { if (verbose) fprintf(stderr, "fabric_restore_shim: " __VA_ARGS__); } while (0)

static unsigned long nv_req(unsigned nr, unsigned size) {
  return (3ul << 30) | ((unsigned long)size << 16) | (NV_IOCTL_TYPE << 8) | nr;
}

static int is_subscribed(uint32_t client) {
  for (int i = 0; i < n_subscribed; i++)
    if (subscribed[i] == client) return 1;
  return 0;
}

static void mark_subscribed(uint32_t client) {
  if (!is_subscribed(client) && n_subscribed < MAX_CLIENTS) subscribed[n_subscribed++] = client;
}

// Caller holds mu.
static void subscribe(int ctl, uint32_t client) {
  char path[64];
  snprintf(path, sizeof path, "/dev/nvidia-caps-imex-channels/channel%d", channel);
  int fd = open(path, O_RDONLY | O_CLOEXEC);  // kept open like libcuda's own channel fd
  if (fd < 0) return;
  SubscribeParams p = {(uint64_t)fd, (uint32_t)channel, 0};
  RmControl c = {client, client, SUBSCRIBE_TO_IMEX_CHANNEL, 0, (uint64_t)(uintptr_t)&p, sizeof p, 0};
  if (real_ioctl(ctl, nv_req(NR_RM_CONTROL, sizeof c), &c) == 0 && c.status == 0) {
    mark_subscribed(client);
    LOG("subscribed restored client %#x to IMEX channel %d\n", client, channel);
  } else {
    LOG("IMEX subscribe of client %#x failed: status %#x\n", client, c.status);
    close(fd);
  }
}

// Caller holds mu. Returns an OS-event fd registered on `client`, or -1.
static int replacement_event(int ctl, uint32_t client) {
  if (event_fd >= 0 && event_client == client) return event_fd;
  if (!os_event_dev[0]) return -1;
  int fd = open(os_event_dev, O_RDWR | O_CLOEXEC);
  if (fd < 0) return -1;
  int ctl_fd = ctl;
  OsEventParams ev = os_event;
  ev.hClient = client;
  ev.fd = (uint32_t)fd;
  ev.status = 0;
  if (real_ioctl(fd, nv_req(NR_REGISTER_FD, sizeof ctl_fd), &ctl_fd) != 0 ||
      real_ioctl(fd, nv_req(NR_ALLOC_OS_EVENT, sizeof ev), &ev) != 0 || ev.status != 0) {
    LOG("OS event allocation for client %#x failed: status %#x\n", client, ev.status);
    close(fd);
    return -1;
  }
  if (event_fd >= 0) close(event_fd);
  event_fd = fd;
  event_client = client;
  LOG("allocated fabric import OS event fd %d on %s for client %#x\n", fd, os_event_dev, client);
  return fd;
}

static void record(int fd, unsigned nr, void* arg) {
  if (nr == NR_RM_CONTROL) {
    RmControl* c = arg;
    if (c->cmd == SUBSCRIBE_TO_IMEX_CHANNEL && c->status == 0) {
      pthread_mutex_lock(&mu);
      channel = (int)((SubscribeParams*)(uintptr_t)c->params)->channel;
      mark_subscribed(c->hClient);
      pthread_mutex_unlock(&mu);
    }
  } else if (nr == NR_ALLOC_OS_EVENT && ((OsEventParams*)arg)->status == 0) {
    char link[32];
    snprintf(link, sizeof link, "/proc/self/fd/%d", fd);
    pthread_mutex_lock(&mu);
    os_event = *(OsEventParams*)arg;
    ssize_t n = readlink(link, os_event_dev, sizeof os_event_dev - 1);
    os_event_dev[n > 0 ? n : 0] = 0;
    pthread_mutex_unlock(&mu);
  }
}

int ioctl(int fd, unsigned long req, ...) {
  if (!real_ioctl) real_ioctl = dlsym(RTLD_NEXT, "ioctl");
  if (verbose < 0) verbose = getenv("VLLM_FABRIC_RESTORE_SHIM_VERBOSE") != NULL;
  va_list ap;
  va_start(ap, req);
  void* arg = va_arg(ap, void*);
  va_end(ap);
  if (fd == -1 && wait_fd >= 0) fd = wait_fd;
  unsigned nr = req & 0xff;
  if (((req >> 8) & 0xff) != NV_IOCTL_TYPE) return real_ioctl(fd, req, arg);

  int rc = real_ioctl(fd, req, arg);
  if (rc != 0) return rc;
  if (nr != NR_RM_ALLOC) {
    record(fd, nr, arg);
    return rc;
  }
  // Act only on FABRIC calls that RM refused: the driver's own restore re-creates allocations through the same
  // ioctl and must not be disturbed.
  RmAlloc* a = arg;
  int fabric = a->hClass == NV_MEMORY_FABRIC || a->hClass == NV_MEMORY_FABRIC_IMPORT_V2 || a->hClass == NV_MEMORY_EXPORT;
  if (fabric && a->status == NV_ERR_INSUFFICIENT_PERMISSIONS) {
    pthread_mutex_lock(&mu);
    int retry = channel >= 0 && !is_subscribed(a->hRoot);
    if (retry) subscribe(fd, a->hRoot);
    retry = retry && is_subscribed(a->hRoot);
    pthread_mutex_unlock(&mu);
    if (retry) {
      a->status = 0;
      rc = real_ioctl(fd, req, arg);
      LOG("retried class %#x on client %#x after subscribing: status %#x\n", a->hClass, a->hRoot, a->status);
      if (rc != 0) return rc;
    }
  }
  FabricImportParams* p = (FabricImportParams*)(uintptr_t)a->pAllocParms;
  if (a->hClass == NV_MEMORY_FABRIC_IMPORT_V2 && a->status != 0 && p && p->pOsEvent == (uint64_t)-1) {
    pthread_mutex_lock(&mu);
    int ev = replacement_event(fd, a->hRoot);
    pthread_mutex_unlock(&mu);
    if (ev >= 0) {
      a->status = 0;
      p->pOsEvent = (uint64_t)ev;
      wait_fd = ev;  // libcuda next waits for the import on its own event, which is fd -1
      rc = real_ioctl(fd, req, arg);
      p->pOsEvent = (uint64_t)-1;
      LOG("retried fabric import on client %#x with OS event fd %d: status %#x\n", a->hRoot, ev, a->status);
    }
  }
  return rc;
}

static int redirect(struct pollfd* fds, nfds_t n) {
  int hit = 0;
  for (nfds_t i = 0; wait_fd >= 0 && i < n; i++)
    if (fds[i].fd == -1) {
      fds[i].fd = wait_fd;
      hit = 1;
    }
  return hit;
}

static void unredirect(struct pollfd* fds, nfds_t n) {
  for (nfds_t i = 0; i < n; i++)
    if (fds[i].fd == wait_fd) fds[i].fd = -1;
}

int poll(struct pollfd* fds, nfds_t n, int timeout) {
  static int (*real)(struct pollfd*, nfds_t, int);
  if (!real) real = dlsym(RTLD_NEXT, "poll");
  int hit = redirect(fds, n);
  int rc = real(fds, n, timeout);
  if (hit) unredirect(fds, n);
  return rc;
}

int ppoll(struct pollfd* fds, nfds_t n, const struct timespec* t, const sigset_t* sig) {
  static int (*real)(struct pollfd*, nfds_t, const struct timespec*, const sigset_t*);
  if (!real) real = dlsym(RTLD_NEXT, "ppoll");
  int hit = redirect(fds, n);
  int rc = real(fds, n, t, sig);
  if (hit) unredirect(fds, n);
  return rc;
}

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <string.h>
#include "error.h"

#define N 20
#define SIZE 1024

int main() {
  cuInit(0);

  int count;
  cuDeviceGetCount(&count);

  for (int i = 0; i < count; i++) {
    CUdevice device;
    cuDeviceGet(&device, i);

    char name[256];
    cuDeviceGetName(name, 256, device);

    size_t bytes;
    cuDeviceTotalMem(&bytes, device);

    int major, minor;
    cuDeviceComputeCapability(&major, &minor, device);

    fprintf(stdout, "Device %d (%s, CC %d.%d, %zuMB):\n", i, name, major, minor, bytes >> 20);

    CUcontext context;
    cuCtxCreate(&context, CU_CTX_BLOCKING_SYNC, device);

    CUdeviceptr ptrs[N];
    struct timeval start, stop;
    int error;
    if ((error = gettimeofday(&start, NULL)) != 0) {
      fprintf(stderr, "Unable to get start time: %s\n", strerror(error));
      return error;
    }
    for (size_t j = 0; j < N; j++)
      cuMemAlloc(&ptrs[j], SIZE);
    if ((error = gettimeofday(&stop, NULL)) != 0) {
      fprintf(stderr, "Unable to get stop time: %s\n", strerror(error));
      return error;
    }

    double time = (double)(stop.tv_sec - start.tv_sec) + ((double)(stop.tv_usec - start.tv_usec) * 1.E-6);
    fprintf(stderr, "  cuMemAlloc: %.3es\n", time / (double)N);

    if ((error = gettimeofday(&start, NULL)) != 0) {
      fprintf(stderr, "Unable to get start time: %s\n", strerror(error));
      return error;
    }
    for (size_t j = 0; j < N; j++)
      cuMemFree(ptrs[j]);
    if ((error = gettimeofday(&stop, NULL)) != 0) {
      fprintf(stderr, "Unable to get stop time: %s\n", strerror(error));
      return error;
    }

    time = (double)(stop.tv_sec - start.tv_sec) + ((double)(stop.tv_usec - start.tv_usec) * 1.E-6);
    fprintf(stderr, "  cuMemFree : %.3es\n", time / (double)N);

    cuCtxDestroy(context);
  }

  return 0;
}

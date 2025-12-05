#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../kernels/vector_add/runner.cu"
#include "../utils/cuda_utils.h"

void print_results(int* h_a, int* h_b, int* h_c, float ms, int n_print) {
  print_vector(h_a, n_print, "A");
  print_vector(h_b, n_print, "B");
  print_vector(h_c, n_print, "C = A + B");
  printf("Kernel execution time: %.3f ms\n", ms);
}

void run_naive_benchmark(int n) {
  int bytes = sizeof(int) * n;
  int *h_a, *h_b, *h_c;
  int *d_a, *d_b, *d_c;

  h_a = (int*)malloc(bytes);
  h_b = (int*)malloc(bytes);
  h_c = (int*)malloc(bytes);

  for (int i = 0; i < n; i++) {
    h_a[i] = i;
    h_b[i] = 2 * i;
  }

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_vector_add_naive(d_a, d_b, d_c, n);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  print_results(h_a, h_b, h_c, milliseconds, 10);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);
}

void run_um_benchmark(int n) {
  int bytes = sizeof(int) * n;
  int device;
  cudaGetDevice(&device);

  int *a, *b, *c;

  cudaMallocManaged(&a, bytes);
  cudaMallocManaged(&b, bytes);
  cudaMallocManaged(&c, bytes);

  for (int i = 0; i < n; i++) {
    a[i] = i;
    b[i] = 2 * i;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_vector_add_unified_memory(a, b, c, n, device);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  print_results(a, b, c, milliseconds, 10);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s <kernel_num>\n", argv[0]);
    printf("  0 - naive\n");
    printf("  1 - unified memory\n");
    return 1;
  }

  int kernel_num = atoi(argv[1]);
  print_device_info();

  int n = 1 << 16;

  switch(kernel_num) {
    case 0:
      run_naive_benchmark(n);
      break;
    case 1:
      run_um_benchmark(n);
      break;
    default:
      printf("Invalid kernel number\n");
      return 1;
  }

  return 0;
}

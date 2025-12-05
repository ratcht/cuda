#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>
#include "../kernels/softmax/runner.cu"
#include "../utils/cuda_utils.h"

void print_softmax_results(float* h_a, float* h_b, int rows, int cols, float ms) {
  if (rows <= 8 && cols <= 8) {
    print_matrix(h_a, rows, cols, "Input");
    print_matrix(h_b, rows, cols, "Output");
  }
  printf("Matrix size: %dx%d\n", rows, cols);
  printf("Kernel execution time: %.3f ms\n", ms);
}

void run_naive_benchmark(int rows, int cols) {
  int bytes = sizeof(float) * rows * cols;
  float *h_a, *h_b;
  float *d_a, *d_b;

  h_a = (float*)malloc(bytes);
  h_b = (float*)malloc(bytes);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);

  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, 462);
  curandGenerateUniform(prng, d_a, rows * cols);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_softmax_naive(d_a, d_b, rows, cols);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);

  print_softmax_results(h_a, h_b, rows, cols, milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_a);
  cudaFree(d_b);
  free(h_a);
  free(h_b);
  curandDestroyGenerator(prng);
}

void run_shared_memory_benchmark(int rows, int cols) {
  int bytes = sizeof(float) * rows * cols;
  float *h_a, *h_b;
  float *d_a, *d_b;

  h_a = (float*)malloc(bytes);
  h_b = (float*)malloc(bytes);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);

  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, 462);
  curandGenerateUniform(prng, d_a, rows * cols);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_softmax_shared_memory(d_a, d_b, rows, cols);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);

  print_softmax_results(h_a, h_b, rows, cols, milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_a);
  cudaFree(d_b);
  free(h_a);
  free(h_b);
  curandDestroyGenerator(prng);
}

void run_warp_shuffle_benchmark(int rows, int cols) {
  int bytes = sizeof(float) * rows * cols;
  float *h_a, *h_b;
  float *d_a, *d_b;

  h_a = (float*)malloc(bytes);
  h_b = (float*)malloc(bytes);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);

  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, 462);
  curandGenerateUniform(prng, d_a, rows * cols);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_softmax_warp_shuffle(d_a, d_b, rows, cols);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);

  print_softmax_results(h_a, h_b, rows, cols, milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_a);
  cudaFree(d_b);
  free(h_a);
  free(h_b);
  curandDestroyGenerator(prng);
}

int main(int argc, char** argv) {
  if (argc < 4) {
    printf("Usage: %s <kernel_num> <rows> <cols>\n", argv[0]);
    printf("  0 - naive\n");
    printf("  1 - shared memory\n");
    printf("  2 - warp shuffle\n");
    return 1;
  }

  int kernel_num = atoi(argv[1]);
  int rows = atoi(argv[2]);
  int cols = atoi(argv[3]);
  print_device_info();

  switch(kernel_num) {
    case 0:
      run_naive_benchmark(rows, cols);
      break;
    case 1:
      run_shared_memory_benchmark(rows, cols);
      break;
    case 2:
      run_warp_shuffle_benchmark(rows, cols);
      break;
    default:
      printf("Invalid kernel number\n");
      return 1;
  }

  return 0;
}

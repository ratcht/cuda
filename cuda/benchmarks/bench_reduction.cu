#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>
#include "../kernels/reduction/runner.cu"
#include "../utils/cuda_utils.h"

void print_reduction_results(float* h_v, float gpu_result, float expected, int n, float ms) {
  printf("Array size: %d\n", n);
  printf("GPU result: %f\n", gpu_result);
  printf("Expected: %f\n", expected);
  printf("Difference: %f\n", fabsf(gpu_result - expected));
  printf("Kernel execution time: %.3f ms\n", ms);

  if (fabsf(gpu_result - expected) < 1e-3) {
    printf("Validation passed!\n");
  } else {
    printf("Validation failed!\n");
  }
}

void run_divergent_benchmark(int n) {
  int bytes = sizeof(float) * n;
  float *h_v, *h_v_r;
  float *d_v, *d_v_r;

  h_v = (float*)malloc(bytes);
  h_v_r = (float*)malloc(bytes);

  cudaMalloc(&d_v, bytes);
  cudaMalloc(&d_v_r, bytes);

  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, 42);
  curandGenerateUniform(prng, d_v, n);

  cudaMemcpy(h_v, d_v, bytes, cudaMemcpyDeviceToHost);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_reduction_divergent(d_v, d_v_r, n);
  run_reduction_divergent(d_v_r, d_v_r, REDUCTION_DIVERGENT_SIZE);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

  float expected = 0;
  for (int i = 0; i < n; i++) {
    expected += h_v[i];
  }

  print_reduction_results(h_v, h_v_r[0], expected, n, milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_v);
  cudaFree(d_v_r);
  free(h_v);
  free(h_v_r);
  curandDestroyGenerator(prng);
}

void run_bank_conflicts_benchmark(int n) {
  int bytes = sizeof(float) * n;
  float *h_v, *h_v_r;
  float *d_v, *d_v_r;

  h_v = (float*)malloc(bytes);
  h_v_r = (float*)malloc(bytes);

  cudaMalloc(&d_v, bytes);
  cudaMalloc(&d_v_r, bytes);

  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, 42);
  curandGenerateUniform(prng, d_v, n);

  cudaMemcpy(h_v, d_v, bytes, cudaMemcpyDeviceToHost);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_reduction_bank_conflicts(d_v, d_v_r, n);
  run_reduction_bank_conflicts(d_v_r, d_v_r, REDUCTION_BANK_CONFLICTS_SIZE);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

  float expected = 0;
  for (int i = 0; i < n; i++) {
    expected += h_v[i];
  }

  print_reduction_results(h_v, h_v_r[0], expected, n, milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_v);
  cudaFree(d_v_r);
  free(h_v);
  free(h_v_r);
  curandDestroyGenerator(prng);
}

void run_no_bank_conflicts_benchmark(int n) {
  int bytes = sizeof(float) * n;
  float *h_v, *h_v_r;
  float *d_v, *d_v_r;

  h_v = (float*)malloc(bytes);
  h_v_r = (float*)malloc(bytes);

  cudaMalloc(&d_v, bytes);
  cudaMalloc(&d_v_r, bytes);

  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, 42);
  curandGenerateUniform(prng, d_v, n);

  cudaMemcpy(h_v, d_v, bytes, cudaMemcpyDeviceToHost);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_reduction_no_bank_conflicts(d_v, d_v_r, n);
  run_reduction_no_bank_conflicts(d_v_r, d_v_r, REDUCTION_NO_BANK_CONFLICTS_SIZE);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

  float expected = 0;
  for (int i = 0; i < n; i++) {
    expected += h_v[i];
  }

  print_reduction_results(h_v, h_v_r[0], expected, n, milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_v);
  cudaFree(d_v_r);
  free(h_v);
  free(h_v_r);
  curandDestroyGenerator(prng);
}

void run_sequential_addressing_benchmark(int n) {
  int bytes = sizeof(float) * n;
  float *h_v, *h_v_r;
  float *d_v, *d_v_r;

  h_v = (float*)malloc(bytes);
  h_v_r = (float*)malloc(bytes);

  cudaMalloc(&d_v, bytes);
  cudaMalloc(&d_v_r, bytes);

  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, 42);
  curandGenerateUniform(prng, d_v, n);

  cudaMemcpy(h_v, d_v, bytes, cudaMemcpyDeviceToHost);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_reduction_sequential_addressing(d_v, d_v_r, n);
  run_reduction_sequential_addressing(d_v_r, d_v_r, REDUCTION_SEQUENTIAL_SIZE);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

  float expected = 0;
  for (int i = 0; i < n; i++) {
    expected += h_v[i];
  }

  print_reduction_results(h_v, h_v_r[0], expected, n, milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_v);
  cudaFree(d_v_r);
  free(h_v);
  free(h_v_r);
  curandDestroyGenerator(prng);
}

void run_first_add_during_load_benchmark(int n) {
  int bytes = sizeof(float) * n;
  float *h_v;
  float *d_v, *d_v_r;

  h_v = (float*)malloc(bytes);

  cudaMalloc(&d_v, bytes);
  cudaMalloc(&d_v_r, sizeof(float));

  for (int i = 0; i < n; i++) {
    h_v[i] = 1.0f;
  }

  cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);
  cudaMemset(d_v_r, 0, sizeof(float));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_reduction_first_add_during_load(d_v, d_v_r, n);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  float gpu_result;
  cudaMemcpy(&gpu_result, d_v_r, sizeof(float), cudaMemcpyDeviceToHost);

  float expected = (float)n;

  print_reduction_results(h_v, gpu_result, expected, n, milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_v);
  cudaFree(d_v_r);
  free(h_v);
}

void run_unroll_last_warp_benchmark(int n) {
  int bytes = sizeof(float) * n;
  float *h_v, *h_v_r;
  float *d_v, *d_v_r;

  h_v = (float*)malloc(bytes);
  h_v_r = (float*)malloc(bytes);

  cudaMalloc(&d_v, bytes);
  cudaMalloc(&d_v_r, bytes);

  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, 42);
  curandGenerateUniform(prng, d_v, n);

  cudaMemcpy(h_v, d_v, bytes, cudaMemcpyDeviceToHost);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_reduction_unroll_last_warp(d_v, d_v_r, n);
  run_reduction_unroll_last_warp(d_v_r, d_v_r, REDUCTION_UNROLL_LAST_SIZE);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

  float expected = 0;
  for (int i = 0; i < n; i++) {
    expected += h_v[i];
  }

  print_reduction_results(h_v, h_v_r[0], expected, n, milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_v);
  cudaFree(d_v_r);
  free(h_v);
  free(h_v_r);
  curandDestroyGenerator(prng);
}

void run_completely_unrolled_benchmark(int n) {
  int bytes = sizeof(float) * n;
  float *h_v;
  float *d_v, *d_v_r;

  h_v = (float*)malloc(bytes);

  cudaMalloc(&d_v, bytes);
  cudaMalloc(&d_v_r, sizeof(float));

  for (int i = 0; i < n; i++) {
    h_v[i] = 1.0f;
  }

  cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);
  cudaMemset(d_v_r, 0, sizeof(float));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_reduction_completely_unrolled(d_v, d_v_r, n);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  float gpu_result;
  cudaMemcpy(&gpu_result, d_v_r, sizeof(float), cudaMemcpyDeviceToHost);

  float expected = (float)n;

  print_reduction_results(h_v, gpu_result, expected, n, milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_v);
  cudaFree(d_v_r);
  free(h_v);
}

void run_cooperative_groups_benchmark(int n) {
  float *d_v, *d_v_r;
  float *v;

  cudaMallocManaged(&v, sizeof(float) * n);
  cudaMallocManaged(&d_v_r, sizeof(float));
  *d_v_r = 0;

  for (int i = 0; i < n; i++) {
    v[i] = 1.0f;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_reduction_cooperative_groups(v, d_v_r, n);
  cudaEventRecord(stop);

  cudaDeviceSynchronize();

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  float expected = (float)n;

  print_reduction_results(v, *d_v_r, expected, n, milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(v);
  cudaFree(d_v_r);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Usage: %s <kernel_num> <array_size>\n", argv[0]);
    printf("  0 - divergent\n");
    printf("  1 - bank conflicts\n");
    printf("  2 - no bank conflicts\n");
    printf("  3 - sequential addressing\n");
    printf("  4 - first add during load\n");
    printf("  5 - unroll last warp\n");
    printf("  6 - completely unrolled\n");
    printf("  7 - cooperative groups\n");
    return 1;
  }

  int kernel_num = atoi(argv[1]);
  int n = atoi(argv[2]);
  print_device_info();

  switch(kernel_num) {
    case 0:
      run_divergent_benchmark(n);
      break;
    case 1:
      run_bank_conflicts_benchmark(n);
      break;
    case 2:
      run_no_bank_conflicts_benchmark(n);
      break;
    case 3:
      run_sequential_addressing_benchmark(n);
      break;
    case 4:
      run_first_add_during_load_benchmark(n);
      break;
    case 5:
      run_unroll_last_warp_benchmark(n);
      break;
    case 6:
      run_completely_unrolled_benchmark(n);
      break;
    case 7:
      run_cooperative_groups_benchmark(n);
      break;
    default:
      printf("Invalid kernel number\n");
      return 1;
  }

  return 0;
}

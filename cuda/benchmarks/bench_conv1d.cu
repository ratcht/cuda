#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../kernels/conv1d/runner.cu"
#include "../utils/cuda_utils.h"

void print_conv1d_results(int* h_input, int* h_kernel, int* h_output, int input_size, int kernel_size, int output_size, float ms) {
  if (output_size <= 20) {
    print_vector(h_input, input_size < 20 ? input_size : 20, "Input");
    print_vector(h_kernel, kernel_size, "Kernel");
    print_vector(h_output, output_size, "Output");
  }
  printf("Input size: %d, Kernel size: %d, Output size: %d\n", input_size, kernel_size, output_size);
  printf("Kernel execution time: %.3f ms\n", ms);
}

void run_naive_benchmark(int input_size, int kernel_size) {
  int output_size = input_size - kernel_size + 1;
  int *h_input, *h_kernel, *h_output;
  int *d_input, *d_kernel, *d_output;

  h_input = (int*)malloc(input_size * sizeof(int));
  h_kernel = (int*)malloc(kernel_size * sizeof(int));
  h_output = (int*)malloc(output_size * sizeof(int));

  for (int i = 0; i < input_size; i++) {
    h_input[i] = i % 10;
  }
  for (int i = 0; i < kernel_size; i++) {
    h_kernel[i] = i + 1;
  }

  cudaMalloc(&d_input, input_size * sizeof(int));
  cudaMalloc(&d_kernel, kernel_size * sizeof(int));
  cudaMalloc(&d_output, output_size * sizeof(int));

  cudaMemcpy(d_input, h_input, input_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(int), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_conv1d_naive(d_input, d_kernel, d_output, input_size, kernel_size);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_output, d_output, output_size * sizeof(int), cudaMemcpyDeviceToHost);

  print_conv1d_results(h_input, h_kernel, h_output, input_size, kernel_size, output_size, milliseconds);
  validate_conv1d(h_input, h_kernel, h_output, input_size, kernel_size);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_input);
  cudaFree(d_kernel);
  cudaFree(d_output);
  free(h_input);
  free(h_kernel);
  free(h_output);
}

void run_constant_memory_benchmark(int input_size, int kernel_size) {
  int output_size = input_size - kernel_size + 1;
  int *h_input, *h_kernel, *h_output;
  int *d_input, *d_output;

  h_input = (int*)malloc(input_size * sizeof(int));
  h_kernel = (int*)malloc(kernel_size * sizeof(int));
  h_output = (int*)malloc(output_size * sizeof(int));

  for (int i = 0; i < input_size; i++) {
    h_input[i] = i % 10;
  }
  for (int i = 0; i < kernel_size; i++) {
    h_kernel[i] = i + 1;
  }

  cudaMalloc(&d_input, input_size * sizeof(int));
  cudaMalloc(&d_output, output_size * sizeof(int));

  cudaMemcpy(d_input, h_input, input_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(conv1d_kernel, h_kernel, kernel_size * sizeof(int));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_conv1d_constant_memory(d_input, d_output, input_size, kernel_size);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_output, d_output, output_size * sizeof(int), cudaMemcpyDeviceToHost);

  print_conv1d_results(h_input, h_kernel, h_output, input_size, kernel_size, output_size, milliseconds);
  validate_conv1d(h_input, h_kernel, h_output, input_size, kernel_size);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_input);
  cudaFree(d_output);
  free(h_input);
  free(h_kernel);
  free(h_output);
}

void run_tiled_benchmark(int input_size, int kernel_size) {
  int output_size = input_size - kernel_size + 1;
  int *h_input, *h_kernel, *h_output;
  int *d_input, *d_output;

  h_input = (int*)malloc(input_size * sizeof(int));
  h_kernel = (int*)malloc(kernel_size * sizeof(int));
  h_output = (int*)malloc(output_size * sizeof(int));

  for (int i = 0; i < input_size; i++) {
    h_input[i] = i % 10;
  }
  for (int i = 0; i < kernel_size; i++) {
    h_kernel[i] = i + 1;
  }

  cudaMalloc(&d_input, input_size * sizeof(int));
  cudaMalloc(&d_output, output_size * sizeof(int));

  cudaMemcpy(d_input, h_input, input_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(conv1d_tiled_kernel, h_kernel, kernel_size * sizeof(int));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_conv1d_tiled(d_input, d_output, input_size, kernel_size);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_output, d_output, output_size * sizeof(int), cudaMemcpyDeviceToHost);

  print_conv1d_results(h_input, h_kernel, h_output, input_size, kernel_size, output_size, milliseconds);
  validate_conv1d(h_input, h_kernel, h_output, input_size, kernel_size);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_input);
  cudaFree(d_output);
  free(h_input);
  free(h_kernel);
  free(h_output);
}

void run_tiled_padded_benchmark(int input_size, int kernel_size, int padding) {
  int padded_input_size = input_size + 2 * padding + kernel_size - 1;
  int output_size = input_size + 2 * padding - kernel_size + 1;
  int *h_input, *h_padded_input, *h_kernel, *h_output;
  int *d_padded_input, *d_output;

  h_input = (int*)malloc(input_size * sizeof(int));
  h_padded_input = (int*)malloc(padded_input_size * sizeof(int));
  h_kernel = (int*)malloc(kernel_size * sizeof(int));
  h_output = (int*)malloc(output_size * sizeof(int));

  for (int i = 0; i < input_size; i++) {
    h_input[i] = i % 10;
  }
  for (int i = 0; i < kernel_size; i++) {
    h_kernel[i] = i + 1;
  }

  memset(h_padded_input, 0, padded_input_size * sizeof(int));
  for (int i = 0; i < input_size; i++) {
    h_padded_input[padding + i] = h_input[i];
  }

  cudaMalloc(&d_padded_input, padded_input_size * sizeof(int));
  cudaMalloc(&d_output, output_size * sizeof(int));

  cudaMemcpy(d_padded_input, h_padded_input, padded_input_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(conv1d_padded_kernel, h_kernel, kernel_size * sizeof(int));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_conv1d_tiled_padded(d_padded_input, d_output, output_size, kernel_size);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_output, d_output, output_size * sizeof(int), cudaMemcpyDeviceToHost);

  print_conv1d_results(h_input, h_kernel, h_output, input_size, kernel_size, output_size, milliseconds);
  validate_conv1d_padded(h_input, h_kernel, h_output, input_size, kernel_size, padding);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_padded_input);
  cudaFree(d_output);
  free(h_input);
  free(h_padded_input);
  free(h_kernel);
  free(h_output);
}

void run_strided_padded_benchmark(int input_size, int kernel_size, int padding, int stride) {
  int output_size = (input_size + 2 * padding - kernel_size) / stride + 1;
  int *h_input, *h_kernel, *h_output;
  int *d_input, *d_output;

  h_input = (int*)malloc(input_size * sizeof(int));
  h_kernel = (int*)malloc(kernel_size * sizeof(int));
  h_output = (int*)malloc(output_size * sizeof(int));

  for (int i = 0; i < input_size; i++) {
    h_input[i] = i % 10;
  }
  for (int i = 0; i < kernel_size; i++) {
    h_kernel[i] = i + 1;
  }

  cudaMalloc(&d_input, input_size * sizeof(int));
  cudaMalloc(&d_output, output_size * sizeof(int));

  cudaMemcpy(d_input, h_input, input_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(conv1d_strided_kernel, h_kernel, kernel_size * sizeof(int));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_conv1d_strided_padded(d_input, d_output, input_size, kernel_size, padding, stride, output_size);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_output, d_output, output_size * sizeof(int), cudaMemcpyDeviceToHost);

  print_conv1d_results(h_input, h_kernel, h_output, input_size, kernel_size, output_size, milliseconds);
  validate_conv1d_strided(h_input, h_kernel, h_output, input_size, kernel_size, padding, stride);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_input);
  cudaFree(d_output);
  free(h_input);
  free(h_kernel);
  free(h_output);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Usage: %s <kernel_num> <input_size> [kernel_size] [padding] [stride]\n", argv[0]);
    printf("  0 - naive (requires kernel_size)\n");
    printf("  1 - constant memory (uses kernel_size=4)\n");
    printf("  2 - tiled (uses kernel_size=4)\n");
    printf("  3 - tiled padded (uses kernel_size=7, padding=3)\n");
    printf("  4 - strided padded (requires kernel_size, padding, stride)\n");
    return 1;
  }

  int kernel_num = atoi(argv[1]);
  int input_size = atoi(argv[2]);
  print_device_info();

  switch(kernel_num) {
    case 0:
      if (argc < 4) {
        printf("Naive requires kernel_size\n");
        return 1;
      }
      run_naive_benchmark(input_size, atoi(argv[3]));
      break;
    case 1:
      run_constant_memory_benchmark(input_size, 4);
      break;
    case 2:
      run_tiled_benchmark(input_size, 4);
      break;
    case 3:
      run_tiled_padded_benchmark(input_size, 7, 3);
      break;
    case 4:
      if (argc < 6) {
        printf("Strided padded requires kernel_size, padding, stride\n");
        return 1;
      }
      run_strided_padded_benchmark(input_size, atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
      break;
    default:
      printf("Invalid kernel number\n");
      return 1;
  }

  return 0;
}

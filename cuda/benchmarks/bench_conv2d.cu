#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../kernels/conv2d/runner.cu"
#include "../utils/cuda_utils.h"

void print_conv2d_results(int* h_input, int* h_kernel, int* h_output, int input_width, int input_height, int kernel_width, int kernel_height, int output_width, int output_height, float ms) {
  if (output_width <= 8 && output_height <= 8) {
    print_matrix(h_input, input_height, input_width, "Input");
    print_matrix(h_kernel, kernel_height, kernel_width, "Kernel");
    print_matrix(h_output, output_height, output_width, "Output");
  }
  printf("Input size: %dx%d, Kernel size: %dx%d, Output size: %dx%d\n",
         input_width, input_height, kernel_width, kernel_height, output_width, output_height);
  printf("Kernel execution time: %.3f ms\n", ms);
}

void run_tiled_benchmark(int input_width, int input_height, int kernel_width, int kernel_height, int padding) {
  int padded_input_width = input_width + 2 * padding + kernel_width - 1;
  int padded_input_height = input_height + 2 * padding + kernel_height - 1;
  int output_width = input_width + 2 * padding - kernel_width + 1;
  int output_height = input_height + 2 * padding - kernel_height + 1;

  int *h_input, *h_padded_input, *h_kernel, *h_output;
  int *d_padded_input, *d_output;

  h_input = (int*)malloc(input_width * input_height * sizeof(int));
  h_padded_input = (int*)malloc(padded_input_width * padded_input_height * sizeof(int));
  h_kernel = (int*)malloc(kernel_width * kernel_height * sizeof(int));
  h_output = (int*)malloc(output_width * output_height * sizeof(int));

  for (int i = 0; i < input_width * input_height; i++) {
    h_input[i] = i % 10;
  }
  for (int i = 0; i < kernel_width * kernel_height; i++) {
    h_kernel[i] = i + 1;
  }

  memset(h_padded_input, 0, padded_input_width * padded_input_height * sizeof(int));
  for (int i = 0; i < input_height; i++) {
    for (int j = 0; j < input_width; j++) {
      h_padded_input[(padding + i) * padded_input_width + padding + j] = h_input[i * input_width + j];
    }
  }

  cudaMalloc(&d_padded_input, padded_input_width * padded_input_height * sizeof(int));
  cudaMalloc(&d_output, output_width * output_height * sizeof(int));

  cudaMemcpy(d_padded_input, h_padded_input, padded_input_width * padded_input_height * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(conv2d_kernel, h_kernel, kernel_width * kernel_height * sizeof(int));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_conv2d_tiled(d_padded_input, d_output, padded_input_width, output_width, output_height, kernel_width, kernel_height);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_output, d_output, output_width * output_height * sizeof(int), cudaMemcpyDeviceToHost);

  print_conv2d_results(h_input, h_kernel, h_output, input_width, input_height, kernel_width, kernel_height, output_width, output_height, milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_padded_input);
  cudaFree(d_output);
  free(h_input);
  free(h_padded_input);
  free(h_kernel);
  free(h_output);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Usage: %s <input_width> <input_height> [kernel_width] [kernel_height] [padding]\n", argv[0]);
    printf("  Default kernel: 2x2, padding: 1\n");
    return 1;
  }

  int input_width = atoi(argv[1]);
  int input_height = atoi(argv[2]);
  int kernel_width = (argc >= 4) ? atoi(argv[3]) : 2;
  int kernel_height = (argc >= 5) ? atoi(argv[4]) : 2;
  int padding = (argc >= 6) ? atoi(argv[5]) : 1;

  print_device_info();

  run_tiled_benchmark(input_width, input_height, kernel_width, kernel_height, padding);

  return 0;
}

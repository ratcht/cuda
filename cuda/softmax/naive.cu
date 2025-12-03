#include <math.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <curand.h>

#define ROWS 8
#define COLS 8
#define BYTES sizeof(float) * ROWS * COLS


__global__ void softmax(float *a, float *b) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < ROWS) {
    float x_max = -INFINITY;
    float norm = 0.0f;

    for (int i = 0; i < COLS; i++) {
      x_max = max(x_max, a[row * COLS + i]);
    }

    for (int i = 0; i < COLS; i++) {
      norm += expf(a[row * COLS + i] - x_max);
    }

    for (int i = 0; i < COLS; i++) {
      b[row * COLS + i] = expf(a[row * COLS + i] - x_max)/norm;
    }
  }

}

void printMatrix(const float* M, int rows, int cols, const char* name) {
  printf("%s =\n", name);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      printf("%4f ", M[r * cols + c]);
    }
    printf("\n");
  }
  printf("\n");
}

void save_tensor(const char* filename, float* data, int size) {
  FILE *f = fopen(filename, "wb");
  if (f == NULL) {
    printf("Error opening file %s\n", filename);
    return;
  }
  fwrite(data, sizeof(float), size, f);
  fclose(f);
}

int main() {
  int id = cudaGetDevice(&id);


  // define
  float *h_a, *h_b;
  float *d_a, *d_b;

  h_a = (float*)malloc(BYTES);
  h_b = (float*)malloc(BYTES);

  cudaMalloc(&d_a, BYTES);
  cudaMalloc(&d_b, BYTES);

  // fill in data
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, 462);
  curandGenerateUniform(prng, d_a, ROWS*COLS);

  int NUM_THREADS = 4;
  dim3 NUM_BLOCKS = (ROWS + NUM_THREADS - 1) / NUM_THREADS;

  // ===== START TRACKING
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // LAUNCH KERNEL
  softmax<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b);

  // ===== STOP TRACKING
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_a, d_a, BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b, d_b, BYTES, cudaMemcpyDeviceToHost);

  printMatrix(h_a, ROWS, COLS, "INPUT");
  printMatrix(h_b, ROWS, COLS, "OUTPUT");

  // assert(validate_conv1d(h_input, h_kernel, h_output, INPUT_SIZE, KERNEL_SIZE, PADDING));

  printf("Successful!\n");
  printf("Kernel execution time: %.3f ms\n", milliseconds);

  save_tensor("input.bin", h_a, ROWS*COLS);
  save_tensor("output.bin", h_b, ROWS*COLS);
  printf("Saved tensors to input.bin, output.bin\n");

  // Cleanup events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_a); cudaFree(d_b);
  free(h_a); free(h_b);
}

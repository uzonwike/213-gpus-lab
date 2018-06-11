#include <stdint.h>
#include <stdio.h>

#define N 32
#define THREADS_PER_BLOCK 32

__global__ void reverse(float* x) {
    // Which index of the array should this thread use?
    size_t index = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    if (index < (N / 2)) {
        float temp = x[N-1 - index];
        x[N-1 - index] = x[index];
        x[index] = temp;
    }
}

int main() {
    // Allocate arrays for X and Y on the CPU
    float* cpu_x = (float*)malloc(sizeof(float) * N);

    // Initialize X and Y
    int i;
    for(i=0; i<N; i++) {
        cpu_x[i] = (float) i;
    }

    // Allocate space for X and Y on the GPU
    float* gpu_x;

    if(cudaMalloc(&gpu_x, sizeof(float) * N) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate X array on GPU\n");
        exit(2);
    }

    // Copy the host X and Y arrays to the device X and Y
    // arrays
    if(cudaMemcpy(gpu_x, cpu_x, sizeof(float) * N, cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        fprintf(stderr, "Failed to copy X to the GPU\n");
    }

    // How many blocks should be run, rounding up to
    // include all threads?
    size_t blocks = (N + THREADS_PER_BLOCK - 1) /
        THREADS_PER_BLOCK;

    // Run the saxpy kernel
    reverse<<<blocks, THREADS_PER_BLOCK>>>(gpu_x);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy values from the GPU back to the CPU
    if(cudaMemcpy(cpu_x, gpu_x, sizeof(float) * N, cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
        fprintf(stderr, "Failed to copy X from the GPU\n");
    }

    for(i=0; i<N; i++) {
        printf("%d: %f\n", i, cpu_x[i]);
    }

    cudaFree(gpu_x);
    free(cpu_x);

    return 0;
}
